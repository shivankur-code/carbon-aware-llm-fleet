"""Run google-research/paper-orchestra unmodified, with its Gemini SDK calls served by OpenRouter.

The official pipeline calls Gemini through ``google.genai`` (``utils.gemini_utils.get_genai_client``).
This shim installs a drop-in client whose ``models.generate_content(model, contents, config)`` translates
the request to OpenRouter's OpenAI-compatible API (same Gemini models, e.g. google/gemini-3.1-pro-preview)
and returns an object exposing the attributes the pipeline reads (``.text`` and
``.candidates[0].content.parts[*].inline_data.data``).

Mapping of GenerateContentConfig fields:
  system_instruction -> system message      temperature / max_output_tokens -> same
  tools=[google_search] -> OpenRouter web plugin with the provider's native search (Google Search for Gemini)
  response_modalities=[IMAGE] + image_config.aspect_ratio -> OpenRouter image output
Backend ``PO_BACKEND=bridge`` (no API credit needed): instead of calling OpenRouter, each request is written
to ``$PO_BRIDGE_DIR/<id>/`` (request.json, request.md, attached images) and the call blocks until
``response.txt`` appears there. The responses are produced by an LLM agent (the agent session that runs this
pipeline), acting as the model; requests flagged ``web_search`` are answered with live web search.

Semantic Scholar: requests from utils.scholar_utils are spaced (~1.1 s) and retried on HTTP 429, since the
public endpoint otherwise drops verifications silently; if a title search is still rate-limited after a retry,
it is answered from Crossref's bibliographic search in Semantic Scholar's response shape (logged to
``$PO_S2_FALLBACK_LOG``); the pipeline's own fuzzy-title and cutoff checks still decide what is kept. Every LLM call is logged (model, tokens) to
``$PO_CALL_LOG`` for auditing. No file of the Google repository is modified.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from types import SimpleNamespace

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_log_lock = threading.Lock()


def _or_model(name: str) -> str:
    return name if "/" in name else f"google/{name}"


def _parts_to_content(contents) -> list[dict]:
    """google.genai contents (str | Part | Content | list thereof) -> OpenAI message content list."""
    out: list[dict] = []
    items = contents if isinstance(contents, (list, tuple)) else [contents]
    for it in items:
        if isinstance(it, str):
            out.append({"type": "text", "text": it})
            continue
        parts = getattr(it, "parts", None)
        for p in (parts if parts is not None else [it]):
            if getattr(p, "text", None):
                out.append({"type": "text", "text": p.text})
            blob = getattr(p, "inline_data", None)
            if blob is not None and getattr(blob, "data", None):
                mime = getattr(blob, "mime_type", None) or "image/png"
                b64 = base64.b64encode(blob.data).decode()
                out.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return out


def _system_text(si) -> str | None:
    if si is None:
        return None
    if isinstance(si, str):
        return si
    parts = getattr(si, "parts", None)
    if parts:
        return "\n".join(p.text for p in parts if getattr(p, "text", None))
    return getattr(si, "text", None) or str(si)


_bridge_lock = threading.Lock()
_bridge_counter = [0]


def _bridge_call(body: dict) -> dict:
    """File-queue backend: write the request, wait for response.txt, return an OpenAI-style payload."""
    root = os.environ["PO_BRIDGE_DIR"]
    with _bridge_lock:
        _bridge_counter[0] += 1
        rid = f"{time.strftime('%H%M%S')}_{_bridge_counter[0]:03d}"
    d = os.path.join(root, rid)
    os.makedirs(d, exist_ok=True)
    md, images = [], []
    for m in body["messages"]:
        md.append(f"## {m['role'].upper()} MESSAGE\n")
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for c in content:
            if c["type"] == "text":
                md.append(c["text"])
            else:
                url = c["image_url"]["url"]
                fn = f"image_{len(images) + 1:02d}.png"
                with open(os.path.join(d, fn), "wb") as f:
                    f.write(base64.b64decode(url.split(",", 1)[1]))
                images.append(fn)
                md.append(f"[attached image: {fn}]")
        md.append("")
    meta = dict(id=rid, model=body["model"], web_search="plugins" in body, image_output="modalities" in body,
                temperature=body.get("temperature"), max_tokens=body.get("max_tokens"), images=images)
    req_text = "\n".join(md)
    with open(os.path.join(d, "request.md"), "w") as f:
        f.write(req_text)
    # Replay cache: an identical request (same prompt and images) answered earlier is reused, so a
    # restarted run does not re-ask the same questions.
    import hashlib
    h = hashlib.sha256(req_text.encode()).hexdigest()
    for other in sorted(os.listdir(root)):
        od = os.path.join(root, other)
        if other == rid or not os.path.isdir(od):
            continue
        try:
            if (os.path.exists(os.path.join(od, "response.txt")) and
                    hashlib.sha256(open(os.path.join(od, "request.md")).read().encode()).hexdigest() == h):
                with open(os.path.join(od, "response.txt")) as f:
                    cached = f.read()
                with open(os.path.join(d, "response.txt"), "w") as f:
                    f.write(cached)
                with open(os.path.join(root, "queue.log"), "a") as f:
                    f.write(f"CACHED {rid} <- {other}\n")
                return {"choices": [{"message": {"content": cached}}], "usage": {}}
        except OSError:
            continue
    with open(os.path.join(d, "request.json"), "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(os.path.join(d, "request.json"), os.path.join(d, "request.json"))
    with open(os.path.join(root, "queue.log"), "a") as f:  # one line per new request (watched by the session)
        f.write(f"NEW {rid} model={body['model']} web_search={meta['web_search']} images={len(images)} "
                f"chars={sum(len(x) for x in md)}\n")
    resp = os.path.join(d, "response.txt")
    deadline = time.time() + float(os.environ.get("PO_BRIDGE_TIMEOUT", 6 * 3600))
    while not os.path.exists(resp) or os.path.exists(resp + ".partial"):
        if time.time() > deadline:
            raise TimeoutError(f"bridge request {rid} not answered")
        time.sleep(3)
    time.sleep(1)
    with open(resp) as f:
        text = f.read()
    return {"choices": [{"message": {"content": text}}], "usage": {}}


class _Models:
    def generate_content(self, model, contents, config=None):
        cfg = config
        body: dict = {"model": _or_model(model), "messages": []}
        sys_txt = _system_text(getattr(cfg, "system_instruction", None)) if cfg is not None else None
        if sys_txt:
            body["messages"].append({"role": "system", "content": sys_txt})
        body["messages"].append({"role": "user", "content": _parts_to_content(contents)})
        if cfg is not None:
            if getattr(cfg, "temperature", None) is not None:
                body["temperature"] = cfg.temperature
            if getattr(cfg, "max_output_tokens", None):
                body["max_tokens"] = cfg.max_output_tokens
            tools = getattr(cfg, "tools", None) or []
            if any(getattr(t, "google_search", None) is not None for t in tools):
                body["plugins"] = [{"id": "web", "engine": "native"}]
            mods = [str(m).upper() for m in (getattr(cfg, "response_modalities", None) or [])]
            if any("IMAGE" in m for m in mods):
                body["modalities"] = ["image", "text"]
                ic = getattr(cfg, "image_config", None)
                if ic is not None and getattr(ic, "aspect_ratio", None):
                    body["image_config"] = {"aspect_ratio": ic.aspect_ratio}
        t0 = time.time()
        if os.environ.get("PO_BACKEND") == "bridge":
            data = _bridge_call(body)
        else:
            r = requests.post(OPENROUTER_URL, json=body, timeout=900,
                              headers={"Authorization": "Bearer proxy-injected", "Content-Type": "application/json"})
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code} OpenRouter error: {r.text[:500]}")
            data = r.json()
        msg = data["choices"][0]["message"]
        text = msg.get("content") or ""
        parts = [SimpleNamespace(text=text, inline_data=None)] if text else []
        for img in msg.get("images") or []:
            url = (img.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                parts.append(SimpleNamespace(text=None, inline_data=SimpleNamespace(
                    data=base64.b64decode(url.split(",", 1)[1]), mime_type=url[5:url.index(";")])))
        self._log(body, data, time.time() - t0)
        return SimpleNamespace(text=text or None,
                               candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])

    @staticmethod
    def _log(body, data, secs):
        path = os.environ.get("PO_CALL_LOG")
        if not path:
            return
        u = data.get("usage") or {}
        rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), model=body["model"], seconds=round(secs, 1),
                   web_search="plugins" in body, image_out="modalities" in body,
                   prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                   cost=u.get("cost"))
        with _log_lock, open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


class OpenRouterGenaiClient:
    def __init__(self):
        self.models = _Models()


def _crossref_as_s2(params: dict):
    """Answer an S2 /paper/search title query from Crossref, in S2's response shape (fallback on 429)."""
    import re
    q = params.get("query", "")
    try:
        r = requests.get("https://api.crossref.org/works", timeout=30,
                         params={"query.bibliographic": q, "rows": int(params.get("limit", 3)),
                                 "select": "DOI,title,author,container-title,issued,abstract,is-referenced-by-count"},
                         headers={"User-Agent": "paper-orchestra-shim/0.1"})
        items = r.json()["message"]["items"] if r.status_code == 200 else []
    except Exception:
        items = []
    data = []
    for it in items:
        dp = (it.get("issued") or {}).get("date-parts", [[None]])[0]
        year = dp[0] if dp and dp[0] else None
        date = "%04d-%02d-%02d" % (dp[0], dp[1], dp[2]) if dp and len(dp) == 3 and all(dp) else None
        venue = (it.get("container-title") or [""])[0]
        abstract = it.get("abstract")
        if abstract:
            abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
        data.append({"title": (it.get("title") or [""])[0],
                     "authors": [{"name": f"{a.get('given', '')} {a.get('family', '')}".strip()} for a in it.get("author", [])],
                     "venue": venue, "year": year, "abstract": abstract,
                     "citationCount": it.get("is-referenced-by-count", 0),
                     "journal": {"name": venue} if venue else None, "publicationDate": date,
                     "externalIds": {"DOI": it.get("DOI")}})
    path = os.environ.get("PO_S2_FALLBACK_LOG")
    if path:
        with _log_lock, open(path, "a") as f:
            f.write(json.dumps({"query": q, "source": "crossref", "n": len(data),
                                "top": data[0]["title"] if data else None}) + "\n")
    return SimpleNamespace(status_code=200, json=lambda: {"data": data}, text=json.dumps({"data": data}))


class _ThrottledRequests:
    """Stand-in for the ``requests`` module inside utils.scholar_utils: spaced GETs with 429 retries."""

    def __init__(self, spacing: float = 1.1, retries: int = 8):
        self._lock, self._last, self.spacing, self.retries = threading.Lock(), 0.0, spacing, retries

    def __getattr__(self, name):
        return getattr(requests, name)

    def get(self, *a, **kw):
        kw["timeout"] = max(kw.get("timeout", 0) or 0, 30)
        rsp = None
        for attempt in range(self.retries):
            with self._lock:
                wait = self._last + self.spacing - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            try:
                rsp = requests.get(*a, **kw)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            if rsp.status_code != 429:
                return rsp
            url = a[0] if a else kw.get("url", "")
            if "api.semanticscholar.org" in url and "/paper/search" in url and attempt >= 1:
                return _crossref_as_s2(kw.get("params") or {})
            time.sleep(2)
        return rsp


def install() -> None:
    import utils.gemini_utils as gu
    import utils.scholar_utils as su
    client = OpenRouterGenaiClient()
    gu.get_genai_client = lambda: client
    gu._genai_client = client
    su.requests = _ThrottledRequests()
