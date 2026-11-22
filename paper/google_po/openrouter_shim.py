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
Semantic Scholar: requests from utils.scholar_utils are spaced (~1.1 s) and retried on HTTP 429, since the
public endpoint otherwise drops verifications silently. Every LLM call is logged (model, tokens) to
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
            time.sleep(3 * (attempt + 1))
        return rsp


def install() -> None:
    import utils.gemini_utils as gu
    import utils.scholar_utils as su
    client = OpenRouterGenaiClient()
    gu.get_genai_client = lambda: client
    gu._genai_client = client
    su.requests = _ThrottledRequests()
