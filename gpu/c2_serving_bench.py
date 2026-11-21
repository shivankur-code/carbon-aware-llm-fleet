"""C2 - Measured value of cascading: LLM-serving efficiency per GPU generation x service tier.

Runs (or attaches to) a vLLM OpenAI-compatible server and measures, with NVML power:

  * LC tier    - Poisson arrivals at a sweep of request rates; TTFT / TPOT per request;
                 SLO attainment; the highest rate meeting the SLO = LC capacity, and the
                 output tokens per joule at that rate.
  * BATCH tier - all requests released at once (throughput-bound); output tokens/s and tokens/J.
  * Idle       - board power with the model loaded and no traffic.

The cascade hypothesis to test: (old GPU / new GPU) efficiency is HIGHER for BATCH than for LC.

Prompts are random token-id sequences of exact length (no dataset download; vLLM accepts
token-id prompts), with lognormal input/output lengths.

Example (one 8-GPU host or a single GPU; repeat on each GPU generation):
    python gpu/c2_serving_bench.py --model Qwen/Qwen2.5-7B-Instruct --tp 1 --out gpu/results
    python gpu/c2_serving_bench.py --base-url http://localhost:8000 --model <served-name> ...   # existing server
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.power import PowerSampler, gpu_inventory  # noqa: E402


@dataclass
class ReqResult:
    ok: bool
    input_len: int
    output_len: int
    ttft: float = math.nan
    tpot: float = math.nan
    latency: float = math.nan
    error: str = ""


@dataclass
class RateResult:
    rate: float                 # requests/s (inf = batch)
    n: int
    duration_s: float
    req_per_s: float
    out_tok_per_s: float
    in_tok_per_s: float
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    tpot_p50: float
    tpot_p90: float
    tpot_p99: float
    slo_attainment: float
    goodput_out_tok_per_s: float
    energy_j: float | None
    mean_power_w: float | None
    out_tok_per_j: float | None
    errors: int
    extra: dict = field(default_factory=dict)


def lengths(rng, n, mean, cv, lo, hi):
    sigma = math.sqrt(math.log(1 + cv * cv))
    mu = math.log(mean) - sigma * sigma / 2
    return np.clip(rng.lognormal(mu, sigma, n).round().astype(int), lo, hi)


async def one_request(session, url, model, prompt_ids, max_tokens, timeout) -> ReqResult:
    import aiohttp
    body = dict(model=model, prompt=prompt_ids, max_tokens=int(max_tokens), temperature=0.0, stream=True,
                ignore_eos=True, stream_options={"include_usage": True})
    t0 = time.perf_counter()
    first = None
    last = None
    n_out = 0
    try:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                return ReqResult(False, len(prompt_ids), 0, error=f"HTTP {resp.status}: {(await resp.text())[:200]}")
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                msg = json.loads(data)
                now = time.perf_counter()
                if msg.get("choices") and msg["choices"][0].get("text"):
                    if first is None:
                        first = now
                    last = now
                if msg.get("usage"):
                    n_out = msg["usage"].get("completion_tokens", n_out)
    except Exception as e:  # noqa: BLE001
        return ReqResult(False, len(prompt_ids), 0, error=repr(e)[:200])
    end = time.perf_counter()
    if first is None:
        return ReqResult(False, len(prompt_ids), 0, error="no tokens")
    n_out = n_out or max_tokens
    tpot = (last - first) / (n_out - 1) if n_out > 1 else 0.0
    return ReqResult(True, len(prompt_ids), n_out, ttft=first - t0, tpot=tpot, latency=end - t0)


async def run_rate(url, model, prompts, out_lens, rate, timeout, max_concurrency, seed):
    import aiohttp
    rng = np.random.default_rng(seed)
    conn = aiohttp.TCPConnector(limit=max_concurrency or 0)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        t_start = time.perf_counter()
        for p, o in zip(prompts, out_lens):
            tasks.append(asyncio.create_task(one_request(session, url, model, p, o, timeout)))
            if math.isfinite(rate):
                await asyncio.sleep(rng.exponential(1.0 / rate))
        res = await asyncio.gather(*tasks)
        return res, time.perf_counter() - t_start


def summarize(rate, res: list[ReqResult], dur, ps: PowerSampler | None, ttft_slo, tpot_slo) -> RateResult:
    ok = [r for r in res if r.ok]
    ttft = np.array([r.ttft for r in ok]) if ok else np.array([math.nan])
    tpot = np.array([r.tpot for r in ok]) if ok else np.array([math.nan])
    meets = [r for r in ok if r.ttft <= ttft_slo and r.tpot <= tpot_slo]
    out_tok = sum(r.output_len for r in ok)
    e = ps.energy_j if ps else None
    q = lambda a, p: float(np.nanpercentile(a, p))  # noqa: E731
    return RateResult(
        rate=rate, n=len(res), duration_s=dur, req_per_s=len(ok) / dur, out_tok_per_s=out_tok / dur,
        in_tok_per_s=sum(r.input_len for r in ok) / dur,
        ttft_p50=q(ttft, 50), ttft_p90=q(ttft, 90), ttft_p99=q(ttft, 99),
        tpot_p50=q(tpot, 50), tpot_p90=q(tpot, 90), tpot_p99=q(tpot, 99),
        slo_attainment=len(meets) / max(1, len(res)), goodput_out_tok_per_s=sum(r.output_len for r in meets) / dur,
        energy_j=e, mean_power_w=ps.mean_w if ps else None, out_tok_per_j=(out_tok / e) if e else None,
        errors=len(res) - len(ok), extra=dict(first_errors=[r.error for r in res if not r.ok][:3]))


def wait_ready(base, timeout=1800):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return json.loads(r.read())
        except Exception:  # noqa: BLE001
            time.sleep(5)
    raise TimeoutError("server not ready")


def free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--base-url", default=None, help="use an already-running server instead of launching vLLM")
    ap.add_argument("--vllm-extra-args", default="", help='e.g. "--max-model-len 8192 --gpu-memory-utilization 0.9"')
    ap.add_argument("--launch-prefix", default="", help='e.g. "numactl --membind=1" (CXL/remote-NUMA proxy for C3)')
    ap.add_argument("--vocab-size", type=int, default=32000, help="random prompt ids are drawn below this")
    ap.add_argument("--input-mean", type=int, default=1024)
    ap.add_argument("--output-mean", type=int, default=256)
    ap.add_argument("--len-cv", type=float, default=0.6)
    ap.add_argument("--max-input", type=int, default=4096)
    ap.add_argument("--max-output", type=int, default=1024)
    ap.add_argument("--num-requests", type=int, default=300)
    ap.add_argument("--rates", default="auto", help='comma list of req/s, or "auto" (geometric search)')
    ap.add_argument("--batch-requests", type=int, default=1000)
    ap.add_argument("--ttft-slo", type=float, default=1.0, help="seconds")
    ap.add_argument("--tpot-slo", type=float, default=0.05, help="seconds per output token (~20 tok/s per user)")
    ap.add_argument("--slo-target", type=float, default=0.9, help="fraction of requests that must meet the SLO")
    ap.add_argument("--idle-seconds", type=float, default=60)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--label", default="", help="free-form tag, e.g. 'A100-80G-SXM' or 'H100-SXM'")
    ap.add_argument("--out", default="gpu/results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    proc = None
    base = a.base_url
    if base is None:
        port = free_port()
        cmd = (shlex.split(a.launch_prefix) + [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", a.model, "--tensor-parallel-size", str(a.tp), "--port", str(port),
               "--disable-log-requests"] + shlex.split(a.vllm_extra_args))
        print("launching:", " ".join(cmd), flush=True)
        log = open(Path(a.out).mkdir(parents=True, exist_ok=True) or Path(a.out) / f"vllm_{port}.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        base = f"http://localhost:{port}"
    try:
        models = wait_ready(base)
        served = models["data"][0]["id"] if models.get("data") else a.model
        url = f"{base}/v1/completions"
        rng = np.random.default_rng(a.seed)
        gpus = gpu_inventory()
        gpu_ids = list(range(len(gpus))) if gpus else None
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            gpu_ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip().isdigit()]

        def workload(n, seed):
            r = np.random.default_rng(seed)
            ins = lengths(r, n, a.input_mean, a.len_cv, 16, a.max_input)
            outs = lengths(r, n, a.output_mean, a.len_cv, 8, a.max_output)
            prompts = [r.integers(100, a.vocab_size - 100, size=int(k)).tolist() for k in ins]
            return prompts, outs

        # warm-up
        p, o = workload(16, 999)
        asyncio.run(run_rate(url, served, p, o, float("inf"), a.timeout, 0, 0))

        # idle power
        idle_w = None
        if a.idle_seconds > 0:
            with PowerSampler(gpus=gpu_ids) as ps:
                time.sleep(a.idle_seconds)
            idle_w = ps.mean_w
            print(f"idle power: {idle_w} W", flush=True)

        def measure(rate, n, seed):
            p, o = workload(n, seed)
            with PowerSampler(gpus=gpu_ids) as ps:
                res, dur = asyncio.run(run_rate(url, served, p, o, rate, a.timeout, 0, seed))
            r = summarize(rate, res, dur, ps, a.ttft_slo, a.tpot_slo)
            print(f"rate={rate:>6}: {r.req_per_s:.2f} req/s, {r.out_tok_per_s:.0f} out tok/s, "
                  f"TTFT p90 {r.ttft_p90:.2f}s, TPOT p90 {r.tpot_p90 * 1000:.0f}ms, SLO {r.slo_attainment:.2f}, "
                  f"{r.mean_power_w or float('nan'):.0f} W, {r.out_tok_per_j or float('nan'):.3f} tok/J", flush=True)
            return r

        # BATCH tier first: gives the saturation rate, used to place the LC sweep
        batch = measure(float("inf"), a.batch_requests, a.seed + 1)
        lc: list[RateResult] = []
        if a.rates == "auto":
            sat = max(batch.req_per_s, 0.05)
            grid = [sat * f for f in (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9)]
        else:
            grid = [float(x) for x in a.rates.split(",")]
        for i, rate in enumerate(grid):
            r = measure(rate, a.num_requests, a.seed + 10 + i)
            lc.append(r)
            if a.rates == "auto" and r.slo_attainment < a.slo_target * 0.5:
                break  # well past the knee
        ok = [r for r in lc if r.slo_attainment >= a.slo_target]
        best = max(ok, key=lambda r: r.req_per_s) if ok else None
        summary = dict(
            lc_capacity_req_per_s=best.req_per_s if best else 0.0,
            lc_out_tok_per_s=best.out_tok_per_s if best else 0.0,
            lc_goodput_out_tok_per_s=best.goodput_out_tok_per_s if best else 0.0,
            lc_out_tok_per_j=best.out_tok_per_j if best else None,
            lc_mean_power_w=best.mean_power_w if best else None,
            batch_out_tok_per_s=batch.out_tok_per_s, batch_out_tok_per_j=batch.out_tok_per_j,
            batch_mean_power_w=batch.mean_power_w, idle_power_w=idle_w,
        )
        out = dict(
            kind="c2_serving", label=a.label, model=a.model, tp=a.tp, gpus=gpus, host=platform.node(),
            time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), args=vars(a),
            vllm_version=_version("vllm"), summary=summary, batch=asdict(batch), lc=[asdict(r) for r in lc])
        Path(a.out).mkdir(parents=True, exist_ok=True)
        gname = (gpus[0]["name"] if gpus else "unknown").replace(" ", "_")
        fn = Path(a.out) / f"c2_{a.label or gname}_{a.model.split('/')[-1]}_tp{a.tp}.json"
        fn.write_text(json.dumps(out, indent=2, default=float))
        print("\nSUMMARY", json.dumps(summary, indent=2, default=float))
        print("wrote", fn)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


def _version(pkg):
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    main()
