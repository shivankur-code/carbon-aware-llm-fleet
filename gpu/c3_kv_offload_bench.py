"""C3a - What does it cost to keep the KV cache in (reused, CXL-attached) host memory?

Two measurements, run under a NUMA memory binding so that host buffers land either on the
local socket's DRAM or on a *remote* NUMA node. Remote-NUMA DRAM is the standard emulation of
CXL-attached memory in the literature (similar added latency, lower bandwidth); if you have a
real CXL Type-3 expander, it shows up as a CPU-less NUMA node - bind to that instead.

  1. Transfer microbenchmark: pinned host <-> GPU bandwidth for KV-block-sized copies.
  2. Decode throughput with Hugging Face transformers: GPU-resident KV cache vs offloaded KV
     cache (``cache_implementation="offloaded"``), at several batch sizes / context lengths,
     with NVML power.

Run it once per binding; ``gpu/run_all.sh`` does this automatically:
    numactl --cpunodebind=0 --membind=0 python gpu/c3_kv_offload_bench.py --label local  ...
    numactl --cpunodebind=0 --membind=1 python gpu/c3_kv_offload_bench.py --label remote ...
cxl_slowdown = tokens/s(offloaded, remote) / tokens/s(offloaded, local) is what the lifecycle model needs.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.power import PowerSampler, gpu_inventory  # noqa: E402


def numa_info() -> dict:
    try:
        out = subprocess.run(["numactl", "--show"], capture_output=True, text=True, timeout=5).stdout
        return dict(numactl_show=out.strip())
    except Exception:  # noqa: BLE001
        return {}


def bandwidth(sizes_mb=(1, 4, 16, 64, 256, 1024), reps=20) -> list[dict]:
    rows = []
    for mb in sizes_mb:
        n = mb * 2**20
        host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        host.fill_(1)  # first touch -> pages placed per the active NUMA policy
        dev = torch.empty(n, dtype=torch.uint8, device="cuda")
        for name, src, dst in (("H2D", host, dev), ("D2H", dev, host)):
            dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / reps
            rows.append(dict(direction=name, size_mb=mb, gb_per_s=n / dt / 1e9, latency_us=dt * 1e6))
        del host, dev
    return rows


@torch.no_grad()
def decode_tput(model, vocab, batch, ctx, new_tokens, cache_impl, gpu_ids) -> dict:
    ids = torch.randint(100, vocab - 100, (batch, ctx), device="cuda")
    kw = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, pad_token_id=0)
    if cache_impl != "dynamic":
        kw["cache_implementation"] = cache_impl
    model.generate(ids[:, :64], max_new_tokens=4, do_sample=False, pad_token_id=0)  # warm-up
    torch.cuda.synchronize()
    try:
        with PowerSampler(gpus=gpu_ids) as ps:
            t0 = time.perf_counter()
            model.generate(ids, **kw)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return dict(cache=cache_impl, batch=batch, context=ctx, error=repr(e)[:300])
    toks = batch * new_tokens
    return dict(cache=cache_impl, batch=batch, context=ctx, new_tokens=new_tokens, seconds=dt,
                tok_per_s=toks / dt, mean_power_w=ps.mean_w, tok_per_j=(toks / ps.energy_j) if ps.energy_j else None,
                peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--contexts", default="2048,8192")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--caches", default="dynamic,offloaded")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--label", default="local", help="local | remote | cxl ... (which memory the host buffers sit on)")
    ap.add_argument("--out", default="gpu/results")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("C3a needs a CUDA GPU.")
    gpu_ids = [torch.cuda.current_device()]
    res = dict(kind="c3_kv_offload", label=a.label, host=platform.node(), gpus=gpu_inventory(), numa=numa_info(),
               torch=torch.__version__, args=vars(a), time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    res["bandwidth"] = bandwidth()
    for r in res["bandwidth"]:
        print(f"{r['direction']} {r['size_mb']:>5} MB: {r['gb_per_s']:.1f} GB/s", flush=True)
    res["decode"] = []
    if not a.skip_decode:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).cuda().eval()
        vocab = model.config.vocab_size
        for ctx in [int(x) for x in a.contexts.split(",")]:
            for b in [int(x) for x in a.batches.split(",")]:
                for c in a.caches.split(","):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    r = decode_tput(model, vocab, b, ctx, a.new_tokens, c, gpu_ids)
                    res["decode"].append(r)
                    print(json.dumps(r), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fn = Path(a.out) / f"c3offload_{a.label}_{torch.cuda.get_device_name(0).replace(' ', '_')}_{a.model.split('/')[-1]}.json"
    fn.write_text(json.dumps(res, indent=2, default=float))
    print("wrote", fn)


if __name__ == "__main__":
    main()
