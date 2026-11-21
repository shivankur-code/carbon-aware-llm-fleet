"""Turn GPU measurements into lifecycle-model parameters (configs/measured.yaml).

Inputs (any subset):
  --c2-old / --c2-new   c2_serving_bench JSONs for the cascade-candidate GPU (e.g. A100) and the
                        current GPU (e.g. H100), same model and settings.
  --c3-local / --c3-remote  c3_kv_offload_bench JSONs with host KV buffers on local vs remote/CXL memory.

Outputs: configs/measured.yaml with
  skus.skus.old_gpu.throughput.{LC,BATCH}      old/new per-GPU work ratio per tier
  skus.skus.{new,old}_gpu.power_{idle,peak}_w  8x GPU board power + host overhead
  skus.skus.green_gpu.cxl_slowdown.{LC,BATCH}  offloaded-KV tokens/s remote / local, blended by
                                               the fraction of work that touches offloaded KV
Every value records its source file, so paper/parameters.md shows provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def per_gpu(run: dict, key: str) -> float:
    return run["summary"][key] / max(1, run.get("tp", 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2-old")
    ap.add_argument("--c2-new")
    ap.add_argument("--c3-local")
    ap.add_argument("--c3-remote")
    ap.add_argument("--gpus-per-host", type=int, default=8)
    ap.add_argument("--host-idle-w", type=float, default=800, help="CPU/DRAM/NIC/fans at idle (not visible to NVML)")
    ap.add_argument("--host-peak-w", type=float, default=2000)
    ap.add_argument("--offload-fraction", type=float, default=0.3,
                    help="share of decode work that reads KV from host memory on a GreenSKU-style host")
    ap.add_argument("--out", default="configs/measured.yaml")
    a = ap.parse_args()
    m: dict[str, dict] = {}
    g = a.gpus_per_host

    if a.c2_old and a.c2_new:
        old, new = json.loads(Path(a.c2_old).read_text()), json.loads(Path(a.c2_new).read_text())
        src = f"{Path(a.c2_old).name} vs {Path(a.c2_new).name}"
        lc = per_gpu(old, "lc_out_tok_per_s") / per_gpu(new, "lc_out_tok_per_s")
        bt = per_gpu(old, "batch_out_tok_per_s") / per_gpu(new, "batch_out_tok_per_s")
        m["skus.skus.old_gpu.throughput.LC"] = dict(value=round(lc, 4), source=f"C2 LC goodput-at-SLO ratio, {src}")
        m["skus.skus.old_gpu.throughput.BATCH"] = dict(value=round(bt, 4), source=f"C2 batch throughput ratio, {src}")
        for sku, run, f in (("new_gpu", new, a.c2_new), ("old_gpu", old, a.c2_old)):
            tp = max(1, run.get("tp", 1))
            s = run["summary"]
            idle = (s.get("idle_power_w") or 0) / tp
            peak = max(s.get("batch_mean_power_w") or 0, s.get("lc_mean_power_w") or 0) / tp
            if idle and peak:
                m[f"skus.skus.{sku}.power_idle_w"] = dict(value=round(g * idle + a.host_idle_w),
                                                          source=f"C2 NVML idle x{g} + {a.host_idle_w:g} W host, {Path(f).name}")
                m[f"skus.skus.{sku}.power_peak_w"] = dict(value=round(g * peak + a.host_peak_w),
                                                          source=f"C2 NVML busy x{g} + {a.host_peak_w:g} W host, {Path(f).name}")
        print(f"old/new per-GPU work ratio: LC {lc:.3f}, BATCH {bt:.3f}  (cascade hypothesis: BATCH > LC -> {bt > lc})")

    if a.c3_local and a.c3_remote:
        loc, rem = json.loads(Path(a.c3_local).read_text()), json.loads(Path(a.c3_remote).read_text())
        key = lambda r: (r["batch"], r["context"])  # noqa: E731
        lo = {key(r): r["tok_per_s"] for r in loc["decode"] if r.get("cache") == "offloaded" and "tok_per_s" in r}
        re_ = {key(r): r["tok_per_s"] for r in rem["decode"] if r.get("cache") == "offloaded" and "tok_per_s" in r}
        common = sorted(set(lo) & set(re_))
        if common:
            small = [re_[k] / lo[k] for k in common if k[0] <= 8]      # latency-bound regime ~ LC
            large = [re_[k] / lo[k] for k in common if k[0] > 8] or small
            f = a.offload_fraction
            for tier, vals in (("LC", small or large), ("BATCH", large)):
                ratio = sum(vals) / len(vals)
                m[f"skus.skus.green_gpu.cxl_slowdown.{tier}"] = dict(
                    value=round(1 - f * (1 - ratio), 4),
                    source=f"C3a offloaded-KV tok/s remote/local = {ratio:.3f}, blended with offload fraction {f}")
            print("C3a remote/local offloaded-KV throughput:", {k: round(re_[k] / lo[k], 3) for k in common})

    Path(a.out).write_text(yaml.safe_dump({"measured": m}, sort_keys=True))
    print(f"wrote {a.out} with {len(m)} measured parameters")


if __name__ == "__main__":
    main()
