# GPU experiments (C2, C3)

These measurements decide the sign of the main result. The Phase 1–3 sensitivity studies show that the lifecycle verdict depends mostly on:
- **how efficient older GPUs are per tier**, which decides whether cascading pays (C2);
- **the slowdown from keeping KV cache in reused, CXL-attached memory** (C3a);
- **whether that memory's errors are tolerable** (C3b).

Every script writes JSON or CSV to `gpu/results/<LABEL>/`. `gpu/ingest.py` turns those files into `configs/measured.yaml`, and the lifecycle model uses the measured values automatically.

## What to run where

| Machine | Purpose | Command |
|---|---|---|
| New-generation GPU (H100 / H200 / L40S / L4) | `new_gpu` baseline | `LABEL=H100-SXM bash gpu/run_all.sh` |
| Older GPU (A100 / A10 / V100 / T4) | cascade candidate `old_gpu` | `LABEL=A100-SXM bash gpu/run_all.sh` |
| Any 2-socket server with a GPU (or a CXL memory expander) | CXL proxy for C3a | runs inside `run_all.sh` automatically when `numactl` shows 2+ NUMA nodes |

**One GPU of each generation is enough.** Use the **same model** on both. The default `Qwen/Qwen2.5-7B-Instruct` fits on a single A100-40G or H100, which keeps the comparison per-GPU and free of tensor-parallel effects. For smaller cards (T4, A10, L4), use `MODEL=Qwen/Qwen2.5-1.5B-Instruct` on **both** machines.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r gpu/requirements.txt

# about 5 minutes: check everything works
QUICK=1 LABEL=H100-SXM bash gpu/run_all.sh

# full run: about 1-2 h per GPU type
LABEL=H100-SXM bash gpu/run_all.sh
git add gpu/results && git commit -m "GPU results: H100-SXM" && git push
```

Then, on any machine:

```bash
python gpu/ingest.py \
  --c2-old gpu/results/A100-SXM/c2_A100-SXM_*.json \
  --c2-new gpu/results/H100-SXM/c2_H100-SXM_*.json \
  --c3-local gpu/results/H100-SXM/c3offload_H100-SXM_local_*.json \
  --c3-remote gpu/results/H100-SXM/c3offload_H100-SXM_remote_*.json
python experiments/p3_study.py all   # re-runs everything with measured parameters
```

## The experiments

### C2: serving efficiency per GPU generation × tier (`c2_serving_bench.py`)
- Launches vLLM, or attaches to a running server with `--base-url`.
- Sends random token-id prompts with lognormal lengths (mean 1024 in / 256 out). This needs no dataset download, and the lengths are exact.
- **BATCH:** all requests released at once. Records output tokens/s and tokens per joule.
- **LC:** Poisson arrivals over a range of rates, as fractions of the saturation rate. The SLO is a time-to-first-token (TTFT) of ≤ 1 s and a time per output token (TPOT) of ≤ 50 ms, met by at least 90% of requests. The highest rate meeting it gives LC capacity and LC tokens/J.
- **Idle:** board power with the model loaded.
- **Hypothesis tested:** old/new efficiency is higher for BATCH than for LC. Phase 3's break-even analysis (`results/p3_cascade.csv`) says cascading only saves carbon when **A100 batch work per joule reaches about 0.9× that of the H100 or more**.

### C3a: KV cache in host memory, local vs remote / CXL (`c3_kv_offload_bench.py`)
- Pinned host↔GPU bandwidth for KV-block-sized copies.
- Hugging Face `generate` with a GPU-resident cache vs `cache_implementation="offloaded"`, across batch sizes and context lengths, with NVML power.
- Run under `numactl --membind=<local>` and `--membind=<remote>`. Remote-NUMA DRAM is the standard CXL emulation. A real CXL Type-3 expander appears as a CPU-less NUMA node; set `REMOTE_NODE` to it.
- **Output:** `cxl_slowdown`, the throughput of the offloaded cache on remote memory divided by that on local memory.
- *Optional:* vLLM's own KV-offload path, e.g. LMCache or the KV-offloading connector in recent vLLM; check your version's flag. Run it through `c2_serving_bench.py --launch-prefix "numactl --membind=1" --vllm-extra-args "<offload flags>"`.

### C3b: KV-cache fault injection (`c3_kv_faults.py`)
- Prefill a context from WikiText-2, flip bits in the cached K/V at bit-error rates from 1e-10 to 1e-3, then teacher-force the continuation.
- Reports the perplexity ratio, top-1 agreement with the clean run, and the fraction of KV blocks recomputed.
- **Protection schemes:** `none`; `secded` (64-bit SECDED ECC as in server DIMMs: 1 flip corrected, 2 detected → block recomputed, 3 or more silent); `checksum` (any flip in a 16-token block → recompute).
- `--bit-classes all,sign,exponent,mantissa` shows which bits matter.
- **Converting field error rates to BER:** a DIMM with FIT rate *F* (failures per 10⁹ device-hours per Mbit) gives BER ≈ *F* × 10⁻⁹ × *t* / 10⁶ per bit for data resident *t* hours. KV caches usually live for seconds to minutes; prefix caches can live for hours.
- **Offline smoke test**, with no model download (the numbers are meaningless):
  `python gpu/c3_kv_faults.py --model tiny-random --device cpu --text-file README.md --samples 1 --context 256`

## Notes
- NVML measures **GPU board power only**. `ingest.py` adds host power as a parameter (`--host-idle-w`, `--host-peak-w`). If your servers expose a BMC or PDU power reading, record it as well.
- Use the same vLLM and driver versions on both GPU generations where possible. The scripts record versions in each result file.
- `gpu/tests/mock_vllm.py` is a CPU mock server used to test the C2 client:
  `python gpu/tests/mock_vllm.py 8765 & python gpu/c2_serving_bench.py --base-url http://localhost:8765 --model mock --idle-seconds 1`
