#!/usr/bin/env bash
# Run every GPU experiment this repo needs on ONE machine, then commit the results.
# Usage:  LABEL=H100-SXM bash gpu/run_all.sh          (on the new-generation GPU box)
#         LABEL=A100-SXM bash gpu/run_all.sh          (on the cascade-candidate box)
# Optional env: MODEL (default Qwen/Qwen2.5-7B-Instruct), TP (1), QUICK=1 (short smoke run),
#               SKIP_C2=1 / SKIP_C3A=1 / SKIP_C3B=1, REMOTE_NODE (NUMA node for the CXL proxy).
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL="${LABEL:?set LABEL, e.g. LABEL=H100-SXM}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TP="${TP:-1}"
OUT="gpu/results/${LABEL}"
mkdir -p "$OUT"
nvidia-smi > "$OUT/nvidia-smi.txt" || true
(numactl --hardware || true) > "$OUT/numa.txt" 2>&1
python -c "import vllm,torch,transformers;print('vllm',vllm.__version__,'torch',torch.__version__,'transformers',transformers.__version__)" > "$OUT/versions.txt" 2>&1 || true

if [[ "${QUICK:-0}" == "1" ]]; then
  C2ARGS="--num-requests 40 --batch-requests 120 --idle-seconds 10"
  C3BARGS="--samples 2 --trials 1 --context 512 --continuation 64 --bers 0,1e-6,1e-4"
  C3AARGS="--batches 1,8 --contexts 1024 --new-tokens 32"
else
  C2ARGS=""; C3BARGS=""; C3AARGS=""
fi

# Use the model's real vocab size for random prompts
VOCAB=$(python - <<PY
from transformers import AutoConfig
print(AutoConfig.from_pretrained("$MODEL").vocab_size)
PY
)

if [[ "${SKIP_C2:-0}" != "1" ]]; then
  echo "== C2: serving efficiency (LC vs BATCH) =="
  python gpu/c2_serving_bench.py --model "$MODEL" --tp "$TP" --vocab-size "$VOCAB" --label "$LABEL" --out "$OUT" $C2ARGS
fi

if [[ "${SKIP_C3A:-0}" != "1" ]]; then
  echo "== C3a: KV offload, local vs remote NUMA memory =="
  NODES=$(numactl --hardware 2>/dev/null | awk '/available:/{print $2}' || echo 1)
  if command -v numactl >/dev/null && [[ "${NODES:-1}" -ge 2 ]]; then
    REMOTE="${REMOTE_NODE:-1}"
    numactl --cpunodebind=0 --membind=0 python gpu/c3_kv_offload_bench.py --model "$MODEL" --label "${LABEL}_local" --out "$OUT" $C3AARGS
    numactl --cpunodebind=0 --membind="$REMOTE" python gpu/c3_kv_offload_bench.py --model "$MODEL" --label "${LABEL}_remote" --out "$OUT" $C3AARGS
  else
    echo "only one NUMA node (or no numactl): running local only; cxl_slowdown cannot be measured on this host"
    python gpu/c3_kv_offload_bench.py --model "$MODEL" --label "${LABEL}_local" --out "$OUT" $C3AARGS
  fi
fi

if [[ "${SKIP_C3B:-0}" != "1" ]]; then
  echo "== C3b: KV-cache fault injection =="
  python gpu/c3_kv_faults.py --model "$MODEL" --label "$LABEL" --out "$OUT" $C3BARGS
fi

echo
echo "Done. Results in $OUT. To share them:"
echo "  git add $OUT && git commit -m \"GPU results: $LABEL\" && git push"
