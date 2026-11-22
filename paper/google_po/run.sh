#!/usr/bin/env bash
# Write the paper with Google's official paper-orchestra (google-research/paper-orchestra), unmodified,
# with its Gemini calls served by OpenRouter (see openrouter_shim.py). Needs OpenRouter credit (~USD 10-20).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PO_DIR="${PO_DIR:-$HERE/.paper-orchestra}"
PO_COMMIT="d9bae997e793a015ed38bb1605be9eb58c00284f"   # commit inspected for the shim
if [ ! -d "$PO_DIR/.git" ]; then git clone -q https://github.com/google-research/paper-orchestra "$PO_DIR"; fi
git -C "$PO_DIR" fetch -q --depth 50 origin || true
git -C "$PO_DIR" checkout -q "$PO_COMMIT"
VENV="${VENV:-$HERE/.venv}"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -r "$PO_DIR/requirements.txt"
OUT="${OUT:-$HERE/output}"
mkdir -p "$OUT"
export PO_CALL_LOG="$OUT/llm_calls.jsonl"
PYTHONUNBUFFERED=1 "$VENV/bin/python" "$HERE/run_paper_orchestra.py" --po-repo "$PO_DIR" -- \
  --raw_materials_dir "$HERE/raw_materials" \
  --latex_template_dir "$PO_DIR/templates/iclr2025" \
  --output_dir "$OUT" \
  --idea_filename idea.md \
  --experimental_log_filename experimental_log.md \
  --research_cutoff 2026-10 \
  --use_plotting false 2>&1 | tee "$OUT/run.log"
