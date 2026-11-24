#!/usr/bin/env bash
# Builds the preprint (non-anonymous, no line numbers, header "Preprint") from the paper-orchestra output.
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import re
s = open("paper_iclr_source.tex").read()
s = s.replace(r"\usepackage{iclr2025_conference,times}", r"\usepackage{iclr2025_conference,times}" "\n" r"\iclrfinalcopy")
s = re.sub(r"\\author\{.*?\n\}\n", lambda m: open("authors.tex").read() + "\n", s, count=1, flags=re.S)
s = s.replace(r"\maketitle", r"\maketitle" "\n" r"\lhead{Preprint}", 1)
# Figure 7 has been regenerated with the correct ktCO2e axis; drop the caption's workaround note.
s = re.sub(r" \\textbf\{Unit note:\}.*?ktCO\$_2\$e, as in Table~\\ref\{tab:pareto_app\}\.", "", s, flags=re.S)
open("paper.tex", "w").write(s)
PY
pdflatex -interaction=nonstopmode paper.tex >/dev/null || true
bibtex paper >/dev/null || true
pdflatex -interaction=nonstopmode paper.tex >/dev/null || true
pdflatex -interaction=nonstopmode paper.tex >/dev/null || true
grep -E "^! " paper.log || echo "no LaTeX errors"
