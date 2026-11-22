"""Invoke the official google-research/paper-orchestra CLI with the OpenRouter shim installed.

    python paper/google_po/run_paper_orchestra.py --po-repo <path to google-research/paper-orchestra> \
        -- --raw_materials_dir ... --latex_template_dir ... --output_dir ... [other official CLI args]
"""
import argparse
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--po-repo", required=True)
args, rest = ap.parse_known_args()
rest = [x for x in rest if x != "--"]
repo = Path(args.po_repo).resolve()
os.chdir(repo)
sys.path.insert(0, str(repo))
ca = "/root/.ccr/ca-bundle.crt"
if os.path.exists(ca):  # TLS through the session's egress proxy
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
    os.environ.setdefault("SSL_CERT_FILE", ca)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import openrouter_shim  # noqa: E402

openrouter_shim.install()
import paper_writing_cli  # noqa: E402  (official entry point, unmodified)

sys.argv = ["paper_writing_cli.py", *rest]
paper_writing_cli.main()
