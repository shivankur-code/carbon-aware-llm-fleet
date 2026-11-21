"""One-at-a-time sensitivity (tornado) of the node-level GreenSKU saving.

For every parameter with a [low, high] range, recompute the green_gpu vs new_gpu
saving (lifecycle gCO2e per unit work, failures counted) at both ends. Tells us
which parameters must be pinned down (verified or measured on GPUs) first.

Usage: python experiments/p1_tornado.py [--region mixed] [--tier LC] [--policy cutoff]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from carbonsched.grid import load_regions  # noqa: E402
from carbonsched.hardware import load_catalog  # noqa: E402
from p1_node_lifecycle import node_lifecycle  # noqa: E402


def saving(overrides, region, tier, policy, years=6, util=0.6):
    cat = load_catalog(ROOT / "configs", overrides)
    reg = load_regions(ROOT / "configs", None, overrides)[region]
    n_q = years * 4
    g = node_lifecycle(cat.skus["green_gpu"], reg, n_q, util, tier, policy, True)["g_per_work"]
    n = node_lifecycle(cat.skus["new_gpu"], reg, n_q, util, tier, policy, True)["g_per_work"]
    return 100 * (1 - g / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="mixed")
    ap.add_argument("--tier", default="LC")
    ap.add_argument("--policy", default="cutoff")
    a = ap.parse_args()
    cat = load_catalog(ROOT / "configs")
    registry = dict(cat.registry)
    load_regions(ROOT / "configs", registry)
    base = saving({}, a.region, a.tier, a.policy)
    rows = []
    for path, p in registry.items():
        if p.low is None or p.high is None or p.low == p.high:
            continue
        lo = saving({path: p.low}, a.region, a.tier, a.policy)
        hi = saving({path: p.high}, a.region, a.tier, a.policy)
        rows.append(dict(param=path, status=p.status, low=p.low, high=p.high, saving_at_low=lo,
                         saving_at_high=hi, swing=abs(hi - lo)))
    df = pd.DataFrame(rows).sort_values("swing", ascending=False)
    tag = f"{a.region}_{a.tier}_{a.policy}"
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / f"p1_tornado_{tag}.csv", index=False)
    print(f"base saving = {base:.2f}%  ({tag})")
    print(df.head(15).round(2).to_string(index=False))

    top = df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(top.itertuples()):
        ax.barh(i, r.saving_at_low - base, left=base, color="#4C78A8")
        ax.barh(i, r.saving_at_high - base, left=base, color="#F58518")
    ax.set_yticks(range(len(top)), [p.replace("components.components.", "").replace("skus.skus.", "")
                                    .replace("regions.regions.", "") for p in top.param], fontsize=7)
    ax.axvline(base, color="k", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("GreenSKU-style node saving vs new node (% lifecycle gCO2e / work)")
    ax.set_title(f"Sensitivity (blue = param at low, orange = at high), {tag}")
    fig.tight_layout()
    out = ROOT / "paper" / "figures" / f"p1_tornado_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
