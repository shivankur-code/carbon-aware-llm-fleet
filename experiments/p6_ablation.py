"""Phase-6 ablation: which part of the GreenSKU-style host actually saves carbon?

``green_gpu`` bundles three changes vs ``new_gpu``: an energy-efficient CPU, reused DRAM behind CXL (which
costs the CXL throughput multiplier and extra platform embodied carbon for the expanders), and reused SSDs
(no throughput cost). A reviewer of the paper-orchestra draft observed that, by the paper's own numbers, most
of the embodied saving comes from the SSDs while the whole slowdown comes from the DRAM. This script
unbundles them into variants that differ from ``green_gpu`` in exactly one lever and re-runs the node-level
study and the fleet LP.

  green_cpu   efficient CPU only (all parts new, local DRAM, no CXL)            -> not a reuse SKU
  green_ssd   efficient CPU + reused SSDs, new *local* DRAM (no CXL, no slowdown)
  green_dram  efficient CPU + reused DRAM over CXL, new SSDs
  green_gpu   efficient CPU + reused DRAM over CXL + reused SSDs (the bundle used in the paper)

Host power deltas are kept equal to green_gpu's for all variants (CXL expander power is not modelled
separately - stated as an assumption). Usage: python experiments/p6_ablation.py [node|fleet|cxl|all]
"""
from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from carbonsched.grid import load_regions  # noqa: E402
from carbonsched.hardware import load_catalog  # noqa: E402
from carbonsched.study import run_ladder  # noqa: E402
from p1_node_lifecycle import node_lifecycle  # noqa: E402

CFG = ROOT / "configs"
OUT = ROOT / "results" / "p6"
VARIANTS = ("green_cpu", "green_ssd", "green_dram", "green_gpu")


def add_variants(cat):
    new, g = cat.skus["new_gpu"], cat.skus["green_gpu"]
    new_slot = {s.component.name: s for s in new.slots}
    g_slot = {s.component.name: s for s in g.slots}
    cxl_platform = g.platform_embodied_kg - new.platform_embodied_kg   # CXL controllers / expanders

    def mk(name, reuse_dram, reuse_ssd, desc):
        slots = tuple(
            (g_slot[c] if (c == "dram_dimm" and reuse_dram) or (c == "ssd" and reuse_ssd) else
             g_slot[c] if c == "cpu_efficient" else new_slot[c])
            for c in [s.component.name for s in g.slots])
        return replace(g, name=name, slots=slots, description=desc,
                       platform_embodied_kg=new.platform_embodied_kg + (cxl_platform if reuse_dram else 0.0),
                       cxl_slowdown=dict(g.cxl_slowdown) if reuse_dram else {},
                       capex_usd=g.capex_usd)

    cat.skus["green_cpu"] = mk("green_cpu", False, False, "efficient CPU, all-new local parts")
    cat.skus["green_ssd"] = mk("green_ssd", False, True, "efficient CPU, reused SSDs, new local DRAM")
    cat.skus["green_dram"] = mk("green_dram", True, False, "efficient CPU, reused DRAM over CXL, new SSDs")
    return cat


def load(overrides=None):
    cat = add_variants(load_catalog(CFG, overrides))
    return cat, load_regions(CFG, cat.registry, overrides)


def cmd_node():
    cat, regions = load()
    rows = [node_lifecycle(cat.skus[s], regions[r], 24, 0.6, t, p, True)
            for s in ("new_gpu",) + VARIANTS for r in regions for t in ("LC", "BATCH") for p in ("cutoff", "residual")]
    df = pd.DataFrame(rows)
    base = df[df.sku == "new_gpu"].set_index(["region", "tier", "policy"]).g_per_work
    df["saving_pct"] = [100 * (1 - g / base[(r, t, p)]) for g, r, t, p in zip(df.g_per_work, df.region, df.tier, df.policy)]
    df.to_csv(OUT / "node_ablation.csv", index=False)
    for p in ("cutoff", "residual"):
        d = df[df.policy == p]
        piv = d.pivot_table(index="sku", columns=["region", "tier"], values="saving_pct").loc[list(VARIANTS)]
        emb = d[(d.region == "mixed") & (d.tier == "LC")].set_index("sku").embodied_kg
        print(f"\n[{p}] node-level saving per unit of work vs new_gpu (%), failures counted, 6 y, 60% util")
        print(piv.round(2).to_string())
        print("embodied kgCO2e per node:", emb.round(0).to_dict())
    return df


def fleet_base(buyable):
    b = yaml.safe_load((CFG / "scenarios" / "fleet.yaml").read_text())
    b["skus"] = sorted(set(b["skus"]) | set(buyable))
    b["buyable"] = list(buyable)
    for s in buyable:
        b["max_age_q"].setdefault(s, b["max_age_q"]["green_gpu"])
    return b


def cmd_fleet(overrides=None, tag=""):
    cat, regions = load(overrides)
    rows = []
    for name, buy in [("bundle (paper)", ["new_gpu", "green_gpu"]),
                      ("cpu only", ["new_gpu", "green_cpu"]),
                      ("ssd reuse only", ["new_gpu", "green_ssd"]),
                      ("dram/CXL reuse only", ["new_gpu", "green_dram"]),
                      ("planner picks", ["new_gpu", *VARIANTS])]:
        b = fleet_base(buy)
        # E0 always = new_gpu only, so every row shares the same baseline
        b0 = dict(b, buyable=["new_gpu"])
        h0, _, r0 = run_ladder(b0, cat, regions, rungs=("E0",))
        _, _, r = run_ladder(b, cat, regions, rungs=("E0", "E2b", "E3g"))
        c0 = r0["E0"].carbon_kg
        mix = {}
        for (s, *_), v in r["E3g"].values["buy"].items():
            mix[s] = mix.get(s, 0.0) + v
        rows.append(dict(design=name, buyable="+".join(buy), Y=100 * (1 - r["E2b"].carbon_kg / c0),
                         saving_E3g=100 * (1 - r["E3g"].carbon_kg / c0),
                         E0_kt=c0 / 1e6, E2b_kt=r["E2b"].carbon_kg / 1e6,
                         bought=" ".join(f"{k}:{v:.0f}" for k, v in sorted(mix.items()) if v > 0.5)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"fleet_ablation{tag}.csv", index=False)
    print(df.round(3).to_string(index=False))
    return df


def cmd_cxl():
    """Sweep the LC CXL multiplier: at what slowdown does DRAM reuse over CXL stop paying at fleet scale?"""
    rows = []
    for m in (0.95, 0.97, 0.98, 0.99, 1.0):
        df = cmd_fleet({"skus.skus.green_gpu.cxl_slowdown.LC": m}, tag=f"_cxl{m}")
        for _, r in df.iterrows():
            rows.append(dict(cxl_LC=m, design=r.design, Y=r.Y, saving_E3g=r.saving_E3g, bought=r.bought))
    pd.DataFrame(rows).to_csv(OUT / "fleet_ablation_cxl_sweep.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("node", "all"):
        cmd_node()
    if what in ("fleet", "all"):
        cmd_fleet()
    if what in ("cxl", "all"):
        cmd_cxl()
