"""Phase-3 study on the reference fleet scenario (configs/scenarios/fleet.yaml, LP at fleet scale).

Subcommands (each writes results/p3_<name>.csv and, where relevant, paper/figures/p3_<name>.png):

  ladder     E0..E3g -> X, Y_naive, Y, Z, Zg; per-region breakdown; Figure 1 (fleet)
  mc         Monte-Carlo replay of E0 / E2a / E2b: LC SLO-violation probability, Poisson vs correlated
  reliab     reused-part failure multiplier sweep -> where does reuse stop paying? (break-even)
  cascade    old-GPU BATCH efficiency sweep -> Z and Zg; the break-even C2 must beat
  headroom   LC failure buffer x utilization cap -> reused share and Y (finding F7)
  policy     reuse embodied accounting: cutoff / residual / full
  fuzzy      alpha-cuts over (failure rates, embodied carbon, CXL slowdown)
  pareto     carbon vs cost epsilon-constraint front
  all        everything above

Usage: python experiments/p3_study.py all
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from carbonsched.fuzzy import FuzzyGroup, alpha_cuts, overrides_for  # noqa: E402
from carbonsched.grid import load_regions  # noqa: E402
from carbonsched.hardware import load_catalog  # noqa: E402
from carbonsched.milp import LifecycleModel, Scenario, pareto  # noqa: E402
from carbonsched.montecarlo import replay_lc, summarize  # noqa: E402
from carbonsched.study import run_ladder  # noqa: E402

CFG = ROOT / "configs"
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
SCEN = CFG / "scenarios" / "fleet.yaml"
COLORS = {"blue": "#4C78A8", "orange": "#F58518", "red": "#E45756", "teal": "#72B7B2", "green": "#54A24B",
          "grey": "#9D9D9D", "purple": "#B279A2"}


def base_scenario(path: Path = SCEN) -> dict:
    return yaml.safe_load(path.read_text())


def load(overrides: dict | None = None):
    cat = load_catalog(CFG, overrides)
    regions = load_regions(CFG, cat.registry, overrides)
    return cat, regions


def save(df: pd.DataFrame, name: str) -> None:
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / f"p3_{name}.csv", index=False)


def fmt(h: dict) -> str:
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in h.items())


# ---------------------------------------------------------------- ladder
def cmd_ladder(args) -> None:
    cat, regions = load()
    base = base_scenario()
    h, table, runs = run_ladder(base, cat, regions)
    print(fmt(h))
    (RES / "p3_ladder_headline.json").write_text(json.dumps(h, indent=2))
    table.reset_index(names="run").to_csv(RES / "p3_ladder_runs.csv", index=False)
    # per-region single-region ladders (same fleet, one grid at a time) for the region-dependence claim
    rows = [dict(region="all (3-region fleet)", **{k: v for k, v in h.items() if isinstance(v, float)})]
    for r in base["regions"]:
        b = copy.deepcopy(base)
        b["regions"] = [r]
        b["demand"]["LC"] = {r: base["demand"]["LC"][r]}
        share = base["demand"]["LC"][r]["start"] / sum(v["start"] for v in base["demand"]["LC"].values())
        b["demand"]["BATCH"] = dict(base["demand"]["BATCH"], start=base["demand"]["BATCH"]["start"] * share)
        b["initial_fleet"] = [f for f in base["initial_fleet"] if f["region"] == r]
        b["pools"] = {c: dict(p, initial=p["initial"] * share, external_per_q=p["external_per_q"] * share)
                      for c, p in base["pools"].items()}
        hr, _, _ = run_ladder(b, cat, regions, rungs=("E0_det", "E0", "E1", "E2a", "E2b", "E3", "E0r", "E4"))
        rows.append(dict(region=r, **{k: v for k, v in hr.items() if isinstance(v, float)}))
        print(r, fmt(hr))
    save(pd.DataFrame(rows), "ladder_by_region")
    figure1(table, h, pd.DataFrame(rows))


def figure1(table: pd.DataFrame, h: dict, by_region: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    order = [("E0", "No reuse"), ("E2a", "Reuse plan\nblind to failures"), ("E2b", "Reuse,\nreliability-aware"),
             ("E3", "+ cascading"), ("E3g", "+ geographic\ncascading")]
    order = [o for o in order if o[0] in table.index]
    base = float(table.loc["E0", "carbon_kg"])
    comps = [("embodied_kg", "Embodied", COLORS["blue"]), ("operational_kg", "Operational", COLORS["orange"]),
             ("repair_kg", "Spares + logistics", COLORS["red"]), ("lease_kg", "Leased capacity", COLORS["teal"]),
             ("ship_kg", "Shipping", COLORS["green"])]
    width = 0.15
    for j, (col, name, c) in enumerate(comps):
        vals = [(float(table.loc[k, col]) - float(table.loc["E0", col])) / 1000 for k, _ in order]
        ax.bar(np.arange(len(order)) + (j - 2) * width, vals, width, color=c, label=name)
    tot = [(float(table.loc[k, "carbon_kg"]) - base) / 1000 for k, _ in order]
    ax.scatter(range(len(order)), tot, color="k", zorder=3, label="Net change", s=18)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(order)), [o[1] for o in order], fontsize=8)
    ax.set_ylabel("Change vs no-reuse baseline (tCO2e, 6 yr)")
    ax.set_title(f"Fleet (3 grids): X={h['X']:.2f}%, Y={h['Y']:.2f}% (naive {h['Y_naive']:.2f}%), "
                 f"Z={h['Z_pp']:+.2f}pp", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    br = by_region[by_region.region != "all (3-region fleet)"]
    x = np.arange(len(br))
    for j, (col, name, c) in enumerate([("X", "X: reuse, failures ignored", COLORS["blue"]),
                                         ("Y_naive", "Plan blind to failures", COLORS["red"]),
                                         ("Y", "Y: reliability-aware", COLORS["orange"]),
                                         ("saving_E3", "Y+Z: + cascading", COLORS["green"])]):
        ax.bar(x + (j - 1.5) * 0.2, br[col], 0.2, color=c, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{r} grid" for r in br.region])
    ax.set_ylabel("Lifecycle carbon saving vs no reuse (%)")
    ax.set_title("By grid (single-region fleets)", fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("Parameters provisional: GPU throughput/power and CXL slowdown await C2/C3 measurements", fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig1_fleet.png", dpi=150)


# ---------------------------------------------------------------- Monte Carlo
def cmd_mc(args) -> None:
    cat, regions = load()
    base = base_scenario()
    h, table, runs = run_ladder(base, cat, regions, rungs=("E0", "E1", "E2a", "E2b"))
    rows = []
    for run in ("E0", "E2a", "E2b"):
        no_reuse = [s for s in base["buyable"] if not cat.skus[s].is_reuse]
        b = dict(copy.deepcopy(base), reliability=True, allow_cascade=False)
        if run == "E0":
            b["buyable"] = no_reuse
        model = LifecycleModel(cat, regions, Scenario(**b))
        for disp in (None, 2.0, 0.5):
            df = replay_lc(model, runs[run], n_samples=args.samples, dispersion=disp)
            s = summarize(df)
            rows.append(dict(run=run, correlation="independent (Poisson)" if disp is None else f"NB k={disp}", **s))
            print(run, rows[-1]["correlation"], fmt(s))
    save(pd.DataFrame(rows), "mc")


# ---------------------------------------------------------------- reliability break-even
def cmd_reliab(args) -> None:
    base = base_scenario()
    rows = []
    for m in [0.5, 1, 2, 3, 5, 8, 12, 20]:
        cat, regions = load({"components.reused_part_afr_multiplier": m})
        h, _, _ = run_ladder(base, cat, regions, rungs=("E0_det", "E0", "E1", "E2a", "E2b"))
        rows.append(dict(multiplier=m, **{k: v for k, v in h.items() if isinstance(v, float)}))
        print(m, fmt(h))
    df = pd.DataFrame(rows)
    save(df, "reliab")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(df.multiplier, df.X, "o-", color=COLORS["blue"], label="X (failures ignored)")
    ax.plot(df.multiplier, df.Y, "o-", color=COLORS["orange"], label="Y (reliability-aware plan)")
    ax.plot(df.multiplier, df.Y_naive, "o-", color=COLORS["red"], label="Plan blind to failures")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Reused-part failure-rate multiplier (beyond age effect)")
    ax.set_ylabel("Fleet carbon saving vs no reuse (%)")
    ax.legend(fontsize=8)
    ax.set_title("How unreliable can reused parts be?", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "p3_reliab.png", dpi=150)


# ---------------------------------------------------------------- cascade break-even
def cmd_cascade(args) -> None:
    base = base_scenario()
    rows = []
    for ratio in [0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]:
        cat, regions = load({"skus.skus.old_gpu.throughput.BATCH": ratio})
        h, _, runs = run_ladder(base, cat, regions, rungs=("E0", "E2b", "E3", "E3g"))
        old = cat.skus["old_gpu"]
        new = cat.skus["new_gpu"]
        eff = (ratio / old.power_w(0.8)) / (1.0 / new.power_w(0.8))
        casc = sum(v for k, v in runs["E3g"].values["casc"].items() if v > 1e-6)
        rows.append(dict(old_batch_throughput=ratio, old_vs_new_work_per_joule=eff, cascaded_nodes=casc,
                         **{k: v for k, v in h.items() if isinstance(v, float)}))
        print(ratio, round(eff, 2), fmt(h))
    df = pd.DataFrame(rows)
    save(df, "cascade")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(df.old_vs_new_work_per_joule, df.Z_pp, "o-", color=COLORS["green"], label="Z: in-place cascading")
    ax.plot(df.old_vs_new_work_per_joule, df.Zg_pp, "s--", color=COLORS["purple"], label="Zg: + geographic cascading")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("A100-class vs H100-class BATCH work per joule (C2 measures this)")
    ax.set_ylabel("Extra saving from cascading (pp)")
    ax.legend(fontsize=8)
    ax.set_title("When does cascading old GPUs pay?", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "p3_cascade.png", dpi=150)


# ---------------------------------------------------------------- headroom / utilization
def cmd_headroom(args) -> None:
    cat, regions = load()
    base = base_scenario()
    rows = []
    for buf in [1.0, 3.0, 6.0, 10.0]:
        for util in [0.5, 0.8]:
            b = dict(copy.deepcopy(base), lc_failure_buffer=buf, util_cap=util)
            h, _, runs = run_ladder(b, cat, regions, rungs=("E0", "E2b"))
            f = runs["E2b"].fleet
            share = f[f.sku == "green_gpu"].n.sum() / f.n.sum()
            rows.append(dict(lc_failure_buffer=buf, util_cap=util, green_node_quarter_share=share,
                             **{k: v for k, v in h.items() if isinstance(v, float)}))
            print(buf, util, round(share, 3), fmt(h))
    save(pd.DataFrame(rows), "headroom")


# ---------------------------------------------------------------- accounting policy
def cmd_policy(args) -> None:
    cat, regions = load()
    base = base_scenario()
    rows = []
    for pol in ("cutoff", "residual", "full"):
        for acc in ("amortized", "upfront"):
            b = dict(copy.deepcopy(base), embodied_policy=pol, embodied_accounting=acc)
            h, _, _ = run_ladder(b, cat, regions, rungs=("E0_det", "E0", "E1", "E2a", "E2b"))
            rows.append(dict(policy=pol, accounting=acc, **{k: v for k, v in h.items() if isinstance(v, float)}))
            print(pol, acc, fmt(h))
    save(pd.DataFrame(rows), "policy")


# ---------------------------------------------------------------- fuzzy
def cmd_fuzzy(args) -> None:
    base = base_scenario()
    cat0, _ = load()
    reg = dict(cat0.registry)
    load_regions(CFG, reg)
    groups = [
        FuzzyGroup("failure", lambda p: p.endswith(".failure.afr"), 0.5, 1.0, 3.0, clip=(1e-5, 0.9)),
        FuzzyGroup("embodied", lambda p: p.endswith("embodied_kg"), 0.5, 1.0, 2.0),
        FuzzyGroup("cxl", lambda p: ".cxl_slowdown." in p, 0.85 / 0.95, 1.0, 1.0 / 0.95, clip=(0.0, 1.0)),
    ]
    log = []

    def evaluate(factors):
        ov = overrides_for(reg, groups, factors)
        cat, regions = load(ov)
        h, _, _ = run_ladder(base, cat, regions, rungs=("E0", "E2b"))
        log.append(dict(**factors, Y=h["Y"]))
        return h["Y"]

    df = alpha_cuts(evaluate, groups)
    print(df.to_string())
    save(df, "fuzzy")
    save(pd.DataFrame(log), "fuzzy_vertices")


# ---------------------------------------------------------------- pareto
def cmd_pareto(args) -> None:
    cat, regions = load()
    sc = Scenario(**base_scenario())
    front = pareto(LifecycleModel(cat, regions, sc), "carbon", "cost", n_points=8)
    save(front, "pareto")
    print(front[["carbon_kg", "cost_usd", "embodied_kg", "operational_kg"]].round(-3).to_string())
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(front.cost_usd / 1e9, front.carbon_kg / 1e6, "o-", color=COLORS["blue"])
    ax.set_xlabel("6-yr cost (billion USD)")
    ax.set_ylabel("6-yr lifecycle carbon (MtCO2e)")
    ax.set_title("Carbon-cost Pareto front (epsilon-constraint)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "p3_pareto.png", dpi=150)


CMDS = dict(ladder=cmd_ladder, mc=cmd_mc, reliab=cmd_reliab, cascade=cmd_cascade, headroom=cmd_headroom,
            policy=cmd_policy, fuzzy=cmd_fuzzy, pareto=cmd_pareto)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=[*CMDS, "all"])
    ap.add_argument("--samples", type=int, default=20000)
    a = ap.parse_args()
    for name in (CMDS if a.cmd == "all" else [a.cmd]):
        print(f"\n===== {name} =====")
        CMDS[name](a)


if __name__ == "__main__":
    main()
