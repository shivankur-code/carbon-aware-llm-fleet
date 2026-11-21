"""E0-E3 pipeline that produces the brief's X / Y / Z for one scenario.

  E0   no reuse: only new hardware can be bought, no cascading (deterministic and with failures)
  E1   reuse allowed, failures ignored                  -> X = 1 - C(E1) / C(E0, det)
  E2a  E1's plan replayed with failures (naive plan)     -> Y_naive = 1 - C(E2a) / C(E0, fail)
  E2b  reuse allowed, reliability-aware, no cascading     -> Y = 1 - C(E2b) / C(E0, fail)
  E3   E2b + cascading LC -> BATCH                        -> Z = (1 - C(E3) / C(E0, fail)) - Y   [pp restored]

Usage: python experiments/p2_pipeline.py --scenario configs/scenarios/pilot.yaml [--region clean]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from carbonsched.grid import load_regions  # noqa: E402
from carbonsched.hardware import load_catalog  # noqa: E402
from carbonsched.milp import LifecycleModel, Scenario  # noqa: E402


def retarget_region(sc_kwargs: dict, region: str) -> dict:
    """Rename the scenario's single region (demand, initial fleet) to ``region``."""
    old = sc_kwargs["regions"][0]
    sc_kwargs["regions"] = [region]
    sc_kwargs["demand"]["LC"] = {region: sc_kwargs["demand"]["LC"][old]}
    sc_kwargs["initial_fleet"] = [dict(f, region=region) for f in sc_kwargs.get("initial_fleet", [])]
    return sc_kwargs


def run(scenario_path: Path, region: str | None, time_limit: float, scale: float = 1.0, lp: bool = False):
    import yaml
    base = yaml.safe_load(scenario_path.read_text())
    if region:
        base = retarget_region(base, region)
    base["scale"] = scale
    if lp:
        base["integer"] = False
    cat = load_catalog(ROOT / "configs")
    regions = load_regions(ROOT / "configs")
    no_reuse = [s for s in base["buyable"] if not cat.skus[s].is_reuse]

    import copy

    def mk(**kw):
        d = copy.deepcopy(base)
        d.update(kw)
        return Scenario(**d)

    runs = {}
    t0 = time.time()
    runs["E0_det"] = LifecycleModel(cat, regions, mk(buyable=no_reuse, allow_cascade=False, reliability=False)).solve(time_limit=time_limit)
    runs["E0"] = LifecycleModel(cat, regions, mk(buyable=no_reuse, allow_cascade=False, reliability=True)).solve(time_limit=time_limit)
    runs["E1"] = LifecycleModel(cat, regions, mk(allow_cascade=False, reliability=False)).solve(time_limit=time_limit)
    replay = LifecycleModel(cat, regions, mk(allow_cascade=False, reliability=True))
    runs["E2a"] = replay.solve(fix={"buy": runs["E1"].values["buy"], "ret": runs["E1"].values["ret"]}, time_limit=time_limit)
    runs["E2b"] = LifecycleModel(cat, regions, mk(allow_cascade=False, reliability=True)).solve(time_limit=time_limit)
    runs["E3"] = LifecycleModel(cat, regions, mk(allow_cascade=True, reliability=True)).solve(time_limit=time_limit)
    elapsed = time.time() - t0
    for k, s in runs.items():
        if s.status != "Optimal":
            raise RuntimeError(f"{k}: {s.status}")

    C = {k: s.carbon_kg for k, s in runs.items()}
    X = 100 * (1 - C["E1"] / C["E0_det"])
    Y_naive = 100 * (1 - C["E2a"] / C["E0"])
    Y = 100 * (1 - C["E2b"] / C["E0"])
    S3 = 100 * (1 - C["E3"] / C["E0"])
    Z = S3 - Y
    headline = dict(scenario=base["name"], region=base["regions"][0], scale=scale,
                    formulation="LP relaxation" if lp else "MILP", X=X, Y_naive=Y_naive, Y=Y,
                    saving_with_cascade=S3, Z_restored_pp=Z, solve_seconds=elapsed,
                    time_limited_runs=[k for k, s in runs.items() if s.time_limited])
    table = pd.DataFrame({k: dict(s.totals, status=s.status) for k, s in runs.items()}).T
    return headline, table, runs


def figure1(table: pd.DataFrame, headline: dict, out: Path) -> None:
    """Left: stacked lifecycle carbon per run (axis from 0, nothing hidden).
    Right: saving vs the matching no-reuse baseline (deterministic for E1, with failures otherwise)."""
    order = [("E0_det", "No reuse\n(no failures)"), ("E1", "Reuse\n(no failures)"),
             ("E0", "No reuse\n(failures)"), ("E2a", "Reuse plan\nblind to failures"),
             ("E2b", "Reuse,\nreliability-aware"), ("E3", "+ cascading")]
    parts = [("embodied_kg", "Embodied", "#4C78A8"), ("operational_kg", "Operational", "#F58518"),
             ("repair_kg", "Spares + logistics", "#E45756"), ("lease_kg", "Leased capacity", "#72B7B2"),
             ("ship_kg", "Shipping", "#54A24B")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 2]})
    for i, (k, _) in enumerate(order):
        bottom = 0.0
        for col, name, color in parts:
            v = float(table.loc[k, col]) / 1000
            ax.bar(i, v, bottom=bottom, color=color, label=name if i == 0 else None, width=0.65)
            bottom += v
        ax.text(i, bottom, f"{bottom:,.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(order)), [o[1] for o in order], fontsize=7)
    ax.set_ylabel("Lifecycle carbon over horizon (tCO2e)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend(fontsize=7, ncol=5, loc="upper left", frameon=False)
    bars = [("X: reuse,\nno failures", headline["X"], "#4C78A8"),
            ("Y naive:\nplan blind\nto failures", headline["Y_naive"], "#E45756"),
            ("Y: reliability-\naware plan", headline["Y"], "#F58518"),
            ("Y + Z:\n+ cascading", headline["saving_with_cascade"], "#54A24B")]
    for i, (lab, v, c) in enumerate(bars):
        ax2.bar(i, v, color=c, width=0.6)
        ax2.text(i, v, f"{v:.2f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=7)
    ax2.set_ylabel("Carbon saving vs no-reuse baseline (%)")
    fig.suptitle(f"{headline['scenario']} scenario, {headline['region']} grid "
                 "(parameters provisional; GPU measurements pending)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(ROOT / "configs/scenarios/pilot.yaml"))
    ap.add_argument("--region", default=None)
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--replot", action="store_true", help="redraw Figure 1 from saved results, no solving")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply fleet size (demand, initial fleet, pools)")
    ap.add_argument("--lp", action="store_true", help="LP relaxation (accurate for large fleets, fast)")
    a = ap.parse_args()
    if a.replot:
        import yaml
        name = yaml.safe_load(Path(a.scenario).read_text())["name"]
        for hp in sorted((ROOT / "results").glob(f"p2_{name}_*_headline.json")):
            headline = json.loads(hp.read_text())
            tag = hp.name[len("p2_"):-len("_headline.json")]
            table = pd.read_csv(ROOT / "results" / f"p2_{tag}_runs.csv", index_col=0)
            figure1(table, headline, ROOT / "paper" / "figures" / f"fig1_prototype_{tag}.png")
            print("replotted", tag)
        return
    headline, table, runs = run(Path(a.scenario), a.region, a.time_limit, a.scale, a.lp)
    tag = f"{headline['scenario']}_{headline['region']}" + (f"_x{a.scale:g}" if a.scale != 1 else "") + ("_lp" if a.lp else "")
    (ROOT / "results").mkdir(exist_ok=True)
    table.to_csv(ROOT / "results" / f"p2_{tag}_runs.csv")
    (ROOT / "results" / f"p2_{tag}_headline.json").write_text(json.dumps(headline, indent=2))
    for k in ("E1", "E2b", "E3"):
        runs[k].decisions.assign(key=runs[k].decisions.key.astype(str)).to_csv(
            ROOT / "results" / f"p2_{tag}_{k}_decisions.csv", index=False)
    figure1(table, headline, ROOT / "paper" / "figures" / f"fig1_prototype_{tag}.png")
    print(json.dumps(headline, indent=2))
    cols = ["carbon_kg", "embodied_kg", "operational_kg", "repair_kg", "lease_kg", "ship_kg", "cost_usd", "shortfall"]
    print(table[cols].astype(float).round(0).to_string())


if __name__ == "__main__":
    main()
