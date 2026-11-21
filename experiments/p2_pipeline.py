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


def run(scenario_path: Path, region: str | None, time_limit: float):
    import yaml
    base = yaml.safe_load(scenario_path.read_text())
    if region:
        base = retarget_region(base, region)
    cat = load_catalog(ROOT / "configs")
    regions = load_regions(ROOT / "configs")
    no_reuse = [s for s in base["buyable"] if not cat.skus[s].is_reuse]

    def mk(**kw):
        d = dict(base)
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
    headline = dict(scenario=base["name"], region=base["regions"][0], X=X, Y_naive=Y_naive, Y=Y,
                    saving_with_cascade=S3, Z_restored_pp=Z, solve_seconds=elapsed)
    table = pd.DataFrame({k: dict(s.totals, status=s.status) for k, s in runs.items()}).T
    return headline, table, runs


def figure1(table: pd.DataFrame, headline: dict, out: Path) -> None:
    order = [("E0", "No reuse\n(failures counted)"), ("E1", "Reuse,\nfailures ignored"),
             ("E2a", "Same plan,\nfailures replayed"), ("E2b", "Reuse,\nreliability-aware"),
             ("E3", "+ cascading\nto BATCH")]
    parts = [("embodied_kg", "Embodied", "#4C78A8"), ("operational_kg", "Operational", "#F58518"),
             ("repair_kg", "Spares + logistics", "#E45756"), ("lease_kg", "Leased capacity", "#72B7B2"),
             ("ship_kg", "Shipping", "#54A24B")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (k, label) in enumerate(order):
        bottom = 0.0
        for col, name, color in parts:
            v = float(table.loc[k, col]) / 1000
            ax.bar(i, v, bottom=bottom, color=color, label=name if i == 0 else None, width=0.65)
            bottom += v
        ax.text(i, bottom * 1.005, f"{bottom:,.0f} t", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(order)), [o[1] for o in order], fontsize=8)
    ax.set_ylabel("Lifecycle carbon over horizon (tCO2e)")
    lo = min(float(table.loc[k, "carbon_kg"]) for k, _ in order) / 1000
    ax.set_ylim(lo * 0.9, None)
    ax.set_title(f"{headline['scenario']} / {headline['region']}:  X={headline['X']:.1f}%  "
                 f"Y={headline['Y']:.1f}% (naive {headline['Y_naive']:.1f}%)  Z={headline['Z_restored_pp']:+.1f} pp",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(ROOT / "configs/scenarios/pilot.yaml"))
    ap.add_argument("--region", default=None)
    ap.add_argument("--time-limit", type=float, default=300)
    a = ap.parse_args()
    headline, table, runs = run(Path(a.scenario), a.region, a.time_limit)
    tag = f"{headline['scenario']}_{headline['region']}"
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
