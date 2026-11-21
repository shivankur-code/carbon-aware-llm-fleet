"""Phase 4: does reuse survive reliability when the LC SLO must be *guaranteed*?

For each SLO target eps (max P(LC capacity < demand) per region-quarter), failure
correlation (Poisson / negative-binomial k) and reused-part failure multiplier, we compute
chance-constrained plans (solve -> Monte Carlo -> raise reserve -> re-solve) for
  * no reuse (E0-style), and
  * reuse allowed, reliability-aware (E2b-style),
then report
  Y_eps        = 1 - C(reuse, eps) / C(no reuse, eps)      reuse saving under a guaranteed SLO
  slo_price_%  = C(plan, eps) / C(plan, expected-value) - 1  carbon price of the SLO itself
  validated max P(violation) from an independent replay (fresh seed, 20k samples).

Usage: python experiments/p4_chance.py [--quick]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from carbonsched.grid import load_regions  # noqa: E402
from carbonsched.hardware import load_catalog  # noqa: E402
from carbonsched.milp import LifecycleModel, Scenario  # noqa: E402
from carbonsched.montecarlo import chance_constrained, replay_lc, summarize  # noqa: E402

CORR = {"independent": None, "correlated (k=2)": 2.0, "strongly correlated (k=0.5)": 0.5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--samples", type=int, default=30000)
    a = ap.parse_args()
    base = yaml.safe_load((ROOT / "configs/scenarios/fleet.yaml").read_text())
    base.update(lc_failure_buffer=1.0, allow_cascade=False, allow_geo_cascade=False)
    eps_list = [0.01] if a.quick else [0.01, 0.001]
    mults = [1.0] if a.quick else [1.0, 3.0]
    corr = {"independent": None} if a.quick else CORR
    rows = []
    for mult in mults:
        cat = load_catalog(ROOT / "configs", {"components.reused_part_afr_multiplier": mult})
        regions = load_regions(ROOT / "configs")
        no_reuse = [s for s in base["buyable"] if not cat.skus[s].is_reuse]
        plans = {"no_reuse": dict(buyable=no_reuse), "reuse": dict()}
        ev = {}
        for pname, kw in plans.items():  # expected-value plans (no reserve) for the SLO price
            b = copy.deepcopy(base)
            b.update(kw)
            ev[pname] = LifecycleModel(cat, regions, Scenario(**b)).solve().carbon_kg
        for cname, disp in corr.items():
            for eps in eps_list:
                res = {}
                for pname, kw in plans.items():
                    def factory(reserve, kw=kw):
                        b = copy.deepcopy(base)
                        b.update(kw)
                        b["lc_reserve"] = reserve
                        return LifecycleModel(cat, regions, Scenario(**b))
                    sol, model, reserve, hist = chance_constrained(factory, eps=eps, dispersion=disp,
                                                                   n_samples=a.samples, tol=0.0005)
                    check = summarize(replay_lc(model, sol, 50000, disp, seed=12345))
                    f = sol.fleet
                    res[pname] = dict(carbon=sol.carbon_kg, iterations=len(hist),
                                      reserve=sum(sum(v) for v in (reserve or {}).values()),
                                      validated_max_p=check["max_p_violation"],
                                      green_share=float(f[f.sku == "green_gpu"].n.sum() / f.n.sum()))
                row = dict(reused_multiplier=mult, correlation=cname, eps=eps,
                           Y_eps=100 * (1 - res["reuse"]["carbon"] / res["no_reuse"]["carbon"]),
                           Y_expected_value=100 * (1 - ev["reuse"] / ev["no_reuse"]),
                           slo_price_no_reuse_pct=100 * (res["no_reuse"]["carbon"] / ev["no_reuse"] - 1),
                           slo_price_reuse_pct=100 * (res["reuse"]["carbon"] / ev["reuse"] - 1),
                           reuse_green_share=res["reuse"]["green_share"],
                           reserve_no_reuse=res["no_reuse"]["reserve"], reserve_reuse=res["reuse"]["reserve"],
                           validated_max_p_no_reuse=res["no_reuse"]["validated_max_p"],
                           validated_max_p_reuse=res["reuse"]["validated_max_p"],
                           iterations=max(res["reuse"]["iterations"], res["no_reuse"]["iterations"]))
                rows.append(row)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "p4_chance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    colors = {1.0: "#4C78A8", 3.0: "#E45756"}
    marks = {0.01: "o", 0.001: "s"}
    for ax, col, title in ((axes[0], "Y_eps", "Reuse saving under a guaranteed LC SLO"),
                           (axes[1], "slo_price_reuse_pct", "Carbon price of the SLO (reuse plan)")):
        for (mult, eps), g in df.groupby(["reused_multiplier", "eps"]):
            ax.plot(range(len(g)), g[col], marker=marks[eps], color=colors.get(mult, "k"),
                    ls="-" if eps == 0.01 else "--",
                    label=f"reused parts {mult:g}x, P(short) <= {eps:g}")
        ax.set_xticks(range(len(g)), [c.replace(" (", "\n(") for c in g.correlation], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("Y_eps: saving vs no reuse (%)")
    axes[1].set_ylabel("vs expected-value plan (%)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Parameters provisional (GPU measurements pending)", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "figures" / "p4_chance.png", dpi=150)


if __name__ == "__main__":
    main()
