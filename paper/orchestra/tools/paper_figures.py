"""Plotting agent (refinement pass): re-render result figures for the paper from results/*.csv
with paper-level labels (no internal identifiers), 300 dpi."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); R = ROOT / "results"
C = dict(blue="#4C78A8", orange="#F58518", red="#E45756", teal="#72B7B2", green="#54A24B", purple="#B279A2", grey="#9D9D9D")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=300); plt.close(fig)


# Fleet ladder
runs = pd.read_csv(R / "p3_ladder_runs.csv").set_index("run")
br = pd.read_csv(R / "p3_ladder_by_region.csv")
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
order = [("E2a", "Failure-blind\nreuse plan"), ("E2b", "Reliability-aware\nreuse plan"), ("E3", "+ in-place\ncascading"), ("E3g", "+ geographic\ncascading")]
comps = [("embodied_kg", "Embodied", C["blue"]), ("operational_kg", "Operational", C["orange"]),
         ("repair_kg", "Spares + logistics", C["red"]), ("lease_kg", "Leased capacity", C["teal"])]
w = 0.18
for j, (col, name, c) in enumerate(comps):
    ax.bar(np.arange(len(order)) + (j - 1.5) * w, [(runs.loc[k, col] - runs.loc["E0", col]) / 1000 for k, _ in order], w, color=c, label=name)
ax.scatter(range(len(order)), [(runs.loc[k, "carbon_kg"] - runs.loc["E0", "carbon_kg"]) / 1000 for k, _ in order], color="k", zorder=3, s=20, label="Net change")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(order)), [o[1] for o in order], fontsize=8)
ax.set_ylabel("Change vs no-reuse fleet (tCO$_2$e, 6 years)")
ax.legend(fontsize=7, ncol=2, frameon=False)
b = br[br.region.isin(["clean", "mixed", "dirty"])]
x = np.arange(len(b))
for j, (col, name, c) in enumerate([("X", "X: failures ignored", C["blue"]), ("Y_naive", "Failure-blind plan", C["red"]), ("Y", "Y: reliability-aware", C["orange"])]):
    ax2.bar(x + (j - 1) * 0.26, b[col], 0.26, color=c, label=name)
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xticks(x, [f"{r} grid" for r in b.region])
ax2.set_ylabel("Lifecycle carbon saving vs no reuse (%)")
ax2.legend(fontsize=7, frameon=False)
save(fig, "fig_fleet_carbon_ladder_by_grid")

# Reliability break-even
d = pd.read_csv(R / "p3_reliab.csv")
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(d.multiplier, d.X, "o-", color=C["blue"], label="X (failures ignored)")
ax.plot(d.multiplier, d.Y, "o-", color=C["orange"], label="Y (reliability-aware plan)")
ax.plot(d.multiplier, d.Y_naive, "o-", color=C["red"], label="Failure-blind plan")
ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log")
ax.set_xlabel("Reused-part failure-rate multiplier $m$ (beyond age effect)")
ax.set_ylabel("Fleet carbon saving vs no reuse (%)")
ax.legend(fontsize=8, frameon=False)
save(fig, "fig_reliability_breakeven_reused_parts")

# Cascade break-even
d = pd.read_csv(R / "p3_cascade.csv")
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(d.old_vs_new_work_per_joule, d.Z_pp, "o-", color=C["green"], label="$Z$: in-place cascading")
ax.plot(d.old_vs_new_work_per_joule, d.Zg_pp, "s--", color=C["purple"], label="$Z_g$: geographic cascading")
ax.axhline(0, color="k", lw=0.8); ax.axvline(1.0, color=C["grey"], lw=0.8, ls=":")
ax.set_xlabel("Previous- vs current-generation BATCH work per joule")
ax.set_ylabel("Extra saving from cascading (pp)")
ax.legend(fontsize=8, frameon=False)
save(fig, "fig_cascade_breakeven_energy_efficiency")

# Chance-constrained SLO
d = pd.read_csv(R / "p4_chance.csv")
labels = ["independent", "correlated\n($k$=2)", "strongly corr.\n($k$=0.5)"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, col, yl in ((axes[0], "Y_eps", "Reuse saving under guaranteed SLO (%)"), (axes[1], "slo_price_reuse_pct", "Carbon price of the SLO, reuse plan (%)")):
    for (m, eps), g in d.groupby(["reused_multiplier", "eps"]):
        ax.plot(range(3), g[col], marker="s" if eps < 0.005 else "o", ls="--" if eps < 0.005 else "-",
                color=C["blue"] if m == 1 else C["red"], label=f"$m$={m:g}, P(short) ≤ {eps:g}")
    ax.set_xticks(range(3), labels, fontsize=8); ax.set_ylabel(yl)
axes[0].set_ylim(0, None); axes[1].set_ylim(0, None)
axes[0].legend(fontsize=7, frameon=False)
save(fig, "fig_chance_constrained_slo")

# Tornado with readable labels
t = pd.read_csv(R / "p1_tornado_mixed_LC_cutoff.csv").head(10).iloc[::-1]
nice = {"skus.skus.green_gpu.cxl_slowdown.LC": "CXL throughput multiplier (LC)",
        "skus.skus.green_gpu.power_peak_delta_w": "Reuse host peak-power delta",
        "components.components.ssd.embodied_kg": "SSD embodied carbon",
        "regions.regions.mixed.ci_base": "Grid carbon intensity",
        "skus.skus.green_gpu.power_idle_delta_w": "Reuse host idle-power delta",
        "skus.skus.green_gpu.platform_embodied_kg": "Reuse host platform embodied",
        "skus.skus.new_gpu.platform_embodied_kg": "New host platform embodied",
        "components.components.ssd.failure.shape": "SSD Weibull shape",
        "skus.skus.new_gpu.power_peak_w": "New host peak power",
        "regions.default_pue": "PUE"}
base = 0.93
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for i, r in enumerate(t.itertuples()):
    ax.barh(i, r.saving_at_low - base, left=base, color=C["blue"], label="parameter at low end" if i == 0 else None)
    ax.barh(i, r.saving_at_high - base, left=base, color=C["orange"], label="parameter at high end" if i == 0 else None)
ax.set_yticks(range(len(t)), [nice.get(p, p) for p in t.param], fontsize=8)
ax.axvline(base, color="k", lw=0.8); ax.axvline(0, color=C["grey"], lw=0.8, ls="--")
ax.set_xlabel("Node-level reuse saving, mixed grid, LC tier (%)")
ax.legend(fontsize=7, frameon=False, loc="lower left")
save(fig, "fig_node_sensitivity_tornado")

# Pareto
f = pd.read_csv(R / "p3_pareto.csv")
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.plot(f.cost_usd / 1e9, f.carbon_kg / 1e6, "o-", color=C["blue"])
ax.set_xlabel("Six-year cost (billion USD)"); ax.set_ylabel("Six-year lifecycle carbon (ktCO$_2$e)")
save(fig, "fig_carbon_cost_pareto")
print("rendered")
