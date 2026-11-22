"""Plotting agent: render fig_lifecycle_model_overview (block diagram, 16:9, 300 dpi) from idea.md."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = sys.argv[1]
fig, ax = plt.subplots(figsize=(11, 6.19))
ax.set_xlim(0, 110); ax.set_ylim(0, 62); ax.axis("off")
C = dict(inp="#E8F0FA", inp_e="#4C78A8", core="#FFF1E0", core_e="#F58518", out="#E9F5E7", out_e="#54A24B",
         mc="#FBEAEA", mc_e="#E45756", txt="#222222")


def box(x, y, w, h, title, lines, fc, ec, tsize=10.5, lsize=8.4, step=2.25, top=5.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top", fontsize=tsize, weight="bold", color=C["txt"])
    for i, l in enumerate(lines):
        ax.text(x + 1.4, y + h - top - i * step, l, ha="left", va="top", fontsize=lsize, color=C["txt"])


def arrow(x0, y0, x1, y1, color="#555555", style="-|>", rad=0.0, lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


# Inputs
inputs = [
    ("Server SKUs", ["new / reuse-based (aged DRAM+SSD,", "CXL) / cascaded previous-gen", "power, throughput per tier"]),
    ("Component reliability", ["Weibull hazards, AFR calibration", "delayed renewal -> expected", "failures, drains, spares per age"]),
    ("Grids and demand", ["regional carbon intensity,", "declining per quarter; LC demand", "per region, BATCH global"]),
    ("Reused-part pools", ["harvested DIMMs / SSDs:", "external supply + screened", "local retirements"]),
]
for i, (t, ls) in enumerate(inputs):
    y = 47 - i * 14.5
    box(1, y, 27, 12.5, t, ls, C["inp"], C["inp_e"])
    arrow(28.6, y + 6.2, 36.5, 31 + (1.5 - i) * 3.2)

# Core LP
box(37, 14, 37, 34, "Multi-period lifecycle LP", [
    "cohorts n[s, tier, region, age, quarter]",
    "flow conservation, forced retirement",
    "decisions each quarter:",
    "   buy  |  keep  |  retire",
    "   cascade LC -> BATCH (in place / to",
    "   a clean-grid region)  |  lease",
    "capacity derated by expected drains",
    "objective: embodied (amortized)",
    "   + operational (power x PUE x CI)",
    "   + spares/logistics + lease + ship",
], C["core"], C["core_e"], tsize=11.5, lsize=8.9, step=2.75, top=6.0)

# MC loop
box(37, 0.5, 37, 10.5, "Monte Carlo chance constraint", [
    "M/G/inf drained nodes (Poisson / neg. binomial)",
    "reserve += demand - eps-quantile(capacity)",
], C["mc"], C["mc_e"], tsize=10, lsize=8.4, step=2.4, top=5.0)
arrow(48, 13.6, 48, 11.4, color=C["mc_e"])
arrow(63, 11.4, 63, 13.6, color=C["mc_e"])
ax.text(64, 12.5, "re-solve until P(short) <= eps", fontsize=7.8, color=C["mc_e"], va="center")

# Outputs
outs = [
    ("Ladder E0 ... E3g", ["X (failures ignored), Y_naive,", "Y (reliability-aware), Z / Zg", "(cascading, geographic)"]),
    ("Break-even analyses", ["reused-part failure multiplier,", "old/new batch work per joule,", "grid, accounting, fuzzy cuts"]),
    ("Guaranteed SLO", ["per region-quarter reserve,", "carbon price of the SLO,", "reuse saving under the SLO"]),
]
for i, (t, ls) in enumerate(outs):
    y = 44 - i * 15
    box(81, y, 28, 12.5, t, ls, C["out"], C["out_e"])
    arrow(74.6, 31 + (1 - i) * 4, 80.4, y + 6.2)

fig.tight_layout(pad=0.2)
fig.savefig(out, dpi=300)
plt.close(fig)
