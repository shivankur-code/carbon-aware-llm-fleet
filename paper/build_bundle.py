"""Build the paper bundle for paper-orchestra from the repo's results.

Reads results/*.csv|json (and configs/measured.yaml, gpu/results/** if present) and writes:
  paper/numbers.json        every number quoted in the paper, with its source file
  paper/claims.md           claim -> evidence (file, figure) -> status (provisional / measured / to-verify)
  paper/outline.md          title, abstract (filled with numbers), section plan, figure list + captions
  paper/brief_headline.md   the X / Y / Z sentence and Figure 1 for the research brief

Re-run after new results (e.g. GPU measurements):  python paper/build_bundle.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"


def opt_csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def crossing(x, y, level=0.0, log=False):
    """First x where y crosses ``level`` going down (linear interpolation; log-x if requested)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(1, len(x)):
        if (y[i - 1] - level) > 0 >= (y[i] - level):
            xa, xb = (np.log(x[i - 1]), np.log(x[i])) if log else (x[i - 1], x[i])
            f = (y[i - 1] - level) / (y[i - 1] - y[i])
            v = xa + f * (xb - xa)
            return float(np.exp(v) if log else v)
    return None


def crossing_up(x, y, level):
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(1, len(x)):
        if y[i - 1] < level <= y[i]:
            return float(x[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1]))
    return None


def main() -> None:
    N: dict = {}
    src: dict = {}

    def put(key, val, source):
        N[key] = None if val is None or (isinstance(val, float) and math.isnan(val)) else val
        src[key] = source

    measured = {}
    if (ROOT / "configs" / "measured.yaml").exists():
        measured = (yaml.safe_load((ROOT / "configs" / "measured.yaml").read_text()) or {}).get("measured", {})
    gpu_measured = any("old_gpu.throughput" in k for k in measured)
    cxl_measured = any("cxl_slowdown" in k for k in measured)
    status_gpu = "MEASURED" if gpu_measured else "PROVISIONAL (C2 pending)"
    status_cxl = "MEASURED" if cxl_measured else "PROVISIONAL (C3a pending)"

    # --- P1 node level
    t = opt_csv("p1_tornado_mixed_LC_cutoff.csv")
    if t is not None:
        top = t.iloc[0]
        put("p1_top_sensitivity_param", top.param, "results/p1_tornado_mixed_LC_cutoff.csv")
        put("p1_top_sensitivity_swing_pp", float(top.swing), "results/p1_tornado_mixed_LC_cutoff.csv")
    n = opt_csv("p1_node_lifecycle.csv")
    if n is not None:
        r = n[(n.sku == "new_gpu") & (n.region == "mixed") & (n.tier == "LC") & (n.policy == "cutoff") & (n.failures)]
        if len(r):
            r = r.iloc[0]
            put("p1_operational_share_mixed", float(r.operational_kg / r.total_kg), "results/p1_node_lifecycle.csv")

    # --- P3 ladder
    hp = RES / "p3_ladder_headline.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        for k in ("X", "Y_naive", "Y", "Z_pp", "Zg_pp"):
            put(f"fleet_{k}", h.get(k), "results/p3_ladder_headline.json")
        if h.get("X"):
            put("fleet_retained_frac_aware", h["Y"] / h["X"], "derived: Y/X")
            put("fleet_retained_frac_naive", h["Y_naive"] / h["X"], "derived: Y_naive/X")
    br = opt_csv("p3_ladder_by_region.csv")
    if br is not None:
        for _, r in br.iterrows():
            if r.region in ("clean", "mixed", "dirty"):
                for k in ("X", "Y_naive", "Y"):
                    put(f"{r.region}_{k}", float(r[k]), "results/p3_ladder_by_region.csv")

    # --- reliability break-even
    rl = opt_csv("p3_reliab.csv")
    if rl is not None:
        put("reliab_breakeven_naive_mult", crossing(rl.multiplier, rl.Y_naive, 0.0, log=True), "results/p3_reliab.csv")
        put("reliab_breakeven_aware_mult", crossing(rl.multiplier, rl.Y, 0.01, log=True), "results/p3_reliab.csv (Y < 0.01%)")
        put("reliab_naive_worst", float(rl.Y_naive.min()), "results/p3_reliab.csv")

    # --- cascade break-even
    cs = opt_csv("p3_cascade.csv")
    if cs is not None:
        put("cascade_breakeven_eff", crossing_up(cs.old_vs_new_work_per_joule, cs.Z_pp, 0.05), "results/p3_cascade.csv (Z >= 0.05 pp)")
        put("cascade_geo_breakeven_eff", crossing_up(cs.old_vs_new_work_per_joule, cs.Zg_pp, 0.05), "results/p3_cascade.csv")
        hi = cs.iloc[-1]
        put("cascade_geo_uplift_frac", float(hi.Zg_pp / hi.Z_pp - 1) if hi.Z_pp > 0 else None, "results/p3_cascade.csv")

    # --- MC tails
    mc = opt_csv("p3_mc.csv")
    if mc is not None:
        for _, r in mc.iterrows():
            tag = "indep" if "independent" in r.correlation else r.correlation.replace("NB k=", "k")
            put(f"mc_{r.run}_{tag}_maxp", 100 * float(r.max_p_violation), "results/p3_mc.csv (percent)")
            put(f"mc_{r.run}_{tag}_q_over_1pct", int(r.quarters_over_1pct), "results/p3_mc.csv")

    # --- accounting policy
    pol = opt_csv("p3_policy.csv")
    if pol is not None:
        for _, r in pol.iterrows():
            put(f"policy_{r.policy}_{r.accounting}_Y", float(r.Y), "results/p3_policy.csv")

    # --- fuzzy
    fz = opt_csv("p3_fuzzy.csv")
    if fz is not None:
        for _, r in fz.iterrows():
            put(f"fuzzy_a{r.alpha:g}_low", float(r.low), "results/p3_fuzzy.csv")
            put(f"fuzzy_a{r.alpha:g}_high", float(r.high), "results/p3_fuzzy.csv")

    # --- pareto
    pf = opt_csv("p3_pareto.csv")
    if pf is not None and len(pf) > 1:
        cmin, cmax = pf.loc[pf.cost_usd.idxmin()], pf.loc[pf.carbon_kg.idxmin()]
        put("pareto_carbon_penalty_of_costmin_pct", float(100 * (cmin.carbon_kg / cmax.carbon_kg - 1)), "results/p3_pareto.csv")
        put("pareto_cost_saving_of_costmin_pct", float(100 * (1 - cmin.cost_usd / cmax.cost_usd)), "results/p3_pareto.csv")

    # --- P4 chance constrained
    p4 = opt_csv("p4_chance.csv")
    if p4 is not None:
        for _, r in p4.iterrows():
            tag = ("indep" if r.correlation.startswith("independent") else
                   "k2" if "k=2" in r.correlation else "k0.5")
            key = f"slo_m{r.reused_multiplier:g}_{tag}_eps{r.eps:g}"
            put(f"{key}_Y", float(r.Y_eps), "results/p4_chance.csv")
            put(f"{key}_price_reuse_pct", float(r.slo_price_reuse_pct), "results/p4_chance.csv")
            put(f"{key}_price_noreuse_pct", float(r.slo_price_no_reuse_pct), "results/p4_chance.csv")
            put(f"{key}_validated_maxp", 100 * float(max(r.validated_max_p_reuse, r.validated_max_p_no_reuse)), "results/p4_chance.csv (percent)")
        strict = p4[(p4.eps == p4.eps.min())]
        put("slo_strict_Y_min", float(strict.Y_eps.min()), "results/p4_chance.csv")
        put("slo_strict_Y_max", float(strict.Y_eps.max()), "results/p4_chance.csv")
        put("slo_price_max_pct", float(max(p4.slo_price_reuse_pct.max(), p4.slo_price_no_reuse_pct.max())), "results/p4_chance.csv")

    # --- C3b fault injection (GPU)
    c3 = sorted(glob.glob(str(ROOT / "gpu" / "results" / "**" / "c3faults_*_summary.csv"), recursive=True))
    c3 = [f for f in c3 if "tiny-random" not in f]
    put("c3b_available", bool(c3), "gpu/results/**/c3faults_*_summary.csv")

    N["_status"] = dict(gpu_params=status_gpu, cxl=status_cxl, measured_params=sorted(measured))
    (PAPER / "numbers.json").write_text(json.dumps(dict(numbers=N, sources=src), indent=2, default=float))

    f = lambda k, d=2, suf="": ("n/a" if N.get(k) is None else f"{N[k]:.{d}f}{suf}")  # noqa: E731
    pct = lambda k, d=2: f(k, d, "%")  # noqa: E731

    # ---------------- claims.md
    claims = f"""# Claims → evidence

Generated by `paper/build_bundle.py`. Do not edit by hand; edit the generator or rerun the experiments.

Status keys: **PROVISIONAL** = depends on placeholder GPU/CXL parameters; **MEASURED** = backed by `gpu/` results;
**TO-VERIFY** = depends on literature values not yet checked against the source PDF.
Current status: GPU parameters **{status_gpu}**; CXL slowdown **{status_cxl}**.

| # | Claim | Evidence | Figure | Status |
|---|---|---|---|---|
| C1 | For 8-GPU inference hosts, operational carbon is ~{f('p1_operational_share_mixed', 0) if N.get('p1_operational_share_mixed') is None else round(100 * N['p1_operational_share_mixed'])}% of 6-yr lifecycle carbon on a mixed grid, so reuse levers that cut embodied carbon have limited reach. | results/p1_node_lifecycle.csv | – | {status_gpu}; TO-VERIFY (embodied values) |
| C2 | The reuse saving is strongly grid-dependent: X = {pct('clean_X')} (clean), {pct('mixed_X')} (mixed), {pct('dirty_X')} (dirty). | results/p3_ladder_by_region.csv | fig1_fleet.png (right) | {status_gpu} |
| C3 | The sign of the node-level reuse saving on mixed grids is decided by the CXL-memory slowdown (largest tornado swing, {f('p1_top_sensitivity_swing_pp', 1)} pp). | results/p1_tornado_mixed_LC_cutoff.csv | p1_tornado_mixed_LC_cutoff.png | {status_cxl} |
| C4 | Reuse survives age-dependent failures **if planned for**: a reliability-aware plan keeps {f('fleet_retained_frac_aware', 2)}x of the deterministic saving (Y = {pct('fleet_Y')} vs X = {pct('fleet_X')}); a failure-blind plan keeps {f('fleet_retained_frac_naive', 2)}x ({pct('fleet_Y_naive')}). | results/p3_ladder_headline.json | fig1_fleet.png (left) | {status_gpu} |
| C5 | Break-even reliability: a failure-blind plan becomes carbon-negative once reused parts fail ~{f('reliab_breakeven_naive_mult', 1)}x more than age predicts; a reliability-aware plan degrades gracefully and stops reusing at ~{f('reliab_breakeven_aware_mult', 1)}x, never doing worse than no reuse. | results/p3_reliab.csv | p3_reliab.png | {status_gpu} |
| C6 | Cascading old GPUs to batch inference saves carbon only if their batch work-per-joule reaches ~{f('cascade_breakeven_eff', 2)}x the new GPU's; geographic cascading to clean grids adds ~{'n/a' if N.get('cascade_geo_uplift_frac') is None else round(100 * N['cascade_geo_uplift_frac'])}% once it pays. | results/p3_cascade.csv | p3_cascade.png | {status_gpu} — C2 decides |
| C7 | Linear (expected-value) failure headroom leaves SLO tail risk: worst region-quarter P(LC short) = {pct('mc_E2b_indep_maxp', 1)} for reuse ({pct('mc_E0_indep_maxp', 1)} without), independent failures; {pct('mc_E2b_k0.5_maxp', 1)} under strong correlation. | results/p3_mc.csv | – | {status_gpu} |
| C8 | Under a guaranteed SLO (chance-constrained headroom, P(short) ≤ 0.1% per region-quarter), reuse still saves {pct('slo_strict_Y_min')} – {pct('slo_strict_Y_max')} across failure correlation and reused-part quality; the SLO itself costs up to {pct('slo_price_max_pct')} extra carbon. | results/p4_chance.csv | p4_chance.png | {status_gpu} |
| C9 | Accounting convention changes Y by ~2x: cutoff/amortized {pct('policy_cutoff_amortized_Y')}, residual/amortized {pct('policy_residual_amortized_Y')}, cutoff/upfront {pct('policy_cutoff_upfront_Y')}; 0 if reused parts are charged as new. | results/p3_policy.csv | – | methodological |
| C10 | Under triangular fuzzy uncertainty (failure ×[0.5,1,3], embodied ×[0.5,1,2], CXL [0.85,0.95,1]), Y ∈ [{pct('fuzzy_a0_low')}, {pct('fuzzy_a0_high')}] at α=0 and [{pct('fuzzy_a0.5_low')}, {pct('fuzzy_a0.5_high')}] at α=0.5 — never negative. | results/p3_fuzzy.csv | – | {status_gpu}; TO-VERIFY |
| C11 | Minimizing cost instead of carbon costs +{pct('pareto_carbon_penalty_of_costmin_pct', 1)} carbon for −{pct('pareto_cost_saving_of_costmin_pct', 1)} cost. | results/p3_pareto.csv | p3_pareto.png | {status_gpu} |
| C12 | KV-cache tolerance to DRAM bit errors (with none / SECDED / checksum) | gpu/results/**/c3faults_*_summary.csv | (to add) | {'MEASURED' if N.get('c3b_available') else 'PENDING (run gpu/c3_kv_faults.py)'} |
"""
    (PAPER / "claims.md").write_text(claims)

    # ---------------- brief_headline.md
    brief = f"""# Headline for the research brief

Generated by `paper/build_bundle.py` (status: GPU parameters {status_gpu}).

**Template sentence (as in the brief):**
> Reuse cut lifecycle carbon by **{pct('fleet_X')}** in a deterministic plan, but age-related failures and drain
> overhead eroded this to **{pct('fleet_Y')}**; cascading older servers to batch inference restored **{f('fleet_Z_pp', 2)} pp**.

**Recommended replacement (sharper and more defensible):**
> GreenSKU-style reuse cut the lifecycle carbon of an LLM-inference fleet by **{pct('clean_X', 1)}** on a clean grid but
> under **{pct('mixed_X', 1)}** on a mixed one, because operational carbon dominates GPU hosts. Reuse **survives
> reliability only if planned for**: a reliability-aware plan kept **{'n/a' if N.get('fleet_retained_frac_aware') is None else round(100 * N['fleet_retained_frac_aware'])}%** of the saving and stayed
> non-negative even with reused parts failing ~{f('reliab_breakeven_aware_mult', 0)}x more than their age predicts, while a
> failure-blind plan turned carbon-negative at ~{f('reliab_breakeven_naive_mult', 1)}x. Cascading old GPUs to batch inference
> pays only if their batch tokens-per-joule reach ~{f('cascade_breakeven_eff', 1)}x the new GPU's — which our GPU
> measurements {'confirm/refute (see C6)' if gpu_measured else 'will test'}.

**Figure 1:** `paper/figures/fig1_fleet.png` (left: change vs no-reuse baseline by carbon component;
right: X / Y / Y+Z by grid). Alternative single-panel: `paper/figures/p3_reliab.png`.

*Caveat to keep in the brief:* modelling study with public parameters only; no proprietary fleet data.
"""
    (PAPER / "brief_headline.md").write_text(brief)

    # ---------------- outline.md
    outline = f"""# Paper outline (input for paper-orchestra)

**Title:** Does Reuse Survive Reliability? Lifecycle Optimization for Low-Carbon AI Inference Fleets

**Status:** GPU parameters {status_gpu}; CXL slowdown {status_cxl}. Numbers below are filled from `numbers.json`.

## Abstract (draft)
Reusing DRAM and SSDs from retired servers is one of the few sustainability levers shown to cut emissions at
cloud scale. But reuse studies evaluate a server design at a single point in time, while operators make
sequential buy, reuse, cascade and retire decisions over years — and reused parts are old, so they fail more.
We build a reliability-aware, multi-period lifecycle optimization for LLM-inference fleets that couples
age-dependent (Weibull) component failure models, repair drains and spare parts, reused-part harvest pools,
regional grid-carbon trajectories, cascading of old GPUs to batch tiers (including across regions), and
chance-constrained SLO headroom validated by Monte Carlo. On a 6-year, three-grid fleet we find that
(i) reuse savings for GPU inference hosts are strongly grid-dependent ({pct('clean_X', 1)} on a clean grid vs
{pct('mixed_X', 1)} on a mixed grid) because operational carbon dominates; (ii) reuse survives reliability only when
planned for — a reliability-aware plan retains {'n/a' if N.get('fleet_retained_frac_aware') is None else round(100 * N['fleet_retained_frac_aware'])}% of the saving and never does worse than no reuse,
whereas a failure-blind plan turns carbon-negative once reused parts fail ~{f('reliab_breakeven_naive_mult', 1)}x more than their age
predicts; (iii) cascading old GPUs pays only above ~{f('cascade_breakeven_eff', 1)}x relative batch energy efficiency;
and (iv) guaranteeing a 99.9% SLO under correlated failures costs up to {pct('slo_price_max_pct', 1)} extra carbon while preserving
{pct('slo_strict_Y_min')}–{pct('slo_strict_Y_max')} reuse savings. GPU measurements of serving efficiency per tier, KV-cache
offload to CXL-like memory, and KV-cache tolerance to DRAM bit errors ground the key parameters.

## Sections
1. **Introduction** — GreenSKU's result; the operator's sequential view; the question; contributions C1–C3 (PLAN.md §1b).
2. **Background** — embodied vs operational carbon of GPU hosts; field reliability of DRAM/SSD/GPU; CXL memory; LLM serving tiers.
3. **Model** — cohorts by age; decisions; reliability as precomputed expected failures (keeps it linear); headroom;
   harvest pools; embodied accounting (amortized + stranded); objectives and ε-constraint; LP-at-scale justification
   (P2 finding: truncated MILPs, LP within 0.05 pp, scale-invariant).
4. **Measurements (C2, C3)** — serving efficiency per GPU generation × tier; KV offload local vs remote NUMA; KV fault injection.
5. **Results** — Figure 1 ladder; grid dependence; reliability break-even; cascade break-even; SLO tails and the
   chance-constrained fix; accounting sensitivity; fuzzy robustness; Pareto.
6. **Implications for low-carbon AI server design** — reliability-aware GreenSKU for GPU hosts; reused memory as the
   KV-cache tier with ECC/checksums; cascading as a cost (not carbon) lever unless efficiency parity; geographic cascading.
7. **Limitations** — public parameters, provisional values listed in parameters.md; expected-value reliability inside the
   LP with Monte Carlo validation; no proprietary fleet data.
8. **Related work** — `related_work.md`.

## Figures
| Fig | File | Caption |
|---|---|---|
| 1 | figures/fig1_fleet.png | Lifecycle carbon change vs a no-reuse fleet for: a failure-blind reuse plan, a reliability-aware plan, and with in-place / geographic cascading (left); savings by grid (right). |
| 2 | figures/p3_reliab.png | Fleet reuse saving vs reused-part failure-rate multiplier: reliability-aware planning degrades gracefully; failure-blind planning turns negative. |
| 3 | figures/p3_cascade.png | Extra saving from cascading vs old/new GPU batch work-per-joule; the break-even sits near energy parity. |
| 4 | figures/p4_chance.png | Reuse saving and carbon price of a guaranteed LC SLO across failure correlation and reused-part quality. |
| 5 | figures/p1_tornado_mixed_LC_cutoff.png | Node-level sensitivity: the CXL slowdown decides the sign on mixed grids. |
| 6 | figures/p3_pareto.png | Carbon–cost Pareto front. |
| 7 | (C2) | Tokens/J per GPU generation × tier — from gpu/results. |
| 8 | (C3b) | Perplexity ratio vs DRAM bit-error rate for none / SECDED / checksum — from gpu/results. |

## Tables
- Parameters with status and source: `parameters.md`.
- Claims → evidence: `claims.md`.
"""
    (PAPER / "outline.md").write_text(outline)
    print(f"wrote numbers.json ({len(N)} entries), claims.md, outline.md, brief_headline.md  [GPU: {status_gpu}]")


if __name__ == "__main__":
    main()
