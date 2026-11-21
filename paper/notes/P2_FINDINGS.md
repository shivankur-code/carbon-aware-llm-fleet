# Phase 2 findings: lifecycle optimizer and first X / Y / Z

Reproduce with:

```bash
python experiments/p2_pipeline.py --region {clean,mixed,dirty} --lp [--scale 50]
```

The `--lp` flag uses the LP relaxation. **Parameters are still provisional** (see `paper/parameters.md`): GPU throughput, power and the CXL slowdown are placeholders until the GPU measurements (C2/C3) replace them.

## What the model is
A quarterly, multi-period lifecycle model (`src/carbonsched/milp.py`), solved as a MILP or as its LP relaxation:
- the fleet is tracked as cohorts by server type × tier × region × node age;
- decisions each quarter: buy, retire, cascade from the latency-critical (LC) tier to the batch tier (optionally to another region), lease, and accept shortfall;
- reused-part harvest pools: GreenSKU-style builds consume harvested DIMMs and SSDs;
- failure-aware capacity, with N+k-style headroom for the LC tier;
- embodied carbon amortized over the design life, with the unamortized remainder charged on early retirement;
- ε-constraint Pareto fronts across objectives.

**Validation.** 11 optimizer tests, including:
- an exact match to a hand calculation;
- demand growth triggering later purchases;
- reuse pools limiting GreenSKU builds;
- cascading conserving node counts;
- replaying a fixed plan under failures;
- a monotone Pareto front;
- LP scale-invariance.

## Formulation lesson: use the LP relaxation at fleet scale
On the pilot (about 36 nodes), the integer solves **hit the 300 s time limit without proving optimality**, and PuLP still reports "Optimal". Rounding to whole nodes moves carbon by about 0.4%, the same order as the effects being measured. The LP relaxation solves in under 1 s and matches the MILP's X and Y within about 0.05 points. It is also exactly scale-invariant, giving identical X/Y/Z at 1× and 50× fleet size. All headline numbers from here on use the **LP at fleet scale**. Runs now record `time_limited` so a truncated MILP can't pass as optimal again.

## Pilot results (3-year horizon, an aging A100/H100 fleet, growing LC and batch demand; LP)

| Grid | X: reuse, failures ignored | Y naive: that plan with failures replayed | Y: reliability-aware plan | Z: cascading |
|---|---|---|---|---|
| clean | **8.65%** | 7.07% | **7.49%** | +0.00 pp |
| mixed | **0.97%** | 0.33% | **0.85%** | +0.00 pp |
| dirty | **0.18%** | **−0.33%** | **0.15%** | +0.00 pp |

### F5. Reuse survives reliability *if the plan accounts for it*
- A reliability-aware plan keeps **about 87% of the deterministic saving on every grid** (clean 7.49/8.65; mixed 0.85/0.97; dirty 0.15/0.18).
- A plan that ignores failures, run in a world with failures, keeps **82% on the clean grid but only 34% on the mixed grid, and on the dirty grid it does worse than not reusing at all**. The losses come from extra leasing to cover drained capacity, plus extra spare parts.
- So the answer to the brief's question has two parts. Reuse is robust to age-dependent failures, but only when procurement accounts for them. Failure-blind reuse can lose most of its benefit or reverse it. This is a sharper, more defensible headline than a single "X → Y".

### F6. Cascading adds nothing in the single-region pilot
In-place cascading has zero value here. The cascade candidates are A100-class nodes, and with the placeholder performance-per-watt they are worse than buying GreenSKU capacity for the batch tier (consistent with P1 finding F4). Phase 3 tests whether **geographic cascading** (multi-region, with shipping carbon) and **measured** A100 batch efficiency (C2) change this. If Z stays about 0 even then, that is itself a finding: **cascading is a cost lever, not a carbon lever, for GPU inference**.

### F7. Light-load capacity should be built from reused hardware
Even on the dirty grid, where GreenSKU loses per unit of work, the optimizer fills the last, lightly used node with a reused-part SKU. At low utilization, embodied carbon dominates. This suggests a design rule: **use reused-component servers for failure headroom, buffer and peak capacity, and new efficient servers for the steadily busy base load.** Phase 3 should test this directly by varying utilization and headroom.

## Prototype Figure 1
`paper/figures/fig1_prototype_pilot_{clean,mixed,dirty}.png`.
- Left panel: stacked lifecycle carbon (embodied, operational, spares and logistics, lease, shipping) for E0_det, E1, E0, E2a, E2b and E3, with the axis starting at 0.
- Right panel: X, Y-naive, Y and Y+Z as savings against the matching baseline.

The MILP-solved versions of these figures are kept for comparison; the `_lp` versions are the reference.

## What Phase 3 needs
1. A realistic scenario: 6-year horizon, 3 regions, demand growth, LP at fleet scale.
2. Geographic cascading enabled; a sweep of utilization and LC headroom (F7).
3. Replace the placeholders with GPU measurements (C2: A100 vs H100, LC vs batch; C3: CXL / KV-offload slowdown). The verdict hinges on them (P1 finding F2).
4. Monte Carlo replay of plans for SLO-violation distributions, plus fuzzy α-cuts.
