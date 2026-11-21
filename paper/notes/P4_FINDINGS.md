# Phase 4 findings: reuse under a guaranteed SLO

Reproduce with:

```bash
python experiments/p4_chance.py
```

Results are in `results/p4_chance.csv`, and the figure is `paper/figures/p4_chance.png`. Parameters are provisional: GPU throughput, power and the CXL slowdown are placeholders until the GPU measurements replace them.

## Method: chance-constrained headroom with Monte Carlo in the loop
Phase 3 (F10) showed that the LP's linear headroom rule (LC capacity derated by a multiple of the *expected* repair drain) leaves some region-quarters with a 15% chance of running short. Phase 4 replaces that rule with a guarantee:

1. Solve the lifecycle LP, using expected-value derating only (buffer = 1).
2. For every LC region-quarter, sample concurrent repair drains under the planned fleet (M/G/∞ occupancy: Poisson, or negative binomial with dispersion *k* for correlated failures). Measure the gap between demand and the ε-quantile of capacity.
3. Add that gap, plus 5%, to a per-region-quarter **LC reserve**. The reserve is a constraint on *installed* capacity: reserve nodes sit idle, paying idle power and embodied carbon.
4. Re-solve. Repeat until every gap is ≤ 0.05% of demand.

Reserves only grow, so the loop terminates. It converged in 4–15 iterations, about 1–3 minutes per plan.

**Independent validation.** Each final plan is replayed with a fresh seed and 50k samples. The worst region-quarter came out at **1.00–1.07% against the 1% target** and **0.10–0.13% against the 0.1% target**, which is within Monte Carlo noise.

*Methodological note:* an earlier version of the replay rounded each age cohort to whole nodes. That dropped capacity the LP had planned and overstated violation probabilities by up to 5×. The replay now uses the planned (fractional) fleet. A real deployment rounds each cohort *up*, which only adds safety.

## F14. Reuse survives a guaranteed SLO, and its saving grows slightly

| Reused-part failure × | Correlation | ε | **Y under guaranteed SLO** | Y, expected-value plan | Carbon price of SLO (reuse / no reuse) |
|---|---|---|---|---|---|
| 1× | independent | 1% | 0.786% | 0.775% | 0.22% / 0.23% |
| 1× | independent | 0.1% | 0.788% | 0.775% | 0.30% / 0.31% |
| 1× | k = 2 | 0.1% | 0.792% | 0.775% | 0.43% / 0.44% |
| 1× | **k = 0.5 (strong)** | **0.1%** | **0.812%** | 0.775% | **0.79% / 0.82%** |
| 3× | independent | 0.1% | 0.484% | 0.480% | 0.31% / 0.31% |
| 3× | k = 0.5 (strong) | 0.1% | 0.499% | 0.480% | 0.80% / 0.82% |

- **Guaranteeing the SLO costs 0.2–0.8% extra carbon.** The cost rises with strictness and, more steeply, with failure correlation: strongly correlated failures need about 2.6× the reserve of independent ones.
- **The reuse plan pays a slightly *lower* SLO price than the no-reuse plan**, so the reuse saving *grows* (0.775% → 0.812%). Reserve capacity is idle by design, and on idle capacity embodied carbon dominates. That is where reused, low-embodied hardware is cheapest. This confirms finding F7 at fleet scale with explicit standby capacity, where Phase 3's derating-based headroom had only weakly supported it (F13).
- The effect holds when reused parts fail 3× more than their age predicts (0.480% → 0.499%).

## What this means for the brief's question
"Does reuse survive reliability?" now has a three-part answer:
1. **Expected value:** yes, if the plan accounts for reliability. It keeps about 90% of the deterministic saving, never goes negative, and stops reusing at about 8× worse parts (F8). A failure-blind plan turns negative at about 2×.
2. **Guaranteed SLO:** yes. Reuse remains a net saving under a 99.9% per-quarter capacity guarantee with strongly correlated failures. Its saving even grows a little, because reused hardware is the cheapest way to hold idle reserve.
3. **Magnitude:** set by the grid and by GPU efficiency, not by reliability. It is about 5% on clean grids and under 1% on mixed and dirty grids (F1, F2).

**Design implication for a "reliability-aware GreenSKU for AI":** build *reserve / standby* capacity from reused-component hosts, and build *steadily loaded* capacity from the most energy-efficient hosts.

## Remaining work
- GPU measurements (C2/C3) to replace placeholder throughput, power and CXL slowdown.
- Verification of the literature values (embodied carbon dominates the fuzzy upper bound).
- Optional: put the reserve directly into the LP as a sample-average approximation, avoiding the outer loop. The loop is already fast enough at this scale.
