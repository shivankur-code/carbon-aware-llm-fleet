# Phase 3 findings: reference fleet, sensitivity, uncertainty

Reproduce with:

```bash
python experiments/p3_study.py all
```

Results go to `results/p3_*.csv` and the figures to `paper/figures/fig1_fleet.png` and `p3_*.png`.

**Scenario.** `configs/scenarios/fleet.yaml`:
- 6 years in quarterly steps, three grids (clean / mixed / dirty);
- LC demand growing 25–30%/yr, batch demand growing 35%/yr;
- an existing fleet of about 1,750 A100- and H100-class 8-GPU hosts;
- reused DIMMs and SSDs supplied by a general-purpose fleet;
- solved as an LP at fleet scale (each full ladder takes about 4 s).

**Parameters are provisional.** GPU throughput, power and the CXL slowdown are placeholders until `gpu/` measurements replace them (see `gpu/README.md`). Treat these as *structural* findings whose magnitudes will move.

## Headline (Figure 1, `fig1_fleet.png`)

| | X: reuse, failures ignored | plan blind to failures | **Y: reliability-aware** | Z: + cascading | Zg: + geographic cascading |
|---|---|---|---|---|---|
| 3-grid fleet | 0.85% | 0.19% | **0.78%** | +0.005 pp | +0.005 pp |
| clean grid only | 5.77% | 4.58% | **5.03%** | +0.005 pp | – |
| mixed grid only | 0.68% | 0.04% | **0.57%** | 0 | – |
| dirty grid only | 0.10% | −0.44% | **0.09%** | 0 | – |

## F8. "Does reuse survive reliability?" Yes, up to a point, but only if the plan accounts for failures (`p3_reliab.png`)
We swept a multiplier on reused-part failure rates, on top of the age effect the Weibull models already capture:

| Reused-part failure multiplier | 0.5× | 1× | 2× | 3× | 5× | 8× and above |
|---|---|---|---|---|---|---|
| Y (reliability-aware) | 0.86% | 0.78% | 0.62% | 0.48% | 0.21% | **0%** |
| Plan blind to failures | 0.29% | 0.20% | **0.02%** | −0.15% | −0.47% | down to −2.25% |

- A **reliability-aware plan degrades gracefully**. It buys less reused hardware as parts get worse, and simply stops at about 8×. It is **never worse than not reusing**.
- A **failure-blind plan breaks even at about 2× and turns carbon-negative beyond**. The losses come from leasing to cover drained capacity, plus extra spare parts.
- For the brief, this gives a concrete reading of the question: **reuse is worth it while reused parts fail less than about 2× more than their age predicts, unless planning accounts for reliability, in which case the threshold is about 8×.**

## F9. Cascading old GPUs pays only if their batch efficiency reaches about 0.9–1.0× the new GPU's (`p3_cascade.png`)
We swept the A100-class node's BATCH throughput, and so its batch work per joule relative to an H100-class node:

| A100/H100 batch work per joule | 0.47 | 0.70 | 0.93 | 1.17 | 1.40 | 1.87 |
|---|---|---|---|---|---|---|
| Z: in-place cascading (pp) | 0 | 0.005 | 0.03 | 0.11 | 0.19 | 0.35 |
| Zg: + geographic cascading (pp) | 0 | 0.005 | 0.03 | 0.18 | 0.32 | 0.52 |

- Cascading avoids only the new server's embodied carbon (about 5% of a GPU host's lifecycle carbon), so it cannot make up for an operational-efficiency gap. **The break-even is close to energy parity.**
- **Geographic cascading** (shipping old nodes to the clean grid's batch tier) adds about 50–70% to Z once cascading pays at all.
- **C2 decides this.** Public spec sheets suggest H100s deliver well over 1.5× A100 inference throughput at about 1.4× the power. If C2 confirms that, **cascading is a cost lever, not a carbon lever, for GPU inference**. That is a clear, falsifiable claim. If C2 instead shows A100 batch efficiency close to H100 for some model or shape, the cascade result reverses, which is also publishable.

## F10. Reliability tails need correlation-aware headroom (`results/p3_mc.csv`)
We replayed each plan against sampled concurrent repair drains. This uses M/G/∞ occupancy: Poisson for independent failures, negative binomial for correlated ones.

| Plan | Correlation | Worst-quarter P(LC capacity < demand) | Quarters above 1% (of 72) |
|---|---|---|---|
| E0 no reuse | independent | 2.7% | 5 |
| E2b reliability-aware reuse | independent | 14.5% | 4 |
| E2b | correlated (k=2) | 15.5% | 14 |
| E2b | strongly correlated (k=0.5) | 15.5% | 44 |
| E0 | strongly correlated (k=0.5) | 8.8% | 48 |

*(Replay on the planned fractional fleet; an earlier version rounded cohorts and slightly overstated tails.)*
**Resolved in Phase 4** (`P4_FINDINGS.md`): chance-constrained headroom meets P(short) ≤ 0.1% at +0.3–0.8% carbon, and reuse keeps (and slightly grows) its saving.

- On average every plan meets the SLO (mean violation probability ≤ 3%). But the LP's **linear headroom rule** (LC capacity derated by 3× the expected drain) leaves **some region-quarters exposed**. Reuse plans have the worst single quarter, because aged parts drain more.
- **Correlated failures** multiply the number of exposed quarters, with or without reuse.
- **Phase 4 fix:** chance-constrained headroom. Iterate solve → Monte Carlo → raise the buffer where P(violation) exceeds the target → re-solve. Then report the carbon price of a given SLO under correlation. This turns "reliability" from an expected-value correction into an SLO guarantee, which is what an operator needs.

## F11. Accounting choices change Y by up to 2× (`results/p3_policy.csv`)

| Accounting for reused parts' embodied carbon | amortized | upfront |
|---|---|---|
| cutoff (reused parts free) | Y = 0.78% | 1.42% |
| residual (unamortized share charged) | 0.60% | 1.07% |
| full (charged as new) | 0 | 0 |

The ranking of plans is stable, but the **magnitude of any headline depends on the carbon-accounting convention**. The paper must state its convention and show this table. *Upfront* gives no credit for hardware life remaining at the horizon, which favours reuse; *amortized* is the fairer default.

## F12. Fuzzy uncertainty: reuse never hurts if planned for, but its magnitude is uncertain (`results/p3_fuzzy.csv`)
We used triangular fuzzy factors on all failure rates (0.5, 1, 3), all embodied-carbon values (0.5, 1, 2) and the CXL slowdown (0.85, 0.95, 1.0):

| α | Y interval |
|---|---|
| 1 (crisp) | 0.78% |
| 0.5 | [0.27%, 1.49%] |
| 0 (full support) | [0.00%, 2.64%] |

The lower bound never goes below 0: a reliability-aware planner cannot be pushed into net harm by these uncertainties. The upper bound is set mostly by embodied-carbon uncertainty, which is the next literature value to verify.

## F13. Utilization and headroom (`results/p3_headroom.csv`)
- Lower utilization raises Y (0.81% at util 0.5 vs 0.78% at 0.8), consistent with F7: embodied carbon matters more on lightly used capacity.
- In the LP, reused hosts hold a 20–30% share of node-quarters across buffer settings. Higher LC buffers slightly *lower* that share, because reused parts drain more and need more derating.
- F7's "reused hardware for headroom" rule is therefore only weakly supported at fleet scale. Testing it properly needs explicit standby nodes, which comes with the chance-constrained formulation in Phase 4.

## Carbon–cost Pareto front (`p3_pareto.png`)
Minimizing cost instead of carbon costs **+2.6% carbon** (487 → 500 ktCO₂e over 6 years) and saves **16% cost**. Nearly the whole carbon gain is available within about +5% cost of the cheapest plan.

## Next
1. **GPU measurements** (`gpu/run_all.sh` on one H100-class and one A100-class host). These resolve F9's cascade verdict and the CXL slowdown, which controls the sign of the result for mixed and dirty grids (P1 finding F2).
2. Chance-constrained headroom with the Monte Carlo replay in the loop (F10).
3. Verify the embodied-carbon literature values. They dominate the fuzzy upper bound (F12).
4. Write the paper bundle (`paper/claims.md`) and hand it to paper-orchestra.
