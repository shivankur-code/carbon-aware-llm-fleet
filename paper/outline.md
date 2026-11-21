# Paper outline (input for paper-orchestra)

**Title:** Does Reuse Survive Reliability? Lifecycle Optimization for Low-Carbon AI Inference Fleets

**Status:** GPU parameters PROVISIONAL (C2 pending); CXL slowdown PROVISIONAL (C3a pending). Numbers below are filled from `numbers.json`.

## Abstract (draft)
Reusing DRAM and SSDs from retired servers is one of the few sustainability levers shown to cut emissions at
cloud scale. But reuse studies evaluate a server design at a single point in time, while operators make
sequential buy, reuse, cascade and retire decisions over years — and reused parts are old, so they fail more.
We build a reliability-aware, multi-period lifecycle optimization for LLM-inference fleets that couples
age-dependent (Weibull) component failure models, repair drains and spare parts, reused-part harvest pools,
regional grid-carbon trajectories, cascading of old GPUs to batch tiers (including across regions), and
chance-constrained SLO headroom validated by Monte Carlo. On a 6-year, three-grid fleet we find that
(i) reuse savings for GPU inference hosts are strongly grid-dependent (5.8% on a clean grid vs
0.7% on a mixed grid) because operational carbon dominates; (ii) reuse survives reliability only when
planned for — a reliability-aware plan retains 91% of the saving and never does worse than no reuse,
whereas a failure-blind plan turns carbon-negative once reused parts fail ~2.1x more than their age
predicts; (iii) cascading old GPUs pays only above ~1.0x relative batch energy efficiency;
and (iv) guaranteeing a 99.9% SLO under correlated failures costs up to n/a extra carbon while preserving
n/a–n/a reuse savings. GPU measurements of serving efficiency per tier, KV-cache
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
