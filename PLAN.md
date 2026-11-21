# Research Plan: Grid-Aware Scheduling for Low-Carbon Server Fleets

## 1. Starting point: what GreenSKU established

Wang, Sriraman et al., *Designing Cloud Servers for Lower Carbon* (ISCA '24, IEEE Micro Top Pick 2025):

- **GreenSKU** server designs lower lifecycle carbon by pairing energy-efficient CPUs with **reused DRAM (attached via CXL)** and **reused SSDs**.
- **GSF**, a framework for evaluating server-design carbon at fleet scale. With Azure's datacenter overheads included, the paper reports about an **8% cut in net cloud emissions**.
- Follow-on work: **EcoScale** (OSDI '25 poster).

**The open gap.** GreenSKU's carbon benefit is evaluated at **provisioning time**. The analysis assumes a fixed (average) grid carbon intensity and a static fleet mix. In practice:

1. Grid carbon intensity (CI) changes by hour and by region. In some grids it swings 3–10× within a day.
2. A GreenSKU has **lower embodied carbon** but can have **higher operational energy per unit of work**. CXL memory latency means more servers-hours, or more power, for the same throughput, and reused parts may fail more often.
3. So which SKU is "greenest" for a given job **depends on the CI at the moment the job runs**:
   - On a clean grid, embodied carbon dominates, so GreenSKU wins clearly.
   - On a dirty grid, operational carbon dominates, so a newer, more energy-efficient baseline server can win.

No one has yet studied **runtime, CI-aware placement across a mixed GreenSKU/baseline fleet**. That question is also what turns GreenSKU from a hardware SKU into a scheduler co-design problem.

## 2. Thesis of this project

> The carbon-optimal server for a job is not fixed. It flips with grid carbon intensity. A scheduler that routes work between GreenSKU and baseline servers based on real-time and forecast CI, and that shifts delay-tolerant work in time, can capture **meaningful extra savings beyond static GreenSKU provisioning**. It also changes **what fleet mix providers should buy** in each region.

The target is something Prof. Sriraman's group would want to read: it builds directly on GSF's carbon model and extends it on a dimension (time-varying CI × hardware heterogeneity) that the original paper held fixed.

## 3. Research questions

- **RQ1 – Crossover.** For each workload class, at what grid CI does a GreenSKU stop being lower-carbon than a baseline SKU? How sensitive is that point to the CXL slowdown, the reused-part failure rate and the embodied-carbon amortization period?
- **RQ2 – Runtime placement.** Given a fixed mixed fleet, how much carbon does CI-aware routing save compared with CI-oblivious placement, while still meeting SLOs?
- **RQ3 – Temporal + hardware co-scheduling.** What does combining SKU routing with temporal shifting of deferrable jobs add, compared with either one alone?
- **RQ4 – Provisioning feedback.** If the scheduler is CI-aware, does the carbon-optimal GreenSKU *fraction* of the fleet change per region? Compare a clean grid (e.g. Sweden/Quebec), a mixed grid (CAISO) and a dirty grid (e.g. PJM/India).
- **RQ5 – Robustness.** How much of the gain survives CI forecast error and capacity limits? How much stays when GreenSKU capacity is scarce?

## 4. Method

### 4.1 Models (all parameterized and cited, nothing invented)
| Component | Source / approach |
|---|---|
| Embodied carbon per SKU (CPU, DRAM, SSD, chassis) | GreenSKU/GSF paper values; ACT (Gupta et al.) model as cross-check; reused parts = 0 embodied or residual-amortized (sensitivity knob) |
| Power vs. utilization curve | SPECpower-style linear/piecewise models per CPU generation |
| Performance penalty | CXL memory slowdown per workload class (memory-bound vs. compute-bound), from published CXL characterization papers |
| Failure / reliability | Reused DRAM/SSD AFR as a knob, with failures costing re-execution energy and spare capacity |
| Grid CI | Hourly public traces: Electricity Maps free datasets, WattTime/CAISO/EIA historic data, at least 3 regions × 1 year |
| Workloads | Azure Public Dataset (VM traces 2019, Azure Functions 2019/2021); jobs labeled latency-critical vs. deferrable |

### 4.2 Policies to compare
1. **Baseline-only fleet**, CI-oblivious (reference).
2. **Static GreenSKU mix** (GSF-style), CI-oblivious bin-packing. This reproduces the paper's setting.
3. **CI-aware SKU routing**, greedy: route each job to the SKU with the lowest marginal (operational + amortized embodied) carbon at the current CI, subject to capacity and SLO.
4. **CI-aware routing + temporal shifting** for deferrable jobs, within a deadline slack.
5. **Offline MILP oracle** (PuLP/HiGHS) with perfect CI foresight, as an upper bound.
6. *(stretch)* **Forecast-driven MPC** with a realistic CI forecaster, to show the gap to the oracle.

### 4.3 Metrics
Total gCO₂e (split into operational and embodied), carbon per unit of work, SLO violation rate, p99 latency inflation, energy, and server-hours.

## 5. Deliverables (repo layout)

```
src/carbonsched/
  skus.py        # SKU carbon/power/perf models (YAML-configurable)
  grid.py        # CI trace loaders + forecaster
  workload.py    # Azure trace loaders, job classes
  carbon.py      # GSF-compatible carbon accounting
  simulator.py   # discrete-time fleet simulator
  policies/      # 5–6 policies above
experiments/     # one script per RQ -> results/*.csv + figures
data/raw/        # download scripts only (traces not committed)
paper/           # figures, tables, claims.md -> input to paper-orchestra
```

Key figures planned:
1. **Crossover map**: CI (x-axis) × CXL slowdown (y-axis), colored by which SKU is greener (RQ1).
2. Carbon savings by policy × region (RQ2/RQ3).
3. Optimal GreenSKU fleet fraction versus regional mean CI and CI variance (RQ4).
4. Savings versus forecast error (RQ5).

## 6. Phases

| Phase | Output | Est. |
|---|---|---|
| **P0 Scaffold** (done) | repo, plan, package layout | — |
| **P1 Models + data** | SKU YAMLs with cited parameters, CI and Azure trace downloaders, unit tests for carbon accounting | 1–2 days |
| **P2 Reproduce GSF-style baseline** | Policy 2 reproduces the *direction and rough magnitude* of GreenSKU's savings under average CI. This is the sanity anchor. | 1 day |
| **P3 Advancement experiments** | Policies 3–5, RQ1–RQ5 figures, sensitivity sweeps | 2–3 days |
| **P4 Paper handoff** | `paper/claims.md` (each claim → figure/CSV), `paper/figures/`, related-work notes → feed to **paper-orchestra** | 0.5 day |

## 7. Honest scoping notes
- This is a **trace-driven simulation study**, not a production deployment. The paper should say so plainly.
- GreenSKU's exact internal Azure parameters are not public. We use the published numbers and run sensitivity sweeps instead of claiming to reproduce Azure exactly.
- The novelty claim is "CI-aware runtime placement × reused-hardware heterogeneity". Before writing, check this against carbon-aware scheduling work (e.g. CarbonScaler, Let's Wait Awhile, Ecovisor, CASPER, carbon-aware Kubernetes) so that related work is positioned correctly.

## 8. Handoff to paper-orchestra
Phase 4 writes `paper/` with: an abstract seed, `claims.md` mapping each claim to evidence, final figures (PDF/PNG), `results/*.csv`, `related_work.md` and BibTeX. paper-orchestra then turns that into the readable paper.
