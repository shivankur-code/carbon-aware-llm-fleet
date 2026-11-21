# Research Plan: Does Reuse Survive Reliability?
### Lifecycle optimization for low-carbon AI inference servers

This repo backs the research brief prepared for Prof. Akshitha Sriraman (CMU ECE). The brief has three empty slots, **X% / Y% / Z%**, and an empty **Figure 1**. This repo has to fill them with reproducible numbers.

> *Reuse cut lifecycle carbon by **X%** in a deterministic plan, but age-related failures and drain overhead eroded this to **Y%**; cascading older servers to batch inference restored **Z%**.*

---

## 1. Starting point and gap

**GreenSKU** (Wang, …, Sriraman, ISCA '24, IEEE Micro Top Pick) builds servers from energy-efficient CPUs, reused DRAM over CXL and reused SSDs. It reports about 28% lower per-core emissions, and about 8% lower net Azure emissions once datacenter overheads are counted. The paper evaluates each server design **at a single point in time**.

**What this project adds**, matching the brief's three gap bullets:
1. **Sequential decisions.** Operators decide buy / reuse / retire quarter by quarter over years. They do not pick one design once.
2. **Age-dependent reliability.** Reused DRAM and SSDs are already old when they enter the second life, so they sit further along their hazard curve. Each failure has costs:
   - it uses up a spare part, which carries embodied carbon;
   - it takes repair labour and logistics;
   - it **drains capacity** while the node is out for repair.

   These costs trade off against the reuse savings, and the drained capacity has to be covered by headroom, which also has embodied carbon.
3. **Cascading.** Older or degraded hardware moves to a **batch or low-priority inference tier**. That tier has looser SLOs, so failures cost less there, and the move is optimized *jointly* with procurement.

**The headline question:** is GreenSKU-style reuse still greener once reliability is priced in, and under which conditions (failure-rate range, grid carbon, tier mix)?

## 1b. Three contributions (what makes this more than a model)

**C1 – Reliability-aware lifecycle optimizer** (the brief's core). A multi-period MILP plus stochastic replay. It produces X/Y/Z, the carbon-optimal retirement age and the Pareto fronts.

**C2 – Measured value of cascading on real GPUs.** Cascading rests on one testable claim: *old GPUs lose less ground on batch inference than on latency-critical inference.* We measure this directly on real GPUs across generations (e.g. T4 / V100 / A10 / L4 / A100 / H100) running open LLMs with vLLM.
- **LC tier:** small batches under a TTFT/TPOT SLO (time to first token / time per output token).
- **BATCH tier:** large batches, throughput-bound.
- Power comes from NVML.
- Output: **tokens/J and gCO₂e per million tokens per generation × tier**, with embodied carbon amortized over a first or second life.

These measured curves replace guessed throughput and power parameters in C1. Lifecycle studies usually assume these numbers; we measure them.

**C3 – Reused DRAM as the KV-cache tier: "GreenSKU for AI".** This is the brief's "reliability-aware GreenSKU designs for AI servers". The KV cache of an LLM server is **recomputable state**, which makes it a natural home for reused, possibly flaky, host DRAM:
- (a) Offload KV blocks to host DRAM (vLLM CPU swap / LMCache-style). Throttle bandwidth and add latency to emulate CXL-attached reused DIMMs. Measure the throughput and SLO cost at each setting.
- (b) **Fault injection:** flip bits and corrupt or drop blocks in the offloaded KV cache at rates taken from DRAM field studies, scaled by age, and measure the effect on output quality (perplexity, task accuracy) and on recompute overhead.
  - With detection (checksum → recompute), a failure costs only performance.
  - Without detection, we measure how much quality is lost.

  The result is a **safe-deployment policy**: which inference data structures can live on aged, reused memory, and at what error rates.

C2 and C3 feed C1. The final optimizer uses measured throughput, energy and error tolerance, not assumptions.

## 2. Research questions
- **RQ1 – Erosion.** How much of the deterministic reuse saving survives once age-dependent failures, spares and drain overhead are counted? *(→ X, Y)*
- **RQ2 – Cascading.** How much of the lost saving does jointly optimized cascading to a batch tier win back? *(→ Z)*
- **RQ3 – Retirement age.** What is the carbon-optimal retirement or cascade age for each component generation, and how does it move with regional grid carbon (clean, mixed, dirty) and with decarbonization trajectories?
- **RQ4 – Trade-offs.** What do the Pareto fronts of carbon vs. cost vs. expected capacity shortfall look like, using ε-constraint?
- **RQ5 – Fragility.** Under fuzzy failure rates and fuzzy embodied-carbon values, where does "reuse is greener" flip? This gives a break-even map.

## 3. Model

### 3.1 State: cohorts by age
The fleet is a set of cohorts indexed by **(SKU s, tier τ, age a, refurbished flag)**. Time steps are quarters, t = 1…T, with T = 12–24 (3–6 years).

SKUs (all parameters in `configs/*.yaml`, each with a citation):
| SKU | Description |
|---|---|
| `new-gpu` | Current-generation GPU inference host, all new components |
| `green-gpu` | GreenSKU-style GPU host: efficient CPU, **reused DRAM over CXL**, **reused SSD** (components arrive at age a₀ from their first life) |
| `old-gpu` | Previous-generation host that came off the latency-critical tier (cascade candidate) |

Tiers: **LC** (latency-critical inference: strict SLO, so failure drains need N+k headroom) and **BATCH** (batch / offline inference: loose SLO, so failures mostly just reschedule).

### 3.2 Decisions in each quarter
`buy[s,τ,t]`, `keep`, `cascade[s,a,t]` (LC→BATCH), `refurbish[s,a,t]` (swap the worn component class, which resets that component's age and costs the new part's embodied carbon), `lease[τ,t]` (rented external capacity to cover shortfall, at a carbon and cost premium; this matches the leasing lever in the 2024 circular-economy model), `retire[s,τ,a,t]`, and **geographic cascade** `ship[s,a,r→r',t]` (move old hardware to a clean-grid region's BATCH tier; P1 finding F4).

### 3.3 Reliability (the key modelling step, and it stays linear)
- Each component class c ∈ {GPU, DRAM, SSD, CPU/board, PSU} gets a **Weibull** hazard h_c(a) = (β/η)(a/η)^(β−1), optionally a bathtub mixture with an infant-mortality term. A reused component starts at age a₀ > 0.
- Age is a **cohort index**, so the expected failures per node in each quarter, λ[s,a], are **precomputed parameters**. The MILP stays linear in node counts.
- Each failure costs: a spare (embodied carbon of the part), repair carbon and cost, and **drain time** d_c. Effective capacity is

  `cap_eff[s,τ,a] = n · u_s,τ · (1 − λ[s,a]·d/Δt) − headroom_τ`.

  The LC tier also needs extra headroom against correlated failures; BATCH does not.

### 3.4 Objectives (ε-constraint)
1. **Lifecycle carbon** = embodied (new nodes + refurbishment parts + spares) + operational (P · PUE · CI[region,t]) + repair/logistics − (optional) end-of-life credit.
2. **Total cost** (capex, opex, repair, lease).
3. **Expected capacity shortfall** against tier demand D[τ,t], with demand growing over time.

### 3.5 Uncertainty
- **Fuzzy** (triangular) failure scale η and embodied-carbon values. Solve at α-cuts and report the band of X/Y/Z. This is the same method as the 2024 fuzzy MILP paper.
- **Monte Carlo replay**: take the MILP's plan, sample actual failures and drains, and measure realized carbon and the SLO-violation distribution. This checks that the expected-value model is not over-optimistic.

## 4. Experiments → slots in the brief
| ID | Setup | Output |
|---|---|---|
| **E0** | No-reuse baseline: only `new-gpu`, no cascading, failures counted | reference carbon C₀ |
| **E1** | Reuse allowed, **failures ignored** (deterministic, the GreenSKU-style point view) | **X** = 1 − C₁/C₀ |
| **E2a** | Take E1's plan and evaluate it with failures + drains (Monte Carlo replay) | the erosion of a naive plan |
| **E2b** | Reuse allowed, reliability-aware MILP, **no cascading** | **Y** = 1 − C₂/C₀ |
| **E3** | E2b plus the cascade decision | **Z** = savings restored = (1 − C₃/C₀) − Y. Report the absolute saving too, and state the exact definition in the brief. |
| E4 | Sweep region CI × decarbonization path × a₀ | carbon-optimal retirement age (RQ3) |
| E5 | ε-constraint sweeps | Pareto fronts (RQ4) |
| E6 | Fuzzy α-cuts + break-even map over (η, embodied DRAM/SSD) | fragility (RQ5) |

**Figure 1** (for the brief): a waterfall or grouped bar of lifecycle carbon with bars **No reuse → Reuse (deterministic, X) → + failures & drains (Y) → + cascading (Z)**. Each bar is split into embodied / operational / repair+spares, with Monte Carlo error bars.

## 5. Data sources (public only; parameters swept wherever data is thin)
- **Embodied carbon:** GreenSKU (ISCA '24) and ACT (Gupta et al., ISCA '22); GPU-host LCAs such as vendor product carbon footprints.
- **DRAM failures in the field:** Schroeder et al. (SIGMETRICS '09), Sridharan et al. (SC '12/ASPLOS '15), Meza et al. (DSN '15).
- **SSD failures in the field:** Meza et al. (SIGMETRICS '15), Schroeder et al. (FAST '16), Backblaze SSD drive stats.
- **GPU / node failures:** Meta's Llama 3 training-cluster interruption breakdown and published ML-cluster reliability studies.
- **Grid CI:** Electricity Maps / Ember regional yearly averages plus decarbonization scenarios.
- **Power:** MLPerf Inference power results and vendor TDPs.

Every parameter goes in YAML with a `source:` field. `paper/parameters.md` is generated from those files.

## 6. Repo layout
```
configs/            SKU, component reliability, region, demand YAMLs (with sources)
src/carbonsched/
  hardware.py       SKU / component generations
  reliability.py    Weibull/bathtub -> λ[s,a], drain & spare costs
  grid.py           regional CI trajectories
  carbon.py         lifecycle carbon accounting
  milp.py           multi-period lifecycle MILP (PuLP + HiGHS), ε-constraint
  montecarlo.py     stochastic replay of plans
  fuzzy.py          α-cut fuzzy solves
experiments/        e0..e6 scripts -> results/*.csv, figures
tests/              accounting & reliability unit tests, tiny-instance MILP checks
paper/              claims.md, figures, parameters.md, related_work.md, refs.bib
```

## 7. Phases
| Phase | Deliverable |
|---|---|
| **P0** ✅ | Plan + skeleton |
| **P1** ✅ | Configs with cited/flagged parameters; `reliability.py` (Weibull, bathtub, delayed renewal) + `carbon.py` with 20 tests; node-level study + tornado → `paper/notes/P1_FINDINGS.md` |
| **P2** ✅ | `milp.py` (cohort MILP/LP, pools, geo-cascade, ε-constraint) + 11 tests incl. hand-checked case; E0–E3 pipeline; pilot X/Y/Z → `paper/notes/P2_FINDINGS.md` |
| **P3** | E0–E3 → **X, Y, Z + Figure 1** (this unblocks the brief) |
| **P4** | E4–E6, Monte Carlo validation, sensitivity sweeps |
| **P4g** (GPU) | C2: vLLM LC vs BATCH sweeps per GPU generation with NVML power; C3: KV offload under emulated CXL + fault injection |
| **P5** | `paper/` bundle → **paper-orchestra** |

## 8. Honesty and scoping
- This is a **modelling study** using public parameters. No Azure-internal data is used or implied. That matters because of the author's employer, and the brief and paper must say so explicitly.
- X/Y/Z depend on the parameters. The brief should quote them with their scenario, e.g. "under median field failure rates, mixed grid", and point to the fragility map.
- Related work to position against: GreenSKU/GSF; ACT; lifecycle and server-refresh studies (e.g. the "Junkyard Computing" reuse work, server lifetime-extension analyses); carbon-aware scheduling (CarbonScaler, Ecovisor), which is complementary to this work, not the same question.
