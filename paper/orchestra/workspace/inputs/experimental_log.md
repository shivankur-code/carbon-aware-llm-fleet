# Experimental Log

## 1. Experimental Setup

* **Datasets / scenarios:**
  * *Node-level study:* one 8-GPU inference node per SKU over a 6-year service life at 60% utilization, quarterly steps, three grids. Grid carbon intensity in the base year was 40 gCO2e/kWh (clean, declining 2%/yr, floor 10), 250 gCO2e/kWh (mixed, declining 5%/yr, floor 20) and 650 gCO2e/kWh (dirty, declining 3%/yr, floor 50); PUE was 1.2.
  * *Pilot fleet:* 3 years, one region, an existing fleet of 24 previous-generation and 12 current-generation 8-GPU hosts, linearly growing LC and BATCH demand.
  * *Reference fleet:* 6 years (24 quarters), three regions (clean, mixed, dirty). LC demand started at 180 (clean), 450 (mixed) and 330 (dirty) node-equivalents and grew 25%, 25% and 30% per year; global BATCH demand started at 150 and grew 35% per year. The initial fleet held 150/500/400 previous-generation hosts (age 0 in their second life, parts 4 years old) and 100/350/250 current-generation hosts (ages 6/8/4 quarters) in clean/mixed/dirty, about 1,750 hosts in total. Reused-part pools started at 3,200 DIMMs and 800 SSDs and received 2,400 DIMMs and 600 SSDs per quarter from an external general-purpose fleet, with a 0.9 screening yield on locally harvested parts. One node-equivalent is one fully busy current-generation 8-GPU host.
* **SKUs:** `new_gpu`: 8 current-generation GPUs, 2 standard CPUs, 32 DIMMs (64 GB), 8 SSDs (3.84 TB), all new; idle 2,500 W, peak 10,200 W; platform embodied carbon 1,500 kgCO2e; 6-year design life. `green_gpu`: new GPUs, 2 energy-efficient CPUs, 32 reused DIMMs and 8 reused SSDs (parts 5 years old at entry); power 50 W below `new_gpu` at idle and 100 W below at peak; CXL throughput multiplier 0.95 (LC) and 0.98 (BATCH). `old_gpu`: previous-generation 8-GPU host cascaded as a whole (parts 4 years old), idle 1,800 W, peak 6,500 W, relative throughput 0.30 (LC) and 0.45 (BATCH).
* **Component parameters:** embodied carbon per unit was 150 kgCO2e (current GPU), 120 (previous GPU), 30 (CPU), 20 (DIMM), 614 (SSD). Weibull shape and annualized failure rate: GPU shape 1.0, AFR 0.0868 (derived from a published interruption report of a 54-day pretraining run on about 16,384 GPUs with 148 faulty-GPU and 72 GPU-memory interruptions, i.e. 0.0908 events per GPU-year); previous-generation GPU shape 1.3, AFR 0.0868; CPU shape 1.0, AFR 0.002; DIMM shape 1.6, AFR 0.01; SSD shape 1.3, AFR 0.01. Drain time per failure was 72 h (GPU), 24 h (CPU), 8 h (DIMM) and 2 h (SSD). Failed parts were replaced by new spares (delayed renewal process).
* **Parameter provenance:** the GPU failure rate was derived from published data; all other parameters were engineering assumptions with ranges and were swept. GPU throughput, power and the CXL slowdown were placeholders because GPU measurements had not yet been run.
* **Evaluation metrics:** lifecycle carbon (kgCO2e) split into embodied, operational, spares plus logistics, leased capacity and shipping; carbon saving of a reuse plan relative to a no-reuse plan, in percent: X (failures ignored in both), Y_naive (failure-blind reuse plan replayed with failures), Y (reliability-aware reuse plan), Z (additional saving from in-place cascading, percentage points), Zg (with geographic cascading). Monte Carlo metrics: per region-quarter probability that instantaneous LC capacity is below demand. Carbon price of an SLO: carbon of the chance-constrained plan relative to the expected-value plan.
* **Baselines compared:** no-reuse fleet (only new hardware can be bought, no cascading); failure-blind reuse plan; expected-value headroom (capacity derated by 3x the expected drain) versus chance-constrained headroom.
* **Implementation details:** cohort-by-age model solved with the HiGHS solver through a Python modeling layer. Embodied carbon was amortized over the design life, with the unamortized remainder charged on early retirement; reused parts carried zero embodied carbon (cut-off accounting) unless stated. LC headroom multiplier 3.0 and utilization cap 0.8 unless stated. Leased capacity carried 1.3x the carbon of a new host per unit work. Shipping a cascaded node between regions cost 300 kgCO2e. The mixed-integer formulation of the 36-node pilot hit a 300 s time limit without proving optimality; the linear relaxation solved in under 1 s, matched the integer results within about 0.05 percentage points on X and Y, and gave identical X, Y and Z at 1x and 50x fleet scale, so all fleet results used the linear relaxation. A full E0 to E3g ladder on the reference fleet solved in about 4 s. A test suite of 40 tests (including a hand-calculated single-SKU case) passed.

## 2. Raw Numeric Data

### Node-level 6-year lifecycle carbon, mixed grid, LC tier, failures counted (kgCO2e per node)

| SKU | Embodied | Operational | Spares | Logistics | Total | Failures per node |
|---|---|---|---|---|---|---|
| new_gpu | 8,312 | 96,662 | 1,263 | 28 | 106,265 | 10.62 |
| green_gpu | 2,860 | 95,511 | 1,542 | 33 | 99,945 | 15.18 |
| old_gpu | 0 | 62,344 | 1,844 | 52 | 64,240 | 18.42 |

### Node-level saving of green_gpu over new_gpu per unit of work (%), cut-off accounting

| Grid | Tier | Failures ignored | Failures counted |
|---|---|---|---|
| clean | LC | 18.23 | 16.04 |
| clean | BATCH | 20.73 | 18.61 |
| mixed | LC | 1.26 | 0.93 |
| mixed | BATCH | 4.28 | 3.96 |
| dirty | LC | -2.04 | -2.15 |
| dirty | BATCH | 1.08 | 0.98 |

### Node-level one-at-a-time sensitivity of the green_gpu saving (mixed grid, LC; base saving 0.93%)

| Parameter | Range | Saving at low end (%) | Saving at high end (%) | Swing (pp) |
|---|---|---|---|---|
| CXL throughput multiplier (LC) | 0.85 – 1.0 | -10.72 | 5.88 | 16.61 |
| green_gpu peak power delta (W) | -400 – 300 | 3.35 | -2.30 | 5.65 |
| SSD embodied carbon (kgCO2e) | 150 – 900 | -2.49 | 2.91 | 5.41 |
| Mixed-grid carbon intensity (g/kWh) | 150 – 350 | 3.80 | -0.41 | 4.20 |
| green_gpu idle power delta (W) | -300 – 200 | 2.28 | -0.41 | 2.69 |

### Pilot fleet savings, linear relaxation (%)

| Grid | X | Y_naive | Y | Z (pp) |
|---|---|---|---|---|
| clean | 8.65 | 7.07 | 7.49 | 0.00 |
| mixed | 0.97 | 0.33 | 0.85 | 0.00 |
| dirty | 0.18 | -0.33 | 0.15 | 0.00 |

### Reference fleet: lifecycle carbon by run (tCO2e over 6 years)

| Run | Total | Embodied | Operational | Spares + logistics | Leased |
|---|---|---|---|---|---|
| E0 no reuse, failures ignored | 484,413 | 24,134 | 460,279 | 0 | 0 |
| E0 no reuse, failures counted | 490,607 | 24,606 | 462,068 | 3,933 | 0 |
| E1 reuse, failures ignored | 480,305 | 18,871 | 460,937 | 0 | 497 |
| E2a failure-blind reuse plan replayed with failures | 489,652 | 18,871 | 453,457 | 4,220 | 13,104 |
| E2b reliability-aware reuse | 486,804 | 19,716 | 462,295 | 4,264 | 530 |
| E3 E2b + in-place cascading | 486,780 | 19,715 | 462,574 | 4,301 | 191 |
| E3g E2b + geographic cascading | 486,780 | 19,715 | 462,574 | 4,301 | 191 |

### Reference fleet savings (%)

| Fleet | X | Y_naive | Y | Z (pp) |
|---|---|---|---|---|
| three-grid fleet | 0.848 | 0.195 | 0.775 | 0.005 |
| clean grid only | 5.766 | 4.584 | 5.028 | 0.005 |
| mixed grid only | 0.675 | 0.039 | 0.568 | 0.000 |
| dirty grid only | 0.103 | -0.443 | 0.087 | 0.000 |

### Reused-part failure-rate multiplier sweep, reference fleet (%)

| Multiplier | X | Y_naive | Y |
|---|---|---|---|
| 0.5 | 0.848 | 0.289 | 0.863 |
| 1 | 0.848 | 0.195 | 0.775 |
| 2 | 0.848 | 0.015 | 0.621 |
| 3 | 0.848 | -0.154 | 0.475 |
| 5 | 0.848 | -0.467 | 0.209 |
| 8 | 0.848 | -0.888 | 0.000 |
| 12 | 0.848 | -1.386 | 0.000 |
| 20 | 0.848 | -2.251 | 0.000 |

### Cascade sweep: old-GPU BATCH throughput, reference fleet

| Old BATCH throughput | Old/new BATCH work per joule | Z (pp) | Zg (pp) |
|---|---|---|---|
| 0.30 | 0.467 | 0.000 | 0.000 |
| 0.45 | 0.701 | 0.005 | 0.005 |
| 0.60 | 0.935 | 0.027 | 0.029 |
| 0.75 | 1.168 | 0.109 | 0.184 |
| 0.90 | 1.402 | 0.189 | 0.321 |
| 1.05 | 1.635 | 0.268 | 0.432 |
| 1.20 | 1.869 | 0.346 | 0.521 |

### Monte Carlo replay of expected-value plans (headroom multiplier 3): LC capacity shortfall risk over 72 region-quarters

| Plan | Failure correlation | Worst region-quarter P(short) | Mean P(short) | Region-quarters above 1% |
|---|---|---|---|---|
| no reuse | independent | 0.0272 | 0.0020 | 5 |
| no reuse | negative binomial k=2 | 0.0508 | 0.0065 | 17 |
| no reuse | negative binomial k=0.5 | 0.0880 | 0.0257 | 48 |
| reliability-aware reuse | independent | 0.1447 | 0.0040 | 4 |
| reliability-aware reuse | negative binomial k=2 | 0.1546 | 0.0077 | 14 |
| reliability-aware reuse | negative binomial k=0.5 | 0.1548 | 0.0242 | 44 |

### Chance-constrained SLO headroom (validated by independent replay with 50,000 samples)

| Reused-part multiplier | Correlation | Target P(short) | Y under guaranteed SLO (%) | Y expected-value (%) | SLO carbon price, reuse (%) | SLO carbon price, no reuse (%) | Validated worst P(short), reuse | Iterations |
|---|---|---|---|---|---|---|---|---|
| 1 | independent | 0.01 | 0.786 | 0.775 | 0.217 | 0.229 | 0.0106 | 4 |
| 1 | independent | 0.001 | 0.788 | 0.775 | 0.301 | 0.314 | 0.0011 | 4 |
| 1 | k=2 | 0.01 | 0.786 | 0.775 | 0.292 | 0.303 | 0.0096 | 5 |
| 1 | k=2 | 0.001 | 0.792 | 0.775 | 0.427 | 0.445 | 0.0010 | 8 |
| 1 | k=0.5 | 0.01 | 0.795 | 0.775 | 0.477 | 0.497 | 0.0105 | 7 |
| 1 | k=0.5 | 0.001 | 0.812 | 0.775 | 0.785 | 0.823 | 0.0012 | 15 |
| 3 | independent | 0.01 | 0.485 | 0.480 | 0.223 | 0.229 | 0.0100 | 4 |
| 3 | independent | 0.001 | 0.484 | 0.480 | 0.310 | 0.314 | 0.0012 | 4 |
| 3 | k=2 | 0.001 | 0.490 | 0.480 | 0.435 | 0.445 | 0.0010 | 6 |
| 3 | k=0.5 | 0.001 | 0.500 | 0.480 | 0.803 | 0.823 | 0.0011 | 15 |

### Embodied-carbon accounting conventions, reference fleet (%)

| Reused-part charge | Amortization | X | Y_naive | Y |
|---|---|---|---|---|
| cut-off (zero) | amortized | 0.848 | 0.195 | 0.775 |
| cut-off (zero) | upfront | 1.544 | 0.918 | 1.423 |
| residual share | amortized | 0.667 | 0.029 | 0.601 |
| residual share | upfront | 1.181 | 0.562 | 1.067 |
| charged as new | amortized | 0.000 | -0.557 | 0.000 |
| charged as new | upfront | 0.000 | -0.506 | 0.000 |

### Fuzzy alpha-cuts of Y (failure rates x[0.5,1,3], embodied carbon x[0.5,1,2], CXL multiplier [0.85,0.95,1.0])

| Alpha | Y low (%) | Y high (%) |
|---|---|---|
| 0 | 0.000 | 2.638 |
| 0.5 | 0.266 | 1.489 |
| 1 | 0.775 | 0.775 |

### Headroom and utilization sweep (reliability-aware reuse plan)

| LC headroom multiplier | Utilization cap | Reuse-host share of node-quarters | Y (%) |
|---|---|---|---|
| 1 | 0.5 | 0.207 | 0.806 |
| 1 | 0.8 | 0.303 | 0.775 |
| 3 | 0.5 | 0.205 | 0.801 |
| 3 | 0.8 | 0.292 | 0.775 |
| 10 | 0.5 | 0.197 | 0.783 |
| 10 | 0.8 | 0.280 | 0.777 |

### Carbon-cost epsilon-constraint front, reference fleet

| Point | 6-yr carbon (ktCO2e) | 6-yr cost (billion USD) |
|---|---|---|
| cost-minimal | 499.5 | 1.437 |
| 2 | 491.0 | 1.477 |
| 3 | 489.1 | 1.518 |
| 4 | 488.0 | 1.559 |
| 5 | 487.4 | 1.599 |
| 6 | 487.0 | 1.640 |
| 7 | 486.8 | 1.680 |
| carbon-minimal | 486.8 | 1.719 |

## 3. Qualitative Observations

* For the new 8-GPU host on the mixed grid, operational carbon was about 91% of 6-year lifecycle carbon; reusing DRAM and SSDs cut the node's embodied carbon by about two thirds (8,312 to 2,860 kgCO2e) but changed total carbon by only about 6%.
* On the dirty grid the reuse-based host lost per unit of work in the LC tier because of the CXL slowdown, yet the integer pilot model still filled the last, lightly used node with a reuse-based host, because at low utilization embodied carbon dominated.
* At node level, counting failures, spares, logistics and drains reduced the reuse saving by only 0.1 to 2.2 percentage points. Over 6 years a new host saw 4.36 GPU, 5.43 DIMM, 0.81 SSD and 0.02 CPU failures (359 drained hours), and a reuse-based host 4.36 GPU, 9.67 DIMM, 1.13 SSD and 0.02 CPU failures (394 drained hours): the aged DIMMs nearly doubled DIMM failures, but GPU failures, common to both SKUs and with 72 h drains, dominated drained hours.
* The failure-blind reuse plan lost carbon mainly by leasing capacity to cover drained nodes (13,104 tCO2e of leased capacity versus 530 tCO2e for the reliability-aware plan) and by extra spares.
* As reused parts became less reliable, the reliability-aware planner bought fewer reuse-based hosts and stopped reusing entirely at a multiplier of about 8; it never produced a plan worse than no reuse. The failure-blind plan crossed zero at a multiplier of about 2.1.
* In-place and geographic cascading added essentially nothing at the placeholder old-GPU efficiency (old/new BATCH work per joule 0.70). Cascading began to pay only near energy parity (about 0.93 to 1.0); above parity, geographic cascading to the clean-grid BATCH tier added roughly 50 to 70% on top of in-place cascading.
* Expected-value headroom met the SLO on average but left individual region-quarters with up to 14.5% (independent) and 15.5% (correlated) probability of LC capacity shortfall for reuse plans; correlated failures greatly increased the number of exposed region-quarters for all plans.
* The chance-constrained loop converged in 4 to 15 iterations, with reserves growing monotonically; strongly correlated failures required about 2.6x the reserve of independent failures at the 0.1% target.
* Under a guaranteed SLO the reuse plan paid a slightly lower carbon price than the no-reuse plan, so the reuse saving increased (0.775% to 0.812%, and 0.480% to 0.500% with 3x worse reused parts). Reserve capacity is idle by design, and reuse-based hosts have the lowest embodied carbon.
* An earlier Monte Carlo replay that rounded each age cohort to whole nodes dropped planned capacity and overstated violation probabilities by up to 5x; replaying the planned fleet resolved the discrepancy.
* Under fuzzy uncertainty the lower bound of Y never went below zero; the upper bound was driven mostly by embodied-carbon uncertainty.
* Cost-minimal planning emitted 2.6% more carbon than carbon-minimal planning while costing 16% less; most of the carbon gain was available within about 5% extra cost over the cheapest plan.
* GPU measurements of serving efficiency per tier, KV-cache offload slowdown and KV-cache fault tolerance had been prepared but not executed; the conclusions about cascading and about the sign of reuse savings on mixed grids depend on those measurements.
