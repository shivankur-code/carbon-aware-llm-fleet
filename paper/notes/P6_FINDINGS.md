# Phase 6: unbundling the GreenSKU-style host

`experiments/p6_ablation.py` (results in `results/p6/`).

## Why
In the paper, `green_gpu` combines three changes at once:
- an energy-efficient CPU;
- reused DRAM behind CXL (this brings the CXL throughput multiplier and +100 kgCO2e of expander platform);
- reused SSDs (no throughput cost).

An independent reviewer of the paper-orchestra draft pointed out that, on the paper's own numbers, about 88% of the embodied saving comes from the SSDs, while all of the slowdown comes from the DRAM. Phase 6 separates the bundle into single-lever variants:

| variant | CPU | DRAM | SSD | CXL slowdown |
|---|---|---|---|---|
| `green_cpu` | efficient | new, local | new | none |
| `green_ssd` | efficient | new, local | reused (5 y) | none |
| `green_dram` | efficient | reused (5 y), over CXL | new | yes |
| `green_gpu` (paper) | efficient | reused (5 y), over CXL | reused (5 y) | yes |

Host power deltas are the same as `green_gpu` for every variant, because CXL expander power is not modelled separately. All GPU throughput and power figures are still placeholders.

## Node level
6-year life, 60% utilization, failures counted, cutoff accounting (the same as the paper; the paper's 0.93% mixed-LC result is reproduced). Saving per unit of work against `new_gpu`:

| | clean LC | clean BATCH | mixed LC | mixed BATCH | dirty LC | dirty BATCH |
|---|---|---|---|---|---|---|
| `green_cpu` | 0.72 | 0.72 | 1.02 | 1.02 | 1.08 | 1.08 |
| **`green_ssd`** | **18.55** | **18.55** | **5.46** | **5.46** | **2.80** | **2.80** |
| `green_dram` | -2.74 | 0.41 | -3.74 | -0.57 | -3.95 | -0.77 |
| `green_gpu` (paper) | 16.04 | 18.61 | 0.93 | 3.96 | -2.15 | 0.98 |

Embodied carbon per node:
- `new_gpu` 8,312 kgCO2e;
- `green_ssd` 3,400;
- `green_dram` 7,772;
- `green_gpu` 2,860.

Reused DRAM over CXL saves only about 540 kg net, and at a 5% slowdown it costs more operational carbon than that.

### CXL sweep (LC tier, `results/p6/node_cxl_sweep.csv`)
- DRAM-over-CXL reuse breaks even only when the slowdown is below about 2.4% (clean), 1.4% (mixed) or 1.2% (dirty).
- The full bundle beats SSD-only reuse only when the slowdown is below about 2% (clean) or 0.6% (mixed). On the dirty grid it never does.

## Fleet level
Reference fleet: 3 regions, about 1,750 hosts, 6 years, reliability-aware LP. The baseline E0 is `new_gpu` only, 490.6 ktCO2e.

| design the planner may buy | Y (no cascade) | with cascading (E3g) |
|---|---|---|
| bundle (paper) | 0.775% | 0.780% |
| efficient CPU only | 0.721% | 0.880% |
| **SSD reuse only** | **1.536%** | **1.615%** |
| DRAM/CXL reuse only | 0.005% | 0.005% |
| **planner may pick any variant** | **1.711%** | **1.870%** |

When free to choose, the planner buys `green_cpu` and `green_ssd` and **never** a DRAM-over-CXL host. SSD-only reuse is capped by the reused-SSD pool: it fills 1,900 hosts, the same number the bundle fills.

### Fleet CXL sweep (`results/p6/fleet_ablation_cxl_sweep.csv`)
| CXL multiplier (LC) | bundle | SSD only | DRAM/CXL only | planner picks | DRAM-over-CXL hosts bought by the free planner |
|---|---|---|---|---|---|
| 0.95 | 0.775 | 1.536 | 0.005 | 1.711 | 0 |
| 0.97 | 0.883 | 1.536 | 0.005 | 1.711 | 0 |
| 0.98 | 0.947 | 1.536 | 0.006 | 1.711 | 0 |
| 0.99 | 1.058 | 1.536 | 0.068 | 1.720 | 838 (178 `green_dram`, 660 `green_gpu`) |
| 1.00 | 1.600 | 1.536 | 0.615 | 1.789 | 1,900 |

## Findings
1. **The bundle hides a loser.** At the placeholder 5% CXL slowdown, reusing DRAM behind CXL is carbon-negative on every grid in the LC tier. Most of the bundle's fleet saving (0.721 of 0.775 pp) actually comes from the efficient CPU.
2. **Reusing SSDs without CXL doubles the fleet saving:** 0.775% → 1.536%. Letting the planner mix single-lever designs raises it to 1.711% (1.870% with cascading).
3. **DRAM-over-CXL reuse is adopted only if the CXL slowdown is at most about 1%.** Even at zero slowdown, it adds only 0.08 pp over the best plan without it.
4. So the number to measure is the **LC CXL slowdown in the 0-2% range**. That is the KV-offload benchmark in `gpu/`, which has been prepared but not run.

## Caveats
- GPU throughput, power and CXL slowdown are placeholders.
- The efficient CPU's power saving is an assumption, so finding 1 depends on it.
- Fleet results come from the LP relaxation.
