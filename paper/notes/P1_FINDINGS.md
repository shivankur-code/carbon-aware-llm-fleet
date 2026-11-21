# Phase 1 findings: single-node lifecycle study

Reproduce with:

```bash
python experiments/p1_node_lifecycle.py
python experiments/p1_tornado.py --region mixed
```

**Setup.** One 8-GPU inference node per server type, over a 6-year life at 60% utilization, with quarterly steps and a grid that decarbonizes over time. Every parameter is listed in `paper/parameters.md`. Almost all of them are still `assumption` values, so read these as **directions, not headline numbers**. This is a single-node study, not the fleet optimizer: it does not yet include headroom for the latency-critical (LC) tier, demand growth, or buy/retire timing.

## F1. GPU hosts are dominated by operational carbon, which shrinks GreenSKU's lever
On the mixed grid, a new 8-GPU host has about 97 t of operational carbon against about 8 t of embodied carbon over 6 years. Reusing DRAM and SSDs cuts embodied carbon by about two thirds (8.3 t → 2.9 t). But embodied carbon is only about 8% of the total.

| Grid | GreenSKU-style saving per unit of work (LC tier, cut-off accounting, failures counted) |
|---|---|
| clean (40 g/kWh) | **16.0%** |
| mixed (250 g/kWh) | **0.9%** |
| dirty (650 g/kWh) | **−2.2%** (worse than new) |

GreenSKU reported about 28% per-core savings on **CPU** servers, where embodied carbon is a much larger share. For **AI inference hosts**, our model says the benefit is strongly grid-dependent. That is a new, testable claim.

## F2. The verdict depends on the memory-slowdown penalty, so C3 is the top-priority measurement
In the sensitivity sweep (mixed grid, LC tier), the largest swing (16.6 percentage points) comes from the GreenSKU node's throughput penalty from CXL-attached reused memory. Over its range of 0.85–1.0, it alone flips the saving from −10.7% to +5.9%. Next come the host power difference (5.7 pp), SSD embodied carbon (5.4 pp) and grid carbon intensity (4.2 pp).
→ Contribution C3 (measuring KV-cache offload to slowed host memory on real GPUs) **decides the sign of the result**. It is not an add-on.
→ The SSD embodied-carbon value is the first literature number to verify.

## F3. At node level, failures erode savings only modestly
Counting age-dependent failures, spare parts, logistics and repair-drain downtime takes **0.1–2.2 percentage points** off the reuse saving. Most failures are GPU failures, which are the same on both node types: about 10.6 failures per node in 6 years for a new node versus 15.2 for a GreenSKU node. The extra failures come from the aged DIMMs and SSDs.
→ If reliability erodes savings substantially, the cost will come from **fleet-level effects** that the optimizer adds:
- N+k headroom in the LC tier;
- correlated failures;
- spare-pool sizing;
- forced early retirement.

This is the real test of "does reuse survive reliability?". The node-level answer is "mostly yes", which makes the fleet-level answer interesting either way.

## F4. Cascading old GPUs only pays on clean grids, which suggests *geographic cascading*
With placeholder A100-vs-H100 throughput and power, a cascaded A100-class node (embodied carbon already paid) uses more carbon per unit of work than a new H100 node:

| Grid | Batch tier | LC tier |
|---|---|---|
| clean | −8% | −62% |
| mixed | −35% | −103% |
| dirty | −41% | −111% |

The batch tier is far less bad than LC, which supports the cascade hypothesis. But operational inefficiency wins except on clean grids.
→ **New lever for the optimizer: cascade to a clean-grid region.** Old hardware is shipped to a low-carbon site's batch tier instead of being cascaded in place.
→ These numbers rest entirely on placeholder performance-per-watt. **Contribution C2 (GPU measurements) has to replace them** before any claim is made.

## Implications for the brief
- "Reuse cut lifecycle carbon by X%" will depend on the region. Expect to quote a range, e.g. "16% on a clean grid, about 1% on a mixed grid".
- Z (savings restored by cascading) may be **negative unless cascading is geographic**. That makes a sharper and more honest story than the template sentence. Write the brief's sentence **after** the Phase 3 results.

## Verification queue (sources the sandbox could not fetch)
GreenSKU (ISCA '24) embodied split and CXL overhead · Schroeder et al. (SIGMETRICS '09) DRAM age effects · Meza et al. (DSN '15) · Schroeder et al. (FAST '16) SSD replacement rates · Tannu & Nair (HotCarbon '22) SSD kg/GB · GPU vendor product carbon footprints. Dropping the PDFs into `data/papers/` is enough for them to be read and the configs updated.
