# Related work (draft for paper-orchestra)

Citation keys refer to `paper/refs.bib`. Keys marked VERIFY there must be checked before submission.

## Low-carbon server design and embodied-carbon accounting
- **GreenSKU** [wang2024greensku] combines energy-efficient CPUs, reused DRAM over CXL and reused SSDs, evaluating server designs at fleet scale under production constraints. This is the work we extend.
- **ACT** [gupta2022act] and *Chasing Carbon* [gupta2021chasing] established that embodied carbon dominates for many computing systems.
- SSD manufacturing carbon [tannu2022ssd] makes storage reuse especially attractive.
- *Junkyard computing* [switzer2023junkyard] shows that second-life hardware can be greener when the workload fits it.

All of these evaluate a design at a point in time. **We add:**
1. sequential buy / reuse / cascade / retire decisions over years;
2. age-dependent reliability of reused parts and its fleet-level costs (spares, repair drains, SLO headroom);
3. AI-inference hosts, where operational carbon dominates.

## Field reliability of DRAM, flash and accelerators
- Large field studies characterize DRAM error rates and their growth with age [schroeder2009dram; sridharan2012dram; sridharan2015memory; meza2015dram].
- The same is true for SSDs [meza2015flash; schroeder2016flash].
- Recent reports quantify GPU and HBM failure rates in large AI clusters [llama3].
- Silent data corruption is now a fleet-scale concern [hochschild2021cores].

We use these studies to calibrate age-dependent hazard models. We then ask what those failure rates mean for the **carbon** case for reuse. That question has not been asked before, as far as we know.

## Carbon-aware operation
- Carbon-aware scheduling shifts *when* and *where* work runs [wiesner2021letswait; radovanovic2023cics; hanafy2023carbonscaler; souza2023ecovisor].
- Those systems take the hardware fleet as given. We optimize the **fleet itself over its lifecycle**, including geographic cascading of old hardware to clean grids. The two approaches are complementary.

## Heterogeneous and tiered LLM serving
- vLLM [kwon2023vllm] and phase-splitting / heterogeneous-GPU serving [patel2024splitwise; griggs2024melange] show that older or cheaper GPUs can serve parts of LLM inference well.
- Generative-AI carbon studies [chien2023genai] project large inference footprints.

We measure (C2) whether this heterogeneity makes **cascading** older GPUs carbon-positive, and find the break-even efficiency it must reach.

## Tiered and CXL memory
- CXL memory pooling and tiering [li2023pond; maruf2023tpp] motivate placing reused DRAM behind CXL, as GreenSKU does.
- Remote-NUMA emulation is the accepted proxy for CXL latency.

We measure the cost of keeping the LLM KV cache in such memory (C3a).

## Fault tolerance of DNN inference
- Fault-injection studies show that DNN accuracy is sensitive mainly to high-order (exponent) bit flips [li2017dnnerrors].

We apply this to the **KV cache** (C3b). The KV cache is recomputable state, so detected errors cost only recompute time. That makes it a natural tenant for aged, reused DRAM under standard SECDED ECC or block checksums.

## Multi-objective and fuzzy optimization for circular economies
- The lifecycle model follows the multi-objective fuzzy MILP structure for reuse, refurbishing, recycling and leasing decisions of [poonia2024circular]. We specialize it to datacenter hardware, with reliability-driven capacity constraints.
