# Does Reuse Survive Reliability?
**Lifecycle optimization for low-carbon AI inference servers.**

GreenSKU (Wang, …, Sriraman, ISCA '24) showed that servers built from energy-efficient CPUs plus *reused* DRAM (over CXL) and *reused* SSDs cut per-core emissions by about 28%. That result evaluates a server design at a single point in time.

Real operators make **sequential** decisions over years: buy, reuse, cascade, refurbish, lease or retire. Reused parts are already old, so they fail more often. Every failure uses up a spare, needs repair work and **drains capacity**. This repo asks how much of the reuse saving survives once those costs are counted, and how much cascading older servers to a batch-inference tier wins back.

**Approach:** a multi-period (quarterly, 3–6 year) lifecycle MILP over server cohorts by age:
- Weibull/bathtub failure models per component generation;
- regional grid-carbon trajectories;
- objectives: carbon, cost and capacity shortfall (ε-constraint Pareto fronts);
- fuzzy α-cuts for uncertain failure and embodied-carbon values;
- Monte Carlo replay to validate plans.

See **[PLAN.md](PLAN.md)** for the model, experiments (E0–E6) and data sources.

## Status
**Phases 1–4 done; GPU experiments ready to run** (`gpu/README.md`). **Draft paper: [`paper/paper.pdf`](paper/paper.pdf)** (written with the PaperOrchestra pipeline; audit trail in `paper/orchestra/`).
Provisional findings (placeholder GPU parameters) – [`P3_FINDINGS.md`](paper/notes/P3_FINDINGS.md):
- Reuse survives reliability **if the plan accounts for it**: a reliability-aware plan keeps ~90% of the saving and degrades gracefully to zero as reused parts get worse (~8x); a failure-blind plan turns carbon-negative at ~2x the age-predicted failure rate.
- Reuse savings for GPU inference hosts are strongly grid-dependent (≈5% clean, <1% mixed/dirty) because operational carbon dominates.
- Cascading old GPUs saves carbon only if their batch work-per-joule nears the new GPU's (~0.9–1.0x); geographic cascading to clean grids adds ~50–70% on top. C2 measures this.
- With chance-constrained headroom (P(short) ≤ 0.1% per region-quarter, correlated failures), reuse still saves – slightly *more* – because reused hardware is the cheapest way to hold idle reserve; the SLO itself costs 0.2–0.8% carbon ([`P4_FINDINGS.md`](paper/notes/P4_FINDINGS.md)).

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
pytest -q
python experiments/p1_node_lifecycle.py
python experiments/p1_tornado.py --region mixed
python experiments/p2_pipeline.py --region mixed --lp
python experiments/p3_study.py all
python experiments/p4_chance.py
python paper/build_bundle.py
```

GPU experiments: see [`gpu/README.md`](gpu/README.md) (`LABEL=H100-SXM bash gpu/run_all.sh`).

## Headline (to be filled by experiments E0–E3)
> Reuse cut lifecycle carbon by **X%** in a deterministic plan, but age-related failures and drain overhead eroded this to **Y%**; cascading older servers to batch inference restored **Z%**.

*Modelling study using public parameters only; no proprietary fleet data.*
