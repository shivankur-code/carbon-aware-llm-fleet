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
**Phase 2 done** – lifecycle MILP/LP with reuse pools, cascading, leasing and failure-aware headroom; E0–E3 pipeline producing X/Y/Z and a Figure 1 prototype.
Provisional headline (pilot, placeholder GPU parameters): a reliability-aware plan keeps ~87% of reuse savings on every grid, while a failure-blind plan keeps 34% on a mixed grid and turns negative on a dirty one – see [`paper/notes/P2_FINDINGS.md`](paper/notes/P2_FINDINGS.md) and [`P1_FINDINGS.md`](paper/notes/P1_FINDINGS.md).

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
pytest -q
python experiments/p1_node_lifecycle.py
python experiments/p1_tornado.py --region mixed
python experiments/p2_pipeline.py --region mixed --lp
```

## Headline (to be filled by experiments E0–E3)
> Reuse cut lifecycle carbon by **X%** in a deterministic plan, but age-related failures and drain overhead eroded this to **Y%**; cascading older servers to batch inference restored **Z%**.

*Modelling study using public parameters only; no proprietary fleet data.*
