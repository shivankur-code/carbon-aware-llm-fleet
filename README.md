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
**Phase 1 done**: cited, sweepable parameters (`configs/`), reliability models (Weibull/bathtub, minimal-repair and delayed-renewal), carbon accounting, 20 tests, node-level study and sensitivity tornado.
Early (parameter-provisional) findings in [`paper/notes/P1_FINDINGS.md`](paper/notes/P1_FINDINGS.md): for GPU inference hosts, reuse savings are strongly grid-dependent, and the CXL-memory slowdown decides their sign.

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
pytest -q
python experiments/p1_node_lifecycle.py
python experiments/p1_tornado.py --region mixed
```

## Headline (to be filled by experiments E0–E3)
> Reuse cut lifecycle carbon by **X%** in a deterministic plan, but age-related failures and drain overhead eroded this to **Y%**; cascading older servers to batch inference restored **Z%**.

*Modelling study using public parameters only; no proprietary fleet data.*
