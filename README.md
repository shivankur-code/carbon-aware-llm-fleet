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
Phase 0: plan and skeleton. Module docstrings describe what each will contain.

## Headline (to be filled by experiments E0–E3)
> Reuse cut lifecycle carbon by **X%** in a deterministic plan, but age-related failures and drain overhead eroded this to **Y%**; cascading older servers to batch inference restored **Z%**.

*Modelling study using public parameters only; no proprietary fleet data.*
