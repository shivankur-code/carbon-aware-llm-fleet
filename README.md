# carbon-aware-scheduling

**Grid-aware workload placement across mixed GreenSKU / baseline server fleets.**

GreenSKU (Wang, Sriraman et al., ISCA '24) showed that cloud servers built from energy-efficient CPUs plus *reused* DRAM (over CXL) and *reused* SSDs cut fleet carbon. That evaluation is done at provisioning time, with grid carbon intensity held fixed.

This project asks what happens **at runtime**. Which server SKU has the lowest carbon for a given job flips with the grid's hourly carbon intensity:

- clean grid → embodied carbon dominates → reused-hardware GreenSKUs win;
- dirty grid → operational carbon dominates → newer, more energy-efficient servers can win.

We build a trace-driven simulator (Azure public traces × hourly regional grid-carbon traces). It quantifies how much extra carbon a CI-aware scheduler saves over static GreenSKU provisioning, and how that changes the optimal fleet mix in each region.

See **[PLAN.md](PLAN.md)** for research questions, method, and phases.

## Status
Phase 0: scaffold and plan. The modules in `src/carbonsched/` are stubs.

## Layout
```
src/carbonsched/   models, simulator, policies
experiments/       one script per research question
data/raw/          download scripts (traces not committed)
paper/             figures + claims for paper-orchestra
```
