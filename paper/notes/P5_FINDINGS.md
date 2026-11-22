# Phase 5: the refurbish decision

Reproduce with:

```bash
python experiments/p5_refurbish.py
python experiments/p3_study.py ladder
```

The ladder now includes rungs E0r (refurbishment without reuse), E4 (reuse + refurbishment) and E5 (all levers).

## What was added
- **Refurbishment paths** are defined in `configs/skus.yaml` under `refurbish:`.
  - A host whose node age falls inside a window (e.g. 14–18 quarters) has some component classes replaced by new parts and continues for a second life of up to `max_age_q` quarters.
  - Each path has a capital cost and a downtime (both assumptions, with ranges).
  - Configured paths: memory only (`*_rfm`: DIMMs) for `new_gpu`, `green_gpu` and `old_gpu`; memory + storage (`new_gpu_rf`: DIMMs + SSDs).
- **One derived SKU per refurbishment age** (`new_gpu_rfm@14` … `@18`, in `hardware.py`).
  - Replaced parts are new.
  - Carried parts keep the parent's component model (including any reused-part hazard multiplier) at their exact age.
- **In the optimizer** (`milp.py`, `Scenario.allow_refurbish`):
  - a refurbishment flow moves hosts from the parent cohort at age `a` into the derived SKU at age 0;
  - the downtime is drained in the first quarter;
  - discarded first-life parts feed the reused-part pools in that quarter.
- **Accounting** is exact at every refurbishment age:
  - discarded parts: their unamortized share is stranded at refurbishment;
  - carried parts: they keep the parent's original amortization schedule;
  - new parts: they amortize over the second life.

  Tests check conservation (parent amortization + stranded share + the child's full schedule = parent embodied + new parts), node conservation, monotonicity (the extra option never raises the optimum), and the pool balance.

## Three accounting pitfalls found and fixed while building this
These make a useful methods point for the paper: refurbishment is easy to over-credit.
1. **Re-spreading carried amortization over the second life** pushed carbon past the horizon's terminal credit. Fleet "gain" was +0.366 pp, of which ~80% was artefact.
2. **Pool harvest from refurbishment discards was summed over all quarters** instead of the current one. This was a bug: it inflated the reused-DIMM supply ~24× and fabricated a clean-grid "doubling" of reuse savings (5.0% → 10.5%). A regression test now checks the pool balance.
3. **A midpoint age for each refurbishment window** let the optimizer pick the window edge that shifted charges past the horizon. It is replaced by one derived SKU per age, which is exact.

## Results (exact accounting)

| Setting | Refurbishment chosen? | Extra saving over reliability-aware reuse |
|---|---|---|
| Baseline parameters, any grid | no | 0 |
| Refurbishment without reuse (E0r) | no | 0 |
| New-DIMM AFR 3%/yr, reused parts ×3 / ×5 | 306 / 548 GreenSKU hosts (memory only) | +0.001 / +0.004 pp |
| New-DIMM AFR 5%/yr, reused parts ×3 / ×5 | 689 / 480 GreenSKU hosts (memory only) | +0.011 / +0.015 pp |
| Reused parts ×8 (no reuse chosen) | no | 0 |

(The sweep allows in-place cascading in all runs, so its Y differs slightly from the ladder's.)

## F15. Refurbishment is a reliability repair for reuse hosts, not a carbon lever
- With baseline parameters the planner never refurbishes. Buying new DIMMs (embodied carbon) plus downtime costs more than the failures it prevents, and a longer life does not matter within a 6-year horizon.
- Refurbishment appears only when recycled DIMMs in GreenSKU-style hosts fail heavily (AFR ≥ 3% with a reused-part multiplier ≥ 3). The planner then replaces just their memory, and the gain is at most 0.015 pp.
- Memory + storage refurbishment is never chosen: new SSDs carry too much embodied carbon (614 kgCO2e each in our configs).
- **Implication for the brief:** "refurbish" belongs in the decision set, and the answer is that it rarely pays in carbon. Its role is to rescue reuse hosts whose recycled memory turns out worse than expected. It complements the finding that reuse survives reliability only when planned for.

Parameters remain provisional: capital cost, downtime, DIMM/SSD embodied carbon and failure rates are assumptions; GPU parameters await measurement.
