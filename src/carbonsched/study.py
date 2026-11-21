"""Shared experiment logic: the E0-E3 ladder that produces the brief's X / Y / Z.

  E0_det  no reuse, failures ignored            E0   no reuse, failures counted
  E1      reuse, failures ignored               -> X = 1 - C(E1)/C(E0_det)
  E2a     E1's plan replayed with failures      -> Y_naive = 1 - C(E2a)/C(E0)
  E2b     reuse, reliability-aware, no cascade  -> Y = 1 - C(E2b)/C(E0)
  E3      E2b + in-place cascading LC->BATCH    -> Z = saving(E3) - Y   (pp)
  E3g     E3 + geographic cascading             -> Zg = saving(E3g) - Y (pp; only if >1 region)
"""
from __future__ import annotations

import copy

import pandas as pd

from .grid import Region
from .hardware import Catalog
from .milp import LifecycleModel, Scenario, Solution


def run_ladder(base: dict, cat: Catalog, regions: dict[str, Region], time_limit: float = 300,
               rungs: tuple[str, ...] = ("E0_det", "E0", "E1", "E2a", "E2b", "E3", "E3g")) -> tuple[dict, pd.DataFrame, dict[str, Solution]]:
    no_reuse = [s for s in base["buyable"] if not cat.skus[s].is_reuse]

    def mk(**kw) -> Scenario:
        d = copy.deepcopy(base)
        d.update(kw)
        return Scenario(**d)

    def solve(sc: Scenario, **kw) -> Solution:
        sol = LifecycleModel(cat, regions, sc).solve(time_limit=time_limit, **kw)
        if sol.status != "Optimal":
            raise RuntimeError(f"{sc.name}: {sol.status}")
        return sol

    multi = len(base["regions"]) > 1
    runs: dict[str, Solution] = {}
    if "E0_det" in rungs:
        runs["E0_det"] = solve(mk(buyable=no_reuse, allow_cascade=False, allow_geo_cascade=False, reliability=False))
    runs["E0"] = solve(mk(buyable=no_reuse, allow_cascade=False, allow_geo_cascade=False, reliability=True))
    if "E1" in rungs or "E2a" in rungs:
        runs["E1"] = solve(mk(allow_cascade=False, allow_geo_cascade=False, reliability=False))
    if "E2a" in rungs:
        runs["E2a"] = solve(mk(allow_cascade=False, allow_geo_cascade=False, reliability=True),
                            fix={"buy": runs["E1"].values["buy"], "ret": runs["E1"].values["ret"]})
    runs["E2b"] = solve(mk(allow_cascade=False, allow_geo_cascade=False, reliability=True))
    if "E3" in rungs:
        runs["E3"] = solve(mk(allow_cascade=True, allow_geo_cascade=False, reliability=True))
    if "E3g" in rungs and multi:
        runs["E3g"] = solve(mk(allow_cascade=True, allow_geo_cascade=True, reliability=True))

    C = {k: s.carbon_kg for k, s in runs.items()}
    sav = lambda k, ref: 100 * (1 - C[k] / C[ref])  # noqa: E731
    h: dict = dict(scenario=base.get("name", "?"), regions=",".join(base["regions"]))
    if "E1" in runs and "E0_det" in runs:
        h["X"] = sav("E1", "E0_det")
    if "E2a" in runs:
        h["Y_naive"] = sav("E2a", "E0")
    h["Y"] = sav("E2b", "E0")
    if "E3" in runs:
        h["saving_E3"] = sav("E3", "E0")
        h["Z_pp"] = h["saving_E3"] - h["Y"]
    if "E3g" in runs:
        h["saving_E3g"] = sav("E3g", "E0")
        h["Zg_pp"] = h["saving_E3g"] - h["Y"]
    h["solve_seconds"] = sum(s.solve_seconds for s in runs.values())
    h["time_limited_runs"] = [k for k, s in runs.items() if s.time_limited]
    table = pd.DataFrame({k: dict(s.totals) for k, s in runs.items()}).T
    return h, table, runs
