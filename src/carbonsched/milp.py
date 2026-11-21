"""Multi-period lifecycle MILP: buy / keep / cascade (incl. geographic) / lease / retire.

State
-----
``n[s, tier, r, a, t]`` – nodes of SKU ``s`` serving ``tier`` in region ``r``, at
node age ``a`` (quarters in the current life), during quarter ``t``.

Flow (per s, tier, r, a, t)::

    n[a, t] = carried[a, t] - retire[a, t] - cascade_out[a, t] + cascade_in[a, t] + (a == 0) * buy[t]
    carried[a, t] = n[a-1, t-1]   (initial fleet at t = 0)

Nodes reaching ``max_age`` must retire (``n[max_age] = 0``). Cascading moves a
node LC -> BATCH, optionally to another region (shipping carbon / cost); it
keeps its age.

Work and capacity
-----------------
``w[s, tier, r, a, t]`` is the work served, in *tier-specific node-equivalent
quarters* (1 = one ``new_gpu`` node fully busy for a quarter)::

    w <= n * throughput_eff * util_cap * (1 - buffer_tier * availability_loss[s, a])

``buffer_LC`` > 1 approximates the extra N+k headroom the latency-critical tier
keeps against concurrent failures; BATCH uses the expected loss only.
LC demand is per region (latency-bound); BATCH demand is global (any region).
Unserved demand is leased (carbon and cost premium) or left as shortfall.

Reused-part pools
-----------------
GreenSKU-style builds need harvested DIMMs/SSDs. Pools are filled by an
external supply (e.g. the CPU fleet's retirements) plus a screening ``yield``
times the first-life parts of nodes retired here, and drained by builds.

Objectives
----------
``carbon`` (kgCO2e), ``cost`` (USD), ``shortfall`` (node-eq-quarters). One is
minimized; the others can be bounded (epsilon-constraint).

Embodied accounting
-------------------
``amortized`` (default): a node's purchase embodied carbon is spread linearly
over its design life; retiring early charges the unamortized remainder
(stranded carbon); life remaining past the horizon is not charged. This avoids
end-of-horizon bias. ``upfront`` charges everything at purchase.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pulp
import yaml

from .carbon import purchase_embodied_kg
from .grid import Region
from .hardware import SKU, TIERS, Catalog
from .reliability import HOURS_PER_YEAR, QUARTER, QuarterReliability, node_reliability_table

HOURS_Q = HOURS_PER_YEAR * QUARTER
OBJECTIVES = ("carbon", "cost", "shortfall")


@dataclass
class Scenario:
    horizon_q: int
    regions: list[str]
    demand: dict[str, Any]                       # {"LC": {region: [..per q..]}, "BATCH": [..per q..]}
    skus: list[str]                              # SKUs in scope
    buyable: list[str]                           # SKUs that can be bought at age 0
    max_age_q: dict[str, int]                    # forced retirement age (quarters, current life)
    initial_fleet: list[dict] = field(default_factory=list)  # {sku, tier, region, age_q, count}
    util_cap: float = 0.8
    lc_failure_buffer: float = 3.0
    reliability: bool = True
    embodied_policy: str = "cutoff"
    embodied_accounting: str = "amortized"
    allow_cascade: bool = True
    allow_geo_cascade: bool = False
    allow_lease: bool = True
    lease_carbon_mult: float = 1.3               # leased work carbon vs new_gpu at util_cap in that region
    lease_cost_per_work: float = 60000.0         # USD per node-eq-quarter
    shortfall_penalty: float = 1e7               # USD-equivalent per node-eq-quarter, only for the cost objective
    shortfall_carbon_penalty: float = 1e6        # kg-equivalent per node-eq-quarter, only for the carbon objective
    energy_price_usd_per_kwh: float = 0.08
    cascade_cost_usd: float = 3000.0
    ship_kg_per_node: float = 300.0
    ship_cost_usd_per_node: float = 5000.0
    pools: dict[str, dict] = field(default_factory=dict)   # component -> {initial, external_per_q, yield}
    buy_regions: list[str] | None = None
    integer: bool = True
    scale: float = 1.0                           # multiplies demand, initial fleet and pools (fleet size)
    name: str = "scenario"

    def __post_init__(self) -> None:
        if self.scale != 1.0:
            k = self.scale
            mul = lambda d: [x * k for x in d] if isinstance(d, list) else d * k  # noqa: E731
            lc = self.demand.get("LC", {})
            self.demand = {"LC": {r: mul(v) for r, v in lc.items()}, "BATCH": mul(self.demand.get("BATCH", 0.0))}
            self.initial_fleet = [dict(f, count=f["count"] * k) for f in self.initial_fleet]
            self.pools = {c: {kk: (v * k if kk in ("initial", "external_per_q") else v) for kk, v in pool.items()}
                          for c, pool in self.pools.items()}
            self.scale = 1.0  # applied once

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "Scenario":
        raw = yaml.safe_load(Path(path).read_text())
        raw.update(overrides)
        return cls(**raw)

    def demand_at(self, tier: str, t: int, region: str | None = None) -> float:
        d = self.demand.get(tier, 0.0)
        if tier == "LC":
            d = d.get(region, 0.0) if isinstance(d, dict) else 0.0
        if isinstance(d, (int, float)):
            return float(d)
        return float(d[t]) if t < len(d) else float(d[-1])


@dataclass
class Solution:
    status: str
    objective: float
    totals: dict[str, float]
    fleet: pd.DataFrame          # one row per (s, tier, r, a, t) with n>0
    decisions: pd.DataFrame      # buys, retires, cascades, leases, shortfalls per t
    values: dict[str, dict]      # raw decision values, for fixing in re-evaluation
    solve_seconds: float = 0.0
    time_limited: bool = False   # True if an integer solve hit the time limit (may not be optimal)

    @property
    def carbon_kg(self) -> float:
        return self.totals["carbon_kg"]


class LifecycleModel:
    def __init__(self, catalog: Catalog, regions: dict[str, Region], sc: Scenario):
        self.cat, self.regions, self.sc = catalog, regions, sc
        self.T = range(sc.horizon_q)
        self.R = sc.regions
        self.S = sc.skus
        self.A = {s: range(sc.max_age_q[s] + 1) for s in self.S}
        self.rel: dict[str, list[QuarterReliability]] = {}
        for s in self.S:
            n_ages = sc.max_age_q[s] + 1
            self.rel[s] = (node_reliability_table(catalog.skus[s].slots, n_ages) if sc.reliability
                           else [QuarterReliability() for _ in range(n_ages)])
        self._coeffs()

    # ---------- precomputed coefficients ----------
    def _coeffs(self) -> None:
        sc = self.sc
        self.E = {s: purchase_embodied_kg(self.cat.skus[s], sc.embodied_policy) for s in self.S}
        self.life_q = {s: int(round(self.cat.skus[s].design_life_years * 4)) for s in self.S}

    def sku(self, s: str) -> SKU:
        return self.cat.skus[s]

    def loss(self, s, a):
        return self.rel[s][a].availability_loss

    def cap(self, s, tier, a):
        buf = self.sc.lc_failure_buffer if tier == "LC" else 1.0
        return self.sku(s).effective_throughput(tier) * self.sc.util_cap * max(0.0, 1.0 - buf * self.loss(s, a))

    def ci(self, r, t):
        return float(self.regions[r].ci(t))

    def pue(self, r):
        return self.regions[r].pue

    def idle_kwh(self, s, a):
        return self.sku(s).power_idle_w / 1000.0 * HOURS_Q * (1.0 - self.loss(s, a))

    def dyn_kwh_per_work(self, s, tier):
        k = self.sku(s)
        return (k.power_peak_w - k.power_idle_w) / 1000.0 * HOURS_Q / k.effective_throughput(tier)

    def embodied_per_q(self, s, a):
        if self.sc.embodied_accounting != "amortized":
            return 0.0
        return self.E[s] / self.life_q[s] if a < self.life_q[s] else 0.0

    def stranded(self, s, a):
        if self.sc.embodied_accounting != "amortized":
            return 0.0
        return self.E[s] * max(0.0, 1.0 - a / self.life_q[s])

    def lease_kg_per_work(self, tier, r, t):
        ref = "new_gpu" if "new_gpu" in self.cat.skus else self.S[0]
        k = self.cat.skus[ref]
        p = k.power_w(self.sc.util_cap)
        kwh = p / 1000.0 * HOURS_Q / (k.effective_throughput(tier) * self.sc.util_cap)
        emb = purchase_embodied_kg(k, "full") / (k.design_life_years * 4) / self.sc.util_cap
        return self.sc.lease_carbon_mult * (kwh * self.pue(r) * self.ci(r, t) / 1000.0 + emb)

    # ---------- model ----------
    def build(self, objective: str = "carbon", bounds: dict[str, float] | None = None,
              fix: dict[str, dict] | None = None) -> pulp.LpProblem:
        sc = self.sc
        cat_int = pulp.LpInteger if sc.integer else pulp.LpContinuous
        prob = pulp.LpProblem(f"lifecycle_{sc.name}", pulp.LpMinimize)
        T, R, S = self.T, self.R, self.S
        buy_regions = sc.buy_regions or R

        # Only create variables for reachable (age, quarter) pairs: a node can be at age a in
        # quarter t only if a <= t + (oldest initial age of that SKU). Cuts model size several-fold.
        amax0 = {s: max([int(f["age_q"]) for f in sc.initial_fleet if f["sku"] == s], default=0) for s in S}
        reach = lambda s, a, t: a <= t + amax0[s]  # noqa: E731
        idx = [(s, tau, r, a, t) for s in S for tau in TIERS for r in R for a in self.A[s] for t in T if reach(s, a, t)]
        n = pulp.LpVariable.dicts("n", idx, lowBound=0, cat=cat_int)
        w = pulp.LpVariable.dicts("w", idx, lowBound=0)
        ret = pulp.LpVariable.dicts("ret", idx, lowBound=0, cat=cat_int)
        buy_idx = [(s, tau, r, t) for s in sc.buyable for tau in TIERS for r in buy_regions for t in T]
        buy = pulp.LpVariable.dicts("buy", buy_idx, lowBound=0, cat=cat_int)
        casc_idx = []
        if sc.allow_cascade:
            for s in S:
                for r in R:
                    for r2 in R:
                        if r2 != r and not sc.allow_geo_cascade:
                            continue
                        for a in self.A[s]:
                            for t in T:
                                if reach(s, a, t):
                                    casc_idx.append((s, r, r2, a, t))
        casc = pulp.LpVariable.dicts("casc", casc_idx, lowBound=0, cat=cat_int)
        dem_idx = [("LC", r, t) for r in R for t in T] + [("BATCH", "*", t) for t in T]
        lease = pulp.LpVariable.dicts("lease", dem_idx, lowBound=0, upBound=None if sc.allow_lease else 0)
        short = pulp.LpVariable.dicts("short", dem_idx, lowBound=0)

        init: dict[tuple, float] = {}
        for f in sc.initial_fleet:
            key = (f["sku"], f["tier"], f["region"], int(f["age_q"]))
            init[key] = init.get(key, 0.0) + f["count"]

        cas_out = {k: [] for k in idx}
        cas_in = {k: [] for k in idx}
        for (s, r, r2, a, t) in casc_idx:
            cas_out[(s, "LC", r, a, t)].append(casc[(s, r, r2, a, t)])
            cas_in[(s, "BATCH", r2, a, t)].append(casc[(s, r, r2, a, t)])

        # Flow conservation + forced retirement + capacity
        for (s, tau, r, a, t) in idx:
            carried = (init.get((s, tau, r, a), 0.0) if t == 0
                       else (n.get((s, tau, r, a - 1, t - 1), 0) if a >= 1 else 0))
            b = buy[(s, tau, r, t)] if (a == 0 and (s, tau, r, t) in buy) else 0
            prob += (n[(s, tau, r, a, t)] == carried - ret[(s, tau, r, a, t)]
                     - pulp.lpSum(cas_out[(s, tau, r, a, t)]) + pulp.lpSum(cas_in[(s, tau, r, a, t)]) + b,
                     f"flow_{s}_{tau}_{r}_{a}_{t}")
            if a == sc.max_age_q[s]:
                prob += n[(s, tau, r, a, t)] == 0, f"maxage_{s}_{tau}_{r}_{t}"
            prob += w[(s, tau, r, a, t)] <= self.cap(s, tau, a) * n[(s, tau, r, a, t)], f"cap_{s}_{tau}_{r}_{a}_{t}"

        # Demand
        for t in T:
            for r in R:
                prob += (pulp.lpSum(w.get((s, "LC", r, a, t), 0) for s in S for a in self.A[s])
                         + lease[("LC", r, t)] + short[("LC", r, t)] >= sc.demand_at("LC", t, r), f"dem_LC_{r}_{t}")
            prob += (pulp.lpSum(w.get((s, "BATCH", r, a, t), 0) for s in S for r in R for a in self.A[s])
                     + lease[("BATCH", "*", t)] + short[("BATCH", "*", t)] >= sc.demand_at("BATCH", t), f"dem_B_{t}")

        # Reused-part pools
        for comp, pool in sc.pools.items():
            inv_prev: Any = pool.get("initial", 0.0)
            for t in T:
                harvest = pulp.lpSum(
                    pool.get("yield", 0.9) * sl.count * ret.get((s, tau, r, a, t), 0)
                    for s in S for sl in self.sku(s).slots if sl.component.name == comp and sl.initial_age == 0
                    for tau in TIERS for r in R for a in self.A[s])
                use = pulp.lpSum(
                    sl.count * buy[(s, tau, r, t)]
                    for s in sc.buyable for sl in self.sku(s).slots if sl.component.name == comp and sl.initial_age > 0
                    for tau in TIERS for r in buy_regions)
                inv = pulp.LpVariable(f"pool_{comp}_{t}", lowBound=0)
                prob += inv == inv_prev + pool.get("external_per_q", 0.0) + harvest - use, f"pool_{comp}_{t}"
                inv_prev = inv

        # Objective pieces
        emb = []
        op = []
        rep = []
        cost = []
        for (s, tau, r, a, t) in idx:
            rel = self.rel[s][a]
            k = self.sku(s)
            g = self.pue(r) * self.ci(r, t) / 1000.0
            emb.append(self.embodied_per_q(s, a) * n[(s, tau, r, a, t)] + self.stranded(s, a) * ret[(s, tau, r, a, t)])
            op.append(self.idle_kwh(s, a) * g * n[(s, tau, r, a, t)] + self.dyn_kwh_per_work(s, tau) * g * w[(s, tau, r, a, t)])
            rep.append((rel.spare_kg + rel.logistics_kg) * n[(s, tau, r, a, t)])
            price = sc.energy_price_usd_per_kwh * self.pue(r)
            cost.append(self.idle_kwh(s, a) * price * n[(s, tau, r, a, t)]
                        + self.dyn_kwh_per_work(s, tau) * price * w[(s, tau, r, a, t)]
                        + rel.repair_cost_usd * n[(s, tau, r, a, t)])
        for (s, tau, r, t) in buy_idx:
            cost.append(self.sku(s).capex_usd * buy[(s, tau, r, t)])
            if sc.embodied_accounting != "amortized":
                emb.append(self.E[s] * buy[(s, tau, r, t)])
        ship = []
        for (s, r, r2, a, t) in casc_idx:
            moved = r2 != r
            cost.append((sc.cascade_cost_usd + (sc.ship_cost_usd_per_node if moved else 0.0)) * casc[(s, r, r2, a, t)])
            if moved:
                ship.append(sc.ship_kg_per_node * casc[(s, r, r2, a, t)])
        lease_kg = []
        for (tau, r, t) in dem_idx:
            rr = r if r != "*" else min(R, key=lambda x: self.ci(x, t))  # batch leases go to cleanest region
            lease_kg.append(self.lease_kg_per_work(tau, rr, t) * lease[(tau, r, t)])
            cost.append(sc.lease_cost_per_work * lease[(tau, r, t)])

        carbon_expr = pulp.lpSum(emb) + pulp.lpSum(op) + pulp.lpSum(rep) + pulp.lpSum(ship) + pulp.lpSum(lease_kg)
        cost_expr = pulp.lpSum(cost)
        short_expr = pulp.lpSum(short.values())
        if objective == "carbon":
            prob += carbon_expr + sc.shortfall_carbon_penalty * short_expr
        elif objective == "cost":
            prob += cost_expr + sc.shortfall_penalty * short_expr
        elif objective == "shortfall":
            prob += short_expr
        else:
            raise ValueError(objective)
        for name, ub in (bounds or {}).items():
            expr = {"carbon": carbon_expr, "cost": cost_expr, "shortfall": short_expr}[name]
            prob += expr <= ub, f"eps_{name}"

        # Fix decisions (plan re-evaluation)
        groups = {"buy": buy, "ret": ret, "casc": casc}
        for gname, vals in (fix or {}).items():
            var = groups[gname]
            for k, v in vals.items():
                if k in var:
                    prob += var[k] == v, f"fix_{gname}_{'_'.join(map(str, k))}"

        self._vars = dict(n=n, w=w, ret=ret, buy=buy, casc=casc, lease=lease, short=short)
        self._exprs = dict(emb=pulp.lpSum(emb), op=pulp.lpSum(op), rep=pulp.lpSum(rep), ship=pulp.lpSum(ship),
                           lease=pulp.lpSum(lease_kg), carbon=carbon_expr, cost=cost_expr, short=short_expr)
        return prob

    def solve(self, objective: str = "carbon", bounds: dict[str, float] | None = None,
              fix: dict[str, dict] | None = None, time_limit: float = 300, gap: float = 1e-4) -> Solution:
        prob = self.build(objective, bounds, fix)
        solver = pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=gap)
        import time as _time
        t0 = _time.time()
        prob.solve(solver)
        elapsed = _time.time() - t0
        time_limited = self.sc.integer and elapsed >= 0.97 * time_limit
        status = pulp.LpStatus[prob.status]
        if status not in ("Optimal",):
            return Solution(status, math.nan, {}, pd.DataFrame(), pd.DataFrame(), {})
        v = self._vars
        val = lambda x: float(pulp.value(x) or 0.0)  # noqa: E731
        totals = dict(
            carbon_kg=val(self._exprs["carbon"]), embodied_kg=val(self._exprs["emb"]),
            operational_kg=val(self._exprs["op"]), repair_kg=val(self._exprs["rep"]),
            ship_kg=val(self._exprs["ship"]), lease_kg=val(self._exprs["lease"]),
            cost_usd=val(self._exprs["cost"]), shortfall=val(self._exprs["short"]),
            work=sum(val(x) for x in v["w"].values()),
        )
        fleet = pd.DataFrame([
            dict(sku=s, tier=tau, region=r, age_q=a, t=t, n=val(v["n"][k]), work=val(v["w"][k]))
            for k in v["n"] for (s, tau, r, a, t) in [k] if val(v["n"][k]) > 1e-6])
        rows = []
        for name in ("buy", "ret", "casc", "lease", "short"):
            for k, x in v[name].items():
                x = val(x)
                if x > 1e-6:
                    rows.append(dict(decision=name, key=k, t=k[-1], value=x))
        values = {g: {k: round(val(x)) if self.sc.integer else val(x) for k, x in v[g].items()}
                  for g in ("buy", "ret", "casc")}
        return Solution(status, float(pulp.value(prob.objective)), totals, fleet, pd.DataFrame(rows), values,
                        solve_seconds=elapsed, time_limited=time_limited)


def pareto(model: LifecycleModel, minimize: str = "carbon", against: str = "cost", n_points: int = 8,
           time_limit: float = 120) -> pd.DataFrame:
    """Epsilon-constraint front: minimize ``minimize`` subject to ``against`` <= eps, eps swept
    between the two single-objective optima."""
    lo = model.solve(objective=against, time_limit=time_limit)
    hi = model.solve(objective=minimize, time_limit=time_limit)
    key = {"carbon": "carbon_kg", "cost": "cost_usd", "shortfall": "shortfall"}
    a_min, a_max = lo.totals[key[against]], hi.totals[key[against]]
    rows = []
    for i in range(n_points):
        eps = a_min + (a_max - a_min) * i / max(1, n_points - 1)
        sol = model.solve(objective=minimize, bounds={against: eps * (1 + 1e-6) + 1e-6}, time_limit=time_limit)
        if sol.status == "Optimal":
            rows.append(dict(eps=eps, **sol.totals))
    return pd.DataFrame(rows)
