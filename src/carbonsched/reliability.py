"""Age-dependent failure models and per-quarter expected-failure tables.

Time is in **years** inside the hazard models; the fleet model works in quarters.

Two repair semantics are supported for a component *slot* in a node:

* ``minimal`` – repair restores the part to "as bad as old" (non-homogeneous
  Poisson process). Expected failures in ``[a, b]`` are ``H(b) - H(a)``.
* ``renewal`` – the failed part is swapped for a **new** spare (the usual
  datacenter practice). The slot is a *delayed* renewal process: the original
  part starts at age ``a0`` (> 0 for reused parts), every replacement starts
  at age 0. Solved numerically on a fine grid.

Because node age is a cohort index in the lifecycle MILP, the outputs here are
precomputed parameters and the optimization stays linear.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import numpy as np

HOURS_PER_YEAR = 8766.0
QUARTER = 0.25  # years


class HazardModel(Protocol):
    def cum_hazard(self, a: np.ndarray | float) -> np.ndarray | float: ...


@dataclass(frozen=True)
class Weibull:
    """Weibull hazard h(a) = (k/eta) (a/eta)^(k-1); cumulative H(a) = (a/eta)^k."""

    shape: float  # k (beta). k<1 infant mortality, k=1 constant, k>1 wear-out
    scale: float  # eta, years

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError("Weibull shape and scale must be positive")

    def cum_hazard(self, a):
        a = np.maximum(np.asarray(a, dtype=float), 0.0)
        return (a / self.scale) ** self.shape

    def hazard(self, a):
        a = np.maximum(np.asarray(a, dtype=float), 1e-12)
        return (self.shape / self.scale) * (a / self.scale) ** (self.shape - 1)

    def survival(self, a):
        return np.exp(-self.cum_hazard(a))

    @classmethod
    def from_afr(cls, afr: float, shape: float, at_age: float = 0.0) -> "Weibull":
        """Calibrate scale so the annual failure probability over [at_age, at_age+1] equals ``afr``.

        This is how field studies (reported as annualized failure / replacement
        rates) are turned into a Weibull with an assumed shape.
        """
        if not 0 < afr < 1:
            raise ValueError("afr must be in (0, 1)")
        k = -math.log1p(-afr)  # H(a+1) - H(a)
        span = (at_age + 1.0) ** shape - at_age ** shape
        return cls(shape=shape, scale=(span / k) ** (1.0 / shape))

    def afr(self, at_age: float = 0.0) -> float:
        return float(1.0 - math.exp(-(self.cum_hazard(at_age + 1.0) - self.cum_hazard(at_age))))


@dataclass(frozen=True)
class Bathtub:
    """Competing-risk bathtub: infant-mortality Weibull + random (constant) + wear-out Weibull."""

    infant: Weibull | None = None
    random_rate: float = 0.0  # per year
    wearout: Weibull | None = None

    def cum_hazard(self, a):
        a = np.maximum(np.asarray(a, dtype=float), 0.0)
        h = self.random_rate * a
        for part in (self.infant, self.wearout):
            if part is not None:
                h = h + part.cum_hazard(a)
        return h

    def survival(self, a):
        return np.exp(-self.cum_hazard(a))


def expected_failures_minimal(model: HazardModel, a0: float, t1: float, t2: float) -> float:
    """Expected failures of one slot during slot-time [t1, t2] under minimal repair."""
    return float(model.cum_hazard(a0 + t2) - model.cum_hazard(a0 + t1))


def renewal_curve(model: HazardModel, a0: float, horizon: float, step: float = 1 / 104) -> tuple[np.ndarray, np.ndarray]:
    """Expected cumulative failures M(t) of a delayed renewal process on a grid.

    The first part has age ``a0`` at t=0 (first-failure CDF is the conditional
    ``1 - S(a0+t)/S(a0)``); replacements are new. Discretized renewal equation:
    ``M_D(t) = F_D(t) + sum_j m(t - u_j) dF_D(u_j)`` with ``m`` the ordinary
    renewal function solved by the standard recursion.
    """
    n = int(math.ceil(horizon / step))
    t = np.arange(n + 1) * step
    S = lambda x: np.exp(-np.asarray(model.cum_hazard(x)))  # noqa: E731
    F = 1.0 - S(t)                                  # new-part lifetime CDF
    FD = 1.0 - S(a0 + t) / S(a0)                    # delayed first-failure CDF
    f = np.diff(F)                                  # mass in each cell
    fD = np.diff(FD)
    m = np.zeros(n + 1)                             # ordinary renewal function
    for i in range(1, n + 1):
        # m(t_i) = F(t_i) + sum_{j=1..i} m(t_i - t_j) f_j   (mass f_j placed at cell end)
        m[i] = F[i] + np.dot(m[i - 1::-1][:i], f[:i])
    MD = np.zeros(n + 1)
    for i in range(1, n + 1):
        MD[i] = FD[i] + np.dot(m[i - 1::-1][:i], fD[:i])
    return t, MD


def expected_failures_renewal(model: HazardModel, a0: float, t1: float, t2: float, step: float = 1 / 104) -> float:
    t, M = renewal_curve(model, a0, t2 + step, step)
    return float(np.interp(t2, t, M) - np.interp(t1, t, M))


@dataclass(frozen=True)
class ComponentReliability:
    """Failure behaviour and repair consequences of one component class."""

    name: str
    model: HazardModel
    drain_hours: float          # node capacity lost per failure (drain + repair + return)
    spare_embodied_kg: float    # embodied carbon of the replacement part
    repair_cost_usd: float = 0.0
    repair_logistics_kg: float = 0.0  # truck roll / shipping / RMA per event
    repair: str = "renewal"     # "renewal" (swap for new) or "minimal"


@dataclass(frozen=True)
class Slot:
    component: ComponentReliability
    count: int
    initial_age: float = 0.0    # years; > 0 for reused / cascaded parts


@dataclass
class QuarterReliability:
    """Expected per-node reliability consequences for one quarter of node life."""

    failures: dict[str, float] = field(default_factory=dict)
    drain_hours: float = 0.0
    spare_kg: float = 0.0
    logistics_kg: float = 0.0
    repair_cost_usd: float = 0.0

    @property
    def total_failures(self) -> float:
        return sum(self.failures.values())

    @property
    def availability_loss(self) -> float:
        """Fraction of the quarter the node is drained (capped at 1)."""
        return min(1.0, self.drain_hours / (HOURS_PER_YEAR * QUARTER))


def node_reliability_table(slots: tuple[Slot, ...] | list[Slot], n_quarters: int,
                           start_age: float = 0.0) -> list[QuarterReliability]:
    """Per-quarter expected failures and costs for a node, for node ages
    ``start_age + q*0.25`` years, ``q = 0..n_quarters-1``.

    ``start_age`` is the node's age (in its current life) at the first quarter;
    each slot's part age is ``slot.initial_age + node age``.
    """
    out = [QuarterReliability() for _ in range(n_quarters)]
    edges = start_age + QUARTER * np.arange(n_quarters + 1)
    for s in slots:
        c = s.component
        if c.repair == "minimal":
            H = np.asarray(c.model.cum_hazard(s.initial_age + edges))
            per_q = np.diff(H)
        elif c.repair == "renewal":
            t, M = _renewal_cached(c.model, s.initial_age, float(edges[-1]))
            per_q = np.diff(np.interp(edges, t, M))
        else:
            raise ValueError(f"unknown repair policy {c.repair!r}")
        for q in range(n_quarters):
            n_fail = s.count * float(per_q[q])
            r = out[q]
            r.failures[c.name] = r.failures.get(c.name, 0.0) + n_fail
            r.drain_hours += n_fail * c.drain_hours
            r.spare_kg += n_fail * c.spare_embodied_kg
            r.logistics_kg += n_fail * c.repair_logistics_kg
            r.repair_cost_usd += n_fail * c.repair_cost_usd
    return out


@lru_cache(maxsize=512)
def _renewal_cached(model: HazardModel, a0: float, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    return renewal_curve(model, a0, horizon + QUARTER)


def build_model(spec: dict) -> HazardModel:
    """Build a hazard model from a resolved config mapping.

    Accepted forms::

        {type: weibull, shape: 1.5, scale: 12.0}
        {type: weibull, shape: 1.5, afr: 0.02, afr_at_age: 1.0}
        {type: bathtub, infant: {...}, random_rate: 0.01, wearout: {...}}
    """
    kind = spec.get("type", "weibull")
    if kind == "weibull":
        if "scale" in spec:
            return Weibull(shape=spec["shape"], scale=spec["scale"])
        return Weibull.from_afr(spec["afr"], spec["shape"], spec.get("afr_at_age", 0.0))
    if kind == "bathtub":
        return Bathtub(
            infant=build_model(spec["infant"]) if spec.get("infant") else None,
            random_rate=spec.get("random_rate", 0.0),
            wearout=build_model(spec["wearout"]) if spec.get("wearout") else None,
        )
    raise ValueError(f"unknown hazard model type {kind!r}")
