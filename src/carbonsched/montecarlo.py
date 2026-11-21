"""Stochastic replay of a plan: does the spare capacity actually hold up?

The lifecycle model plans with *expected* drained capacity (times a buffer for
the LC tier). Here we check a plan against the *distribution* of
simultaneously-drained nodes.

At a random instant, the number of nodes of a cohort that are down for repair
is the occupancy of an M/G/inf queue: Poisson with mean
``N * drain_hours_per_quarter / hours_per_quarter`` (failures arrive at rate
lambda, each takes d hours). Real fleets see **correlated** failures (shared
racks, power, firmware, bad batches), so we optionally use a negative binomial
with the same mean and dispersion ``k`` (variance = m + m^2 / k; k -> inf
recovers Poisson).

For each LC region and quarter we sample down-counts per cohort, compute
instantaneous capacity (served + leased) and report the probability that it
falls below demand (an SLO violation instant) and the expected unserved fraction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .milp import HOURS_Q, LifecycleModel, Solution


def _down(rng, mean: np.ndarray, n_samples: int, dispersion: float | None) -> np.ndarray:
    mean = np.maximum(mean, 0.0)
    if dispersion is None or np.isinf(dispersion):
        return rng.poisson(mean, size=(n_samples, mean.size))
    # Gamma-Poisson mixture = negative binomial with mean m, variance m + m^2/k
    lam = rng.gamma(shape=dispersion, scale=np.where(mean > 0, mean / dispersion, 0.0), size=(n_samples, mean.size))
    return rng.poisson(lam)


def replay_lc(model: LifecycleModel, sol: Solution, n_samples: int = 20000, dispersion: float | None = None,
              seed: int = 0) -> pd.DataFrame:
    """Per (region, quarter): P(LC capacity < demand) and expected unserved fraction."""
    rng = np.random.default_rng(seed)
    sc = model.sc
    lease = {}
    if not sol.decisions.empty:
        d = sol.decisions[sol.decisions.decision == "lease"]
        for k, v in zip(d.key, d.value):
            lease[(k[0], k[1], k[2])] = v
    fl = sol.fleet[sol.fleet.tier == "LC"]
    rows = []
    for (r, t), g in fl.groupby(["region", "t"]):
        N = np.round(g.n.to_numpy())
        cap_nom = np.array([model.sku(s).effective_throughput("LC") * sc.util_cap for s in g.sku])
        mean_down = np.array([N_i * model.rel[s][a].drain_hours / HOURS_Q for N_i, s, a in zip(N, g.sku, g.age_q)])
        down = np.minimum(_down(rng, mean_down, n_samples, dispersion), N)
        cap = ((N - down) * cap_nom).sum(axis=1) + lease.get(("LC", r, t), 0.0)
        D = sc.demand_at("LC", int(t), r)
        unserved = np.maximum(0.0, D - cap) / max(D, 1e-9)
        rows.append(dict(region=r, t=int(t), demand=D, nodes=N.sum(), mean_down=mean_down.sum(),
                         p_violation=float((cap < D - 1e-9).mean()), exp_unserved=float(unserved.mean()),
                         p99_unserved=float(np.quantile(unserved, 0.99))))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    return dict(max_p_violation=float(df.p_violation.max()), mean_p_violation=float(df.p_violation.mean()),
                quarters_over_1pct=int((df.p_violation > 0.01).sum()), mean_exp_unserved=float(df.exp_unserved.mean()))
