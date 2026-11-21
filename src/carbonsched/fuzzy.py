"""Fuzzy (triangular) uncertainty on grouped parameters, propagated by alpha-cuts.

Each group scales a set of parameters by a triangular fuzzy factor
(low, mode, high). At membership level alpha the factor lies in
``[low + alpha*(mode-low), high - alpha*(high-mode)]``; alpha = 1 is the crisp
model, alpha = 0 the full support. The output interval at each alpha is taken
over all 2^G vertices of the box (vertex method; exact when the output is
monotone in each factor, which we check by also evaluating the box centre).

This follows the fuzzy-MILP approach of Poonia, Kulshrestha, Sangwan & Sharma
(2024) and answers RQ5: how fragile is "reuse is greener"?
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .params import Param


@dataclass(frozen=True)
class FuzzyGroup:
    name: str
    match: Callable[[str], bool]      # which parameter paths this group scales
    low: float
    mode: float
    high: float
    clip: tuple[float, float] | None = None

    def cut(self, alpha: float) -> tuple[float, float]:
        return (self.low + alpha * (self.mode - self.low), self.high - alpha * (self.high - self.mode))


def overrides_for(registry: dict[str, Param], groups: list[FuzzyGroup], factors: dict[str, float]) -> dict[str, float]:
    ov: dict[str, float] = {}
    for g in groups:
        f = factors[g.name]
        for path, p in registry.items():
            if g.match(path):
                v = p.value * f
                if g.clip:
                    v = min(max(v, g.clip[0]), g.clip[1])
                ov[path] = v
    return ov


def alpha_cuts(evaluate: Callable[[dict[str, float]], float], groups: list[FuzzyGroup],
               alphas: tuple[float, ...] = (0.0, 0.5, 1.0)) -> pd.DataFrame:
    """``evaluate(factors) -> scalar``; returns min/max of the scalar per alpha."""
    rows = []
    cache: dict[tuple, float] = {}
    for a in alphas:
        cuts = {g.name: g.cut(a) for g in groups}
        vals = []
        for combo in itertools.product(*[sorted(set(cuts[g.name])) for g in groups]):
            key = tuple(round(x, 9) for x in combo)
            if key not in cache:
                cache[key] = evaluate(dict(zip([g.name for g in groups], combo)))
            vals.append(cache[key])
        centre = tuple(round(sum(cuts[g.name]) / 2, 9) for g in groups)
        if centre not in cache:
            cache[centre] = evaluate(dict(zip([g.name for g in groups], centre)))
        rows.append(dict(alpha=a, low=min(vals), high=max(vals), centre=cache[centre],
                         monotone_ok=min(vals) - 1e-9 <= cache[centre] <= max(vals) + 1e-9))
    return pd.DataFrame(rows)
