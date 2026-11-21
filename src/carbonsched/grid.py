"""Regional grid carbon intensity (gCO2e/kWh) per quarter.

Trajectory: ``ci(t) = max(floor, ci_base * (1 - annual_decline) ** years_since_base)``.
Decarbonization matters for lifecycle decisions: a cleaner future grid shrinks
the operational benefit of buying newer, more efficient hardware and so
favours keeping / reusing older hardware longer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .params import Param, load_yaml


@dataclass(frozen=True)
class Region:
    name: str
    ci_base: float          # gCO2e/kWh in base_year
    annual_decline: float   # fractional decline per year
    floor: float = 0.0
    pue: float = 1.2

    def ci(self, quarter: int | np.ndarray) -> np.ndarray | float:
        years = np.asarray(quarter, dtype=float) / 4.0
        return np.maximum(self.floor, self.ci_base * (1.0 - self.annual_decline) ** years)


def load_regions(config_dir: str | Path = "configs", registry: dict[str, Param] | None = None,
                 overrides: dict[str, float] | None = None) -> dict[str, Region]:
    raw = load_yaml(Path(config_dir) / "regions.yaml", registry, overrides)
    return {
        name: Region(name=name, ci_base=r["ci_base"], annual_decline=r["annual_decline"],
                     floor=r.get("floor", 0.0), pue=r.get("pue", raw.get("default_pue", 1.2)))
        for name, r in raw["regions"].items()
    }
