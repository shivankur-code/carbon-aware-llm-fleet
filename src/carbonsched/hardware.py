"""Server SKUs and component catalog, loaded from ``configs/components.yaml`` and ``configs/skus.yaml``.

A SKU is a set of component *slots* (component class x count x initial part
age) plus platform-level embodied carbon, a power curve, and throughput per
service tier. ``initial_age > 0`` marks reused / cascaded parts (GreenSKU-style
reused DRAM and SSDs come from a first life of several years).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .params import Param, load_yaml
from .reliability import ComponentReliability, Slot, build_model

TIERS = ("LC", "BATCH")


@dataclass(frozen=True)
class SKU:
    name: str
    slots: tuple[Slot, ...]
    embodied_new_kg: dict[str, float]   # embodied carbon of each component class if bought new, per unit
    platform_embodied_kg: float         # chassis, board, PSUs, NICs, ...
    power_idle_w: float
    power_peak_w: float
    throughput: dict[str, float]        # work units per node-hour at full use, per tier
    capex_usd: float
    design_life_years: float            # first-life design lifetime, used for residual-embodied accounting
    description: str = ""
    cxl_slowdown: dict[str, float] = field(default_factory=dict)  # throughput multiplier per tier (<=1)
    platform_initial_age: float = 0.0   # > 0 when the whole chassis is reused / cascaded

    def power_w(self, util: float) -> float:
        """Linear power model between idle and peak (SPECpower-style)."""
        u = min(max(util, 0.0), 1.0)
        return self.power_idle_w + (self.power_peak_w - self.power_idle_w) * u

    def effective_throughput(self, tier: str) -> float:
        return self.throughput[tier] * self.cxl_slowdown.get(tier, 1.0)

    @property
    def is_reuse(self) -> bool:
        return any(s.initial_age > 0 for s in self.slots)


@dataclass
class Catalog:
    components: dict[str, ComponentReliability]
    skus: dict[str, SKU]
    registry: dict[str, Param]


def load_catalog(config_dir: str | Path = "configs", overrides: dict[str, float] | None = None) -> Catalog:
    config_dir = Path(config_dir)
    registry: dict[str, Param] = {}
    comps_raw = load_yaml(config_dir / "components.yaml", registry, overrides)
    skus_raw = load_yaml(config_dir / "skus.yaml", registry, overrides)

    components: dict[str, ComponentReliability] = {}
    embodied: dict[str, float] = {}
    for name, c in comps_raw["components"].items():
        components[name] = ComponentReliability(
            name=name,
            model=build_model(c["failure"]),
            drain_hours=c["drain_hours"],
            spare_embodied_kg=c["embodied_kg"],
            repair_cost_usd=c.get("repair_cost_usd", 0.0),
            repair_logistics_kg=c.get("repair_logistics_kg", 0.0),
            repair=c.get("repair", "renewal"),
        )
        embodied[name] = c["embodied_kg"]

    skus: dict[str, SKU] = {}
    raw_skus = skus_raw["skus"]
    for name, s in raw_skus.items():
        # Power may be given relative to another SKU with the same GPUs, so that
        # sensitivity sweeps move both together and only the *difference* is uncertain.
        if "power_relative_to" in s:
            ref = raw_skus[s["power_relative_to"]]
            s = dict(s, power_idle_w=ref["power_idle_w"] + s.get("power_idle_delta_w", 0.0),
                     power_peak_w=ref["power_peak_w"] + s.get("power_peak_delta_w", 0.0))
        slots = tuple(
            Slot(component=components[sl["component"]], count=int(sl["count"]), initial_age=sl.get("initial_age", 0.0))
            for sl in s["slots"]
        )
        missing = set(TIERS) - set(s["throughput"])
        if missing:
            raise ValueError(f"SKU {name} missing throughput for tiers {missing}")
        skus[name] = SKU(
            name=name,
            slots=slots,
            embodied_new_kg={sl.component.name: embodied[sl.component.name] for sl in slots},
            platform_embodied_kg=s["platform_embodied_kg"],
            power_idle_w=s["power_idle_w"],
            power_peak_w=s["power_peak_w"],
            throughput=dict(s["throughput"]),
            capex_usd=s["capex_usd"],
            design_life_years=s["design_life_years"],
            description=s.get("description", ""),
            cxl_slowdown=dict(s.get("cxl_slowdown", {})),
            platform_initial_age=s.get("platform_initial_age", 0.0),
        )
    return Catalog(components=components, skus=skus, registry=registry)
