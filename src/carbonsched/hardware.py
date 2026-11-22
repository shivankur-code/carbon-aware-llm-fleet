"""Server SKUs and component catalog, loaded from ``configs/components.yaml`` and ``configs/skus.yaml``.

A SKU is a set of component *slots* (component class x count x initial part
age) plus platform-level embodied carbon, a power curve, and throughput per
service tier. ``initial_age > 0`` marks reused / cascaded parts (GreenSKU-style
reused DRAM and SSDs come from a first life of several years).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
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


@dataclass(frozen=True)
class RefurbSpec:
    """Refurbishment of a ``parent`` host at node age ``ages_q[0] == ages_q[1]`` (quarters): its ``replace``
    component classes are swapped for new parts and it continues as SKU ``name`` at age 0 of a second life
    lasting at most ``max_age_q`` quarters. A config path with an age window is expanded into one spec (and
    one derived SKU, named ``family@age``) per age, so carried parts' ages and their remaining amortization
    are exact - no midpoint approximation the optimizer could exploit."""
    name: str
    parent: str
    replace: tuple[str, ...]
    ages_q: tuple[int, int]
    max_age_q: int
    capex_usd: float
    downtime_hours: float
    family: str = ""            # config-level path name; one spec (and derived SKU) per refurbishment age

    @property
    def a_rep_q(self) -> float:
        return 0.5 * (self.ages_q[0] + self.ages_q[1])


@dataclass
class Catalog:
    components: dict[str, ComponentReliability]
    skus: dict[str, SKU]
    registry: dict[str, Param]
    refurb: dict[str, RefurbSpec] = field(default_factory=dict)


def derive_refurbished(parent: SKU, spec: RefurbSpec, components: dict[str, ComponentReliability]) -> SKU:
    """Refurbished SKU: replaced slots become new parts (base component model, age 0); carried slots keep
    the parent's component model (incl. any reused-part hazard multiplier) and age by ``a_rep_q``."""
    dt = spec.a_rep_q / 4.0
    slots = tuple(
        Slot(component=components[sl.component.name], count=sl.count, initial_age=0.0)
        if sl.component.name in spec.replace else
        Slot(component=sl.component, count=sl.count, initial_age=sl.initial_age + dt)
        for sl in parent.slots)
    return replace(parent, name=spec.name, slots=slots, capex_usd=spec.capex_usd,
                   platform_initial_age=parent.platform_initial_age + dt,
                   description=f"{parent.name} refurbished at ~{spec.a_rep_q:g} q: new {', '.join(spec.replace)}")


def load_measured(config_dir: Path) -> dict[str, dict]:
    """``configs/measured.yaml`` (written by ``gpu/ingest.py``): {param_path: {value, source}}."""
    import yaml
    f = Path(config_dir) / "measured.yaml"
    if not f.exists():
        return {}
    return (yaml.safe_load(f.read_text()) or {}).get("measured", {})


def load_catalog(config_dir: str | Path = "configs", overrides: dict[str, float] | None = None,
                 use_measured: bool = True) -> Catalog:
    """Load components + SKUs. Measured values (``configs/measured.yaml``) replace assumptions
    unless ``use_measured=False``; explicit ``overrides`` (sweeps) win over both."""
    config_dir = Path(config_dir)
    registry: dict[str, Param] = {}
    measured = load_measured(config_dir) if use_measured else {}
    overrides = {**{k: v["value"] for k, v in measured.items()}, **(overrides or {})}
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

    reused_mult = float(comps_raw.get("reused_part_afr_multiplier", 1.0))
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
            Slot(component=_reused_variant(components[sl["component"]], reused_mult) if sl.get("initial_age", 0.0) > 0
                 else components[sl["component"]], count=int(sl["count"]), initial_age=sl.get("initial_age", 0.0))
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
    for path, m in measured.items():
        if path in registry:
            old = registry[path]
            registry[path] = Param(path=path, value=float(m["value"]), low=None, high=None,
                                   source=f"MEASURED: {m.get('source', '')} (was {old.value:g}: {old.source})",
                                   status="measured", unit=old.unit)
    refurb: dict[str, RefurbSpec] = {}
    for family, r in (skus_raw.get("refurbish") or {}).items():
        if r["from"] not in skus:
            raise ValueError(f"refurbish {family}: unknown parent {r['from']}")
        for age in range(int(r["ages_q"][0]), int(r["ages_q"][1]) + 1):
            name = f"{family}@{age}"
            spec = RefurbSpec(name=name, parent=r["from"], replace=tuple(r["replace"]), ages_q=(age, age),
                              max_age_q=int(r["max_age_q"]), capex_usd=float(r["capex_usd"]),
                              downtime_hours=float(r.get("downtime_hours", 0.0)), family=family)
            refurb[name] = spec
            skus[name] = derive_refurbished(skus[spec.parent], spec, components)
    return Catalog(components=components, skus=skus, registry=registry, refurb=refurb)


def _reused_variant(c: ComponentReliability, m: float) -> ComponentReliability:
    """Reused parts may fail more than their age alone predicts (screening escapes, handling,
    different first-life stress). Proportional-hazards multiplier ``m`` on the part's model.
    Conservative: replacement spares inherit the multiplier too."""
    if m == 1.0:
        return c
    return replace(c, model=c.model.scaled(m))
