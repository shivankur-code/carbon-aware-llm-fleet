"""Lifecycle carbon accounting (kgCO2e).

Lifecycle carbon of a fleet plan = embodied (new nodes, refurbishment parts,
spares) + operational (power x PUE x grid CI) + repair logistics.

**Reused-component embodied carbon** is a methodological choice that can flip
conclusions, so it is explicit:

* ``cutoff`` – reused parts carry zero embodied carbon (GHG-Protocol-style
  cut-off; the first life already paid for them). Most favourable to reuse.
* ``residual`` – reused parts carry the *unamortized* share of their embodied
  carbon: ``E * max(0, 1 - age / design_life)``. Linear amortization.
* ``full`` – reused parts are charged as if new. A pessimistic bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .grid import Region
from .hardware import SKU
from .reliability import HOURS_PER_YEAR, QUARTER, QuarterReliability

REUSE_POLICIES = ("cutoff", "residual", "full")


@dataclass
class CarbonBreakdown:
    embodied_kg: float = 0.0
    operational_kg: float = 0.0
    spares_kg: float = 0.0
    logistics_kg: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def repair_kg(self) -> float:
        return self.spares_kg + self.logistics_kg

    @property
    def total_kg(self) -> float:
        return self.embodied_kg + self.operational_kg + self.repair_kg + sum(self.extra.values())

    def __add__(self, other: "CarbonBreakdown") -> "CarbonBreakdown":
        extra = dict(self.extra)
        for k, v in other.extra.items():
            extra[k] = extra.get(k, 0.0) + v
        return CarbonBreakdown(self.embodied_kg + other.embodied_kg, self.operational_kg + other.operational_kg,
                               self.spares_kg + other.spares_kg, self.logistics_kg + other.logistics_kg, extra)

    def scaled(self, k: float) -> "CarbonBreakdown":
        return CarbonBreakdown(self.embodied_kg * k, self.operational_kg * k, self.spares_kg * k,
                               self.logistics_kg * k, {n: v * k for n, v in self.extra.items()})


def operational_kg(power_w: float, hours: float, pue: float, ci_g_per_kwh: float) -> float:
    """Facility-level operational carbon: W -> kWh, times PUE, times grid CI."""
    return power_w / 1000.0 * hours * pue * ci_g_per_kwh / 1000.0


def part_embodied_kg(new_kg: float, age_years: float, design_life_years: float, policy: str) -> float:
    if policy not in REUSE_POLICIES:
        raise ValueError(f"unknown reuse policy {policy!r}")
    if age_years <= 0 or policy == "full":
        return new_kg
    if policy == "cutoff":
        return 0.0
    return new_kg * max(0.0, 1.0 - age_years / design_life_years)


def purchase_embodied_kg(sku: SKU, policy: str = "cutoff") -> float:
    """Embodied carbon charged when one node of ``sku`` enters the fleet."""
    total = part_embodied_kg(sku.platform_embodied_kg, sku.platform_initial_age, sku.design_life_years, policy)
    for s in sku.slots:
        total += s.count * part_embodied_kg(sku.embodied_new_kg[s.component.name], s.initial_age,
                                            sku.design_life_years, policy)
    return total


def node_quarter_carbon(sku: SKU, rel: QuarterReliability, region: Region, quarter: int, util: float) -> CarbonBreakdown:
    """Operational + repair carbon for one node over one quarter.

    Power is drawn only while the node is in service; drained hours draw ~0
    (node powered down for repair). Spare-part embodied carbon and logistics
    are charged in the quarter the failure happens.
    """
    hours = HOURS_PER_YEAR * QUARTER * (1.0 - rel.availability_loss)
    op = operational_kg(sku.power_w(util), hours, region.pue, float(region.ci(quarter)))
    return CarbonBreakdown(operational_kg=op, spares_kg=rel.spare_kg, logistics_kg=rel.logistics_kg)


def carbon_per_work_g(sku: SKU, tier: str, region: Region, quarter: int, util: float,
                      amortized_embodied_kg_per_hour: float = 0.0) -> float:
    """gCO2e per unit of work in a tier at a given moment (used for the crossover analyses)."""
    work = sku.effective_throughput(tier) * util
    if work <= 0:
        return float("inf")
    op_g_per_h = sku.power_w(util) / 1000.0 * region.pue * float(region.ci(quarter))
    return (op_g_per_h + amortized_embodied_kg_per_hour * 1000.0) / work
