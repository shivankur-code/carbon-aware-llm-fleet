import pytest

from conftest import CONFIGS
from carbonsched.carbon import (
    CarbonBreakdown, node_quarter_carbon, operational_kg, part_embodied_kg, purchase_embodied_kg,
)
from carbonsched.grid import Region, load_regions
from carbonsched.hardware import load_catalog
from carbonsched.params import parameters_table
from carbonsched.reliability import QuarterReliability, node_reliability_table


def test_operational_hand_check():
    # 10 kW for 100 h at PUE 1.2 on a 250 g/kWh grid = 1200 kWh * 0.25 kg = 300 kg
    assert operational_kg(10_000, 100, 1.2, 250) == pytest.approx(300.0)


def test_reuse_policies():
    assert part_embodied_kg(100, 0, 6, "cutoff") == 100   # new parts always full
    assert part_embodied_kg(100, 3, 6, "cutoff") == 0
    assert part_embodied_kg(100, 3, 6, "residual") == pytest.approx(50)
    assert part_embodied_kg(100, 9, 6, "residual") == 0
    assert part_embodied_kg(100, 3, 6, "full") == 100
    with pytest.raises(ValueError):
        part_embodied_kg(100, 3, 6, "bogus")


def test_catalog_loads_and_reuse_lowers_embodied():
    cat = load_catalog(CONFIGS)
    new, green, old = (cat.skus[k] for k in ("new_gpu", "green_gpu", "old_gpu"))
    assert green.is_reuse and old.is_reuse and not new.is_reuse
    e_new = purchase_embodied_kg(new, "cutoff")
    e_green = purchase_embodied_kg(green, "cutoff")
    assert e_green < e_new
    # 'full' policy charges reused parts as new -> green_gpu only differs by CPU/platform
    assert purchase_embodied_kg(green, "full") >= e_green
    assert purchase_embodied_kg(old, "cutoff") == 0.0
    # every parameter carries a valid status
    assert all(p.status in ("verified", "derived", "assumption") for p in cat.registry.values())


def test_overrides_flow_through():
    cat = load_catalog(CONFIGS, overrides={"skus.skus.new_gpu.power_peak_w": 9000})
    assert cat.skus["new_gpu"].power_peak_w == 9000


def test_reused_parts_fail_more_than_new_in_same_node_type():
    cat = load_catalog(CONFIGS)
    new = node_reliability_table(cat.skus["new_gpu"].slots, 4)
    green = node_reliability_table(cat.skus["green_gpu"].slots, 4)
    assert green[0].failures["dram_dimm"] > new[0].failures["dram_dimm"]
    assert green[0].failures["ssd"] > new[0].failures["ssd"]


def test_drained_hours_draw_no_power():
    cat = load_catalog(CONFIGS)
    sku = cat.skus["new_gpu"]
    r = Region("r", ci_base=100, annual_decline=0.0, pue=1.0)
    up = node_quarter_carbon(sku, QuarterReliability(), r, 0, 1.0)
    down = node_quarter_carbon(sku, QuarterReliability(drain_hours=2191.5), r, 0, 1.0)
    assert down.operational_kg == pytest.approx(0.0, abs=1e-9)
    assert up.operational_kg > 0


def test_breakdown_algebra():
    a = CarbonBreakdown(1, 2, 3, 4, {"lease": 5})
    b = (a + a).scaled(0.5)
    assert b.total_kg == pytest.approx(a.total_kg) == 15


def test_regions_decline_and_floor():
    regs = load_regions(CONFIGS)
    assert set(regs) == {"clean", "mixed", "dirty"}
    m = regs["mixed"]
    assert m.ci(4) == pytest.approx(m.ci_base * (1 - m.annual_decline))
    assert m.ci(10_000) == pytest.approx(m.floor)


def test_parameters_table_renders():
    cat = load_catalog(CONFIGS)
    md = parameters_table(cat.registry)
    assert "Llama 3" in md and md.count("\n") > 20


def test_reused_part_multiplier_only_hits_reused_parts():
    base = load_catalog(CONFIGS)
    worse = load_catalog(CONFIGS, overrides={"components.reused_part_afr_multiplier": 3.0})
    b = node_reliability_table(base.skus["green_gpu"].slots, 2)[0].failures
    w = node_reliability_table(worse.skus["green_gpu"].slots, 2)[0].failures
    assert w["gpu_hopper"] == pytest.approx(b["gpu_hopper"])       # new GPUs unchanged
    assert w["dram_dimm"] == pytest.approx(3 * b["dram_dimm"], rel=0.05)  # reused DIMMs ~3x
