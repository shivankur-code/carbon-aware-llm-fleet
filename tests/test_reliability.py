import math

import numpy as np
import pytest

from carbonsched.reliability import (
    Bathtub, ComponentReliability, Slot, Weibull, expected_failures_minimal,
    expected_failures_renewal, node_reliability_table,
)


def test_weibull_cum_hazard_matches_integral_of_hazard():
    w = Weibull(shape=1.7, scale=6.0)
    a = np.linspace(1e-6, 5, 200001)
    numeric = np.trapezoid(w.hazard(a), a)
    assert numeric == pytest.approx(float(w.cum_hazard(5.0)), rel=1e-4)


@pytest.mark.parametrize("shape,age", [(1.0, 0.0), (1.6, 0.0), (1.6, 4.0), (0.7, 2.0)])
def test_from_afr_roundtrip(shape, age):
    w = Weibull.from_afr(0.03, shape, at_age=age)
    assert w.afr(age) == pytest.approx(0.03, rel=1e-12)


def test_wearout_reused_part_fails_more():
    w = Weibull.from_afr(0.01, 1.6)
    assert w.afr(5.0) > 2 * w.afr(0.0)


def test_exponential_renewal_equals_minimal_and_is_memoryless():
    w = Weibull(shape=1.0, scale=10.0)  # rate 0.1/yr
    for a0 in (0.0, 5.0):
        r = expected_failures_renewal(w, a0, 0.0, 3.0, step=1 / 365)
        m = expected_failures_minimal(w, a0, 0.0, 3.0)
        assert m == pytest.approx(0.3, rel=1e-12)
        assert r == pytest.approx(0.3, rel=5e-3)


def test_renewal_below_minimal_for_wearout():
    # Replacing a worn part with a new one must give fewer failures than "as bad as old".
    w = Weibull(shape=2.5, scale=4.0)
    r = expected_failures_renewal(w, 5.0, 0.0, 4.0)
    m = expected_failures_minimal(w, 5.0, 0.0, 4.0)
    assert r < m


def test_renewal_long_run_rate_is_one_over_mean_life():
    w = Weibull(shape=2.0, scale=2.0)
    mean = w.scale * math.gamma(1 + 1 / w.shape)
    r = expected_failures_renewal(w, 0.0, 20.0, 30.0, step=1 / 104)
    assert r / 10.0 == pytest.approx(1 / mean, rel=1e-2)


def test_bathtub_is_sum_of_parts():
    inf, wo = Weibull(0.5, 50.0), Weibull(3.0, 8.0)
    b = Bathtub(infant=inf, random_rate=0.01, wearout=wo)
    a = 3.0
    assert float(b.cum_hazard(a)) == pytest.approx(float(inf.cum_hazard(a) + wo.cum_hazard(a)) + 0.03)


def test_node_table_scales_with_count_and_charges_consequences():
    comp = ComponentReliability("dimm", Weibull(1.0, 10.0), drain_hours=8, spare_embodied_kg=20,
                                repair_cost_usd=300, repair_logistics_kg=1, repair="minimal")
    tab = node_reliability_table([Slot(comp, 32)], n_quarters=4)
    q = tab[0]
    assert q.failures["dimm"] == pytest.approx(32 * 0.025)
    assert q.drain_hours == pytest.approx(32 * 0.025 * 8)
    assert q.spare_kg == pytest.approx(32 * 0.025 * 20)
    assert q.repair_cost_usd == pytest.approx(32 * 0.025 * 300)
    assert 0 < q.availability_loss < 1


def test_scaled_weibull_multiplies_hazard():
    w = Weibull(1.7, 6.0)
    w3 = w.scaled(3.0)
    for a in (0.5, 2.0, 7.0):
        assert float(w3.cum_hazard(a)) == pytest.approx(3.0 * float(w.cum_hazard(a)))
