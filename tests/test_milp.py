import math

import pytest

from conftest import CONFIGS
from carbonsched.carbon import purchase_embodied_kg
from carbonsched.grid import load_regions
from carbonsched.hardware import load_catalog
from carbonsched.milp import HOURS_Q, LifecycleModel, Scenario, pareto

TOY = CONFIGS / "scenarios" / "toy.yaml"


@pytest.fixture(scope="module")
def cat():
    return load_catalog(CONFIGS)


@pytest.fixture(scope="module")
def regions():
    return load_regions(CONFIGS)


def single_sku(**kw):
    base = dict(name="single", horizon_q=8, regions=["mixed"], skus=["new_gpu"], buyable=["new_gpu"],
                max_age_q={"new_gpu": 28}, demand={"LC": {"mixed": 10.0}, "BATCH": 0.0},
                reliability=False, allow_lease=False, allow_cascade=False, util_cap=0.8)
    base.update(kw)
    return Scenario(**base)


def test_single_sku_matches_hand_calculation(cat, regions):
    sc = single_sku()
    sol = LifecycleModel(cat, regions, sc).solve()
    assert sol.status == "Optimal"
    nodes = math.ceil(10 / 0.8)  # 13
    buys = sol.decisions[sol.decisions.decision == "buy"]
    assert buys.value.sum() == pytest.approx(nodes)
    assert set(buys.t) == {0}
    k, r = cat.skus["new_gpu"], regions["mixed"]
    E = purchase_embodied_kg(k, "cutoff")
    hand = 0.0
    for t in range(8):
        g = r.pue * float(r.ci(t)) / 1000.0
        hand += nodes * E / 24                                         # amortized embodied (6-yr life)
        hand += nodes * k.power_idle_w / 1000 * HOURS_Q * g            # idle energy of every node
        hand += 10 * (k.power_peak_w - k.power_idle_w) / 1000 * HOURS_Q * g  # dynamic energy of served work
    assert sol.carbon_kg == pytest.approx(hand, rel=1e-6)
    assert sol.totals["shortfall"] == pytest.approx(0)


def test_failures_need_at_least_as_many_nodes_and_more_carbon(cat, regions):
    a = LifecycleModel(cat, regions, single_sku()).solve()
    b = LifecycleModel(cat, regions, single_sku(reliability=True)).solve()
    nodes = lambda s: round(s.decisions[s.decisions.decision == "buy"].value.sum())  # noqa: E731
    assert nodes(b) >= nodes(a)
    assert b.carbon_kg > a.carbon_kg
    assert b.totals["repair_kg"] > 0 == a.totals["repair_kg"]


def test_demand_growth_triggers_later_buys(cat, regions):
    sc = single_sku(demand={"LC": {"mixed": [10, 10, 10, 10, 20, 20, 20, 20]}, "BATCH": 0.0})
    sol = LifecycleModel(cat, regions, sc).solve()
    buys = sol.decisions[sol.decisions.decision == "buy"].groupby("t").value.sum().round()
    assert buys[0] == 13 and buys[4] == 12  # 20 units need ceil(20/0.8) = 25 nodes; 13 exist


def two_sku(region, pool=10_000, **kw):
    return single_sku(regions=[region], skus=["new_gpu", "green_gpu"], buyable=["new_gpu", "green_gpu"],
                      max_age_q={"new_gpu": 28, "green_gpu": 28}, demand={"LC": {region: 10.0}, "BATCH": 0.0},
                      pools={"dram_dimm": {"initial": pool * 32}, "ssd": {"initial": pool * 8}}, **kw)


def bought(sol, s):
    d = sol.decisions[sol.decisions.decision == "buy"]
    return round(d[d.key.apply(lambda k: k[0] == s)].value.sum())


def test_green_chosen_on_clean_grid_new_on_dirty(cat, regions):
    # Consistent with P1 finding F1: reuse wins on clean grids, loses for LC on dirty ones.
    clean = LifecycleModel(cat, regions, two_sku("clean")).solve()
    dirty = LifecycleModel(cat, regions, two_sku("dirty")).solve()
    assert bought(clean, "green_gpu") > 0 and bought(clean, "new_gpu") == 0
    # On the dirty grid the busy nodes are new; a reused-part node may still fill the
    # last, lightly-used slot because at low utilization embodied carbon dominates.
    assert bought(dirty, "new_gpu") >= 12 and bought(dirty, "green_gpu") <= 1


def test_empty_reuse_pool_blocks_green(cat, regions):
    sol = LifecycleModel(cat, regions, two_sku("clean", pool=0)).solve()
    assert bought(sol, "green_gpu") == 0 and bought(sol, "new_gpu") > 0


def test_pool_limits_green_builds(cat, regions):
    sol = LifecycleModel(cat, regions, two_sku("clean", pool=5)).solve()
    assert bought(sol, "green_gpu") == 5
    assert bought(sol, "new_gpu") == 13 - 5 or bought(sol, "new_gpu") + 5 >= 13


def test_cascade_instead_of_retire_and_rebuy(cat, regions):
    sc = single_sku(allow_cascade=True, demand={"LC": {"mixed": [10, 10, 10, 10, 0, 0, 0, 0]},
                                               "BATCH": [0, 0, 0, 0, 10, 10, 10, 10]})
    sol = LifecycleModel(cat, regions, sc).solve()
    d = sol.decisions
    assert d[d.decision == "casc"].value.sum() == 13
    assert d[(d.decision == "buy") & (d.t > 0)].empty
    # node count is conserved across the cascade
    per_t = sol.fleet.groupby("t").n.sum()
    assert per_t.round(6).eq(13).all()


def test_lease_covers_unmet_demand(cat, regions):
    # forbid buying entirely -> must lease everything
    sc = single_sku(buyable=[], allow_lease=True)
    sol = LifecycleModel(cat, regions, sc).solve()
    leased = sol.decisions[sol.decisions.decision == "lease"].value.sum()
    assert leased == pytest.approx(80)
    assert sol.totals["lease_kg"] > 0 and sol.totals["shortfall"] == pytest.approx(0)


def test_fixed_plan_reevaluation_under_failures(cat, regions):
    # A plan made ignoring failures, replayed with failures: same fleet, more carbon, gaps leased.
    model0 = LifecycleModel(cat, regions, single_sku(allow_lease=True))
    plan = model0.solve()
    model1 = LifecycleModel(cat, regions, single_sku(allow_lease=True, reliability=True))
    replay = model1.solve(fix={"buy": plan.values["buy"], "ret": plan.values["ret"]})
    assert replay.status == "Optimal"
    assert bought(replay, "new_gpu") == bought(plan, "new_gpu")
    assert replay.carbon_kg > plan.carbon_kg


def test_toy_scenario_and_pareto(cat, regions):
    sc = Scenario.from_yaml(TOY)
    m = LifecycleModel(cat, regions, sc)
    sol = m.solve()
    assert sol.status == "Optimal" and sol.totals["shortfall"] == pytest.approx(0, abs=1e-6)
    front = pareto(m, "carbon", "cost", n_points=4)
    assert len(front) >= 3
    f = front.sort_values("cost_usd")
    # along the front, allowing more cost never increases minimum carbon
    assert all(f.carbon_kg.diff().dropna() <= 1e-3 * f.carbon_kg.max())


def test_lp_relaxation_is_scale_invariant_and_fast(cat, regions):
    a = LifecycleModel(cat, regions, Scenario.from_yaml(TOY, integer=False)).solve()
    b = LifecycleModel(cat, regions, Scenario.from_yaml(TOY, integer=False, scale=50)).solve()
    assert b.carbon_kg == pytest.approx(50 * a.carbon_kg, rel=1e-6)
    assert not a.time_limited and b.solve_seconds < 30


def test_lc_reserve_raises_installed_capacity_and_carbon(cat, regions):
    a = LifecycleModel(cat, regions, single_sku(integer=False)).solve()
    b_model = LifecycleModel(cat, regions, single_sku(integer=False, lc_reserve={"mixed": [2.0] * 8}))
    b = b_model.solve()
    for t in range(8):
        f = b.fleet[(b.fleet.t == t) & (b.fleet.tier == "LC")]
        installed = sum(b_model.cap(s, "LC", a_) * n for s, a_, n in zip(f.sku, f.age_q, f.n))
        assert installed >= 12.0 - 1e-6           # demand 10 + reserve 2
    assert b.carbon_kg > a.carbon_kg              # idle reserve nodes cost carbon


def test_chance_constrained_loop_meets_target(cat, regions):
    from carbonsched.montecarlo import chance_constrained, replay_lc
    sc_kw = dict(integer=False, reliability=True, lc_failure_buffer=1.0, demand={"LC": {"mixed": 200.0}, "BATCH": 0.0})

    def factory(reserve):
        return LifecycleModel(cat, regions, single_sku(lc_reserve=reserve, **sc_kw))
    sol, model, reserve, hist = chance_constrained(factory, eps=0.01, n_samples=4000, tol=0.0005)
    assert hist[-1]["worst_rel_gap"] <= 0.0005 or len(hist) == 12
    check = replay_lc(model, sol, 20000, None, seed=7)
    assert check.p_violation.max() <= 0.03       # 1% target, MC noise + rounding allowance
