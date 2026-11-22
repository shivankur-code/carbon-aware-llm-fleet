"""Test the mechanism behind 'reuse saving grows under a guaranteed SLO':
what is the extra installed LC capacity (chance-constrained minus expected-value plan) made of,
and what does it cost in embodied vs operational carbon, for reuse and no-reuse plans?"""
import copy, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml  # noqa: E402
from carbonsched.grid import load_regions  # noqa: E402
from carbonsched.hardware import load_catalog  # noqa: E402
from carbonsched.milp import LifecycleModel, Scenario  # noqa: E402
from carbonsched.montecarlo import chance_constrained  # noqa: E402

cat = load_catalog(ROOT / "configs"); regions = load_regions(ROOT / "configs")
base = yaml.safe_load((ROOT / "configs/scenarios/fleet.yaml").read_text())
base.update(lc_failure_buffer=1.0, allow_cascade=False, allow_geo_cascade=False)
no_reuse = [s for s in base["buyable"] if not cat.skus[s].is_reuse]
out = {}
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for pname, kw in (("no_reuse", dict(buyable=no_reuse)), ("reuse", {})):
    def mk(reserve):
        b = copy.deepcopy(base); b.update(kw); b["lc_reserve"] = reserve
        return LifecycleModel(cat, regions, Scenario(**b))
    ev_model = mk(None); ev = ev_model.solve()
    cc, cc_model, reserve, hist = chance_constrained(mk, eps=0.001, dispersion=0.5, n_samples=30000, tol=0.0005, seed=SEED)
    def lc_cap(sol, model):
        f = sol.fleet[sol.fleet.tier == "LC"]
        by = {}
        for s, a, n in zip(f.sku, f.age_q, f.n):
            by[s] = by.get(s, 0.0) + model.cap(s, "LC", a) * n
        return by
    def nodeq(sol):
        return sol.fleet.groupby("sku").n.sum().to_dict()
    cap_ev, cap_cc = lc_cap(ev, ev_model), lc_cap(cc, cc_model)
    nq_ev, nq_cc = nodeq(ev), nodeq(cc)
    out[pname] = dict(
        carbon_ev=ev.carbon_kg, carbon_cc=cc.carbon_kg,
        delta_carbon_kg=cc.carbon_kg - ev.carbon_kg,
        delta_embodied_kg=cc.totals["embodied_kg"] - ev.totals["embodied_kg"],
        delta_operational_kg=cc.totals["operational_kg"] - ev.totals["operational_kg"],
        delta_repair_kg=cc.totals["repair_kg"] - ev.totals["repair_kg"],
        delta_lease_kg=cc.totals["lease_kg"] - ev.totals["lease_kg"],
        delta_lc_capacity_by_sku={s: cap_cc.get(s, 0) - cap_ev.get(s, 0) for s in set(cap_cc) | set(cap_ev)},
        delta_node_quarters_by_sku={s: nq_cc.get(s, 0) - nq_ev.get(s, 0) for s in set(nq_cc) | set(nq_ev)},
        total_reserve=sum(sum(v) for v in reserve.values()), iterations=len(hist))
    print(pname, json.dumps({k: (round(v, 1) if isinstance(v, float) else {kk: round(vv, 1) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in out[pname].items()}))
(ROOT / "results" / (f"p4_reserve_composition_seed{SEED}.json" if SEED else "p4_reserve_composition.json")).write_text(json.dumps(out, indent=2))
