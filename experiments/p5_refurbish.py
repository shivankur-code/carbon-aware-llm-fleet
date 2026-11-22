"""Phase 5: when does refurbishment (swapping worn DIMMs / SSDs for new parts) pay in carbon?

Sweeps DIMM failure rate (AFR of a new DIMM) x reused-part hazard multiplier on the reference fleet and
reports the extra saving of allowing refurbishment on top of reliability-aware reuse (E2b -> E4), plus the
refurbishment volume by path. Accounting is exact per refurbishment age (one derived SKU per age).

Usage: python experiments/p5_refurbish.py
"""
import copy, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from carbonsched.grid import load_regions  # noqa: E402
from carbonsched.hardware import load_catalog  # noqa: E402
from carbonsched.milp import LifecycleModel, Scenario  # noqa: E402

base = yaml.safe_load((ROOT / "configs/scenarios/fleet.yaml").read_text())
regions = load_regions(ROOT / "configs")
rows = []
for afr in (0.01, 0.03, 0.05):
    for mult in (1, 3, 5, 8):
        cat = load_catalog(ROOT / "configs", {"components.components.dram_dimm.failure.afr": afr,
                                              "components.reused_part_afr_multiplier": mult})
        def solve(**kw):
            b = copy.deepcopy(base); b.update(kw)
            return LifecycleModel(cat, regions, Scenario(**b)).solve()
        e0 = solve(buyable=["new_gpu"]); e2b = solve(); e4 = solve(allow_refurbish=True)
        d = e4.decisions; r = d[d.decision == "ref"]
        fam = {}
        for k, v in zip(r.key, r.value):
            f = cat.refurb[k[0]].family; fam[f] = fam.get(f, 0) + v
        rows.append(dict(dimm_afr=afr, reused_multiplier=mult, Y=100 * (1 - e2b.carbon_kg / e0.carbon_kg),
                         saving_with_refurb=100 * (1 - e4.carbon_kg / e0.carbon_kg),
                         refurb_gain_pp=100 * (e2b.carbon_kg - e4.carbon_kg) / e0.carbon_kg,
                         refurbished_nodes=sum(fam.values()), **{f"ref_{k}": round(v, 1) for k, v in fam.items()}))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
df = pd.DataFrame(rows).fillna(0)
df.to_csv(ROOT / "results" / "p5_refurbish.csv", index=False)
