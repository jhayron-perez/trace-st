"""Night-3 N0: probe the structure of run outputs (PNW pickles, bundles, orography). PBS only."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
p = R1 / "A1_pnw_twin/z500_2006/run_0000/ensemble_results.pkl"
res = pickle.load(open(p, "rb"))
print("n members", len(res), "keys", list(res[0].keys()))
r = [x for x in res if x["success"]][0]
print("centers", np.asarray(r["centers"]).shape, "parents", np.asarray(r["parents"]))
out = r["out"]
print("out keys", list(out.keys()) if out is not None else None)
if out is not None and "debug" in out:
    dbg = out["debug"]
    print("n debug", len(dbg), [None if d is None else d.shape for d in dbg[:5]])
    d = [x for x in dbg if x is not None][0]
    print(d.columns.tolist()); print(d.head(10).to_string()); print(d.dtypes)
    print("winners[0]", out["winners"][0])
for f in [R1 / "A2_debby_twin/vo_2004/bundles/debby_trial_00084.npz",
          sorted((R1 / "A3_pinatubo_twin/so2twin_TMSO203/bundles").glob("*.npz"))[0]]:
    b = np.load(f, allow_pickle=True)
    for k in b.files:
        print(f.name, k, b[k].shape, b[k].dtype, (b[k].ravel()[:3] if b[k].size else ""))
import xarray as xr
ds = xr.open_dataset("/glade/u/home/jhayron/Causality_CaStLe/trace-st-rev/Scripts_Paper/GMTED2010_15n015_00625deg.nc")
print(ds)
for case in ["T6r_pnw", "A1_pnw_twin/z500_2006"]:
    s = pd.read_csv(R1 / case / "hyperparam_search_summary.csv")
    print(case, s.shape)
ids = [int(x) for x in open(R1 / "T6r_pnw/retained_run_ids.txt").read().split()]
P = pd.DataFrame([json.load(open(R1 / f"T6r_pnw/run_{i:04d}/params.json")) for i in ids])
for c in P.columns:
    v = P[c].astype(str)
    print("PNW param", c, v.nunique(), v.unique()[:8])
