"""N4 (night 9): 'pooled' decoy pool for PNW: realization m takes the statistics (variance, r1, L_lat, L_lon) of ONE final-set
variable chosen at random (default_rng(20261123 + 99), seed recorded; picks.csv), instead of the median over the variables.
Each variable's calibration uses make_grf_surrogates.calibrate (2 calibration realizations, seed BASE+1e6*case+9000+10*i);
realization m uses seed BASE + 1e6*case + 7000 + m. Validation: each realization's measured statistics vs its own variable's
target (mean relative error per statistic within 10 %). Files decoy_MM.nc (variable DECOY) on the case analysis array.
PBS only:  python make_pooled_pool.py --case pnw --outdir DIR [--n 30]"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import make_grf_surrogates as G
import make_typical_pool as P

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="pnw")
ap.add_argument("--outdir", required=True)
ap.add_argument("--n", type=int, default=30)
a = ap.parse_args()
case, out = a.case, Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)
v = pd.read_csv(P.VAL[case])
v = v[v["source"].isin(P.FINAL[case])].reset_index(drop=True)
tg = [{k: float(r[f"{k}_src"]) for k in ("var", "r1", "L_lat", "L_lon")} for _, r in v.iterrows()]
src = list(v["source"])
fd, dims = P.case_axes(case)
coords = {d: fd[d] for d in dims}
shape = tuple(len(fd[d]) for d in dims)
lat, lon = "lat" if "lat" in dims else "latitude", "lon" if "lon" in dims else "longitude"
order = [dims.index("time"), dims.index(lat), dims.index(lon)]
nt, ny, nx = (shape[i] for i in order)
lonv = np.asarray(fd[lon].values, dtype=float)
periodic = bool(abs(nx * abs(float(lonv[1] - lonv[0])) - 360.0) < 1e-6)
ml = P.MAX_LAG[case]
tmask, ones = np.ones(nt, bool), np.ones((ny, nx))
pick = np.random.default_rng(20261123 + 99).integers(0, len(src), a.n)
need = sorted(set(int(i) for i in pick))
params = {}
for i in need:
    p, h = G.calibrate(tg[i], (nt, ny, nx), tmask, ones, ones, np.ones((ny, nx), bool), P.seed_for(case, 9000 + 10 * i),
                       periodic, False, max_lag=ml, n_real=2, censor=True)
    params[i] = p
    print("calibrated", src[i], p, flush=True)
rows = []
for m in range(a.n):
    i = int(pick[m])
    p = params[i]
    sd = int(P.BASE_SEED + 1000000 * P.CASE_IDX[case] + 7000 + m)
    u = G.unit_grf_ar1(nt, ny, nx, p["phi"], p["s_lat"], p["s_lon"], np.random.default_rng(sd), periodic_lon=periodic,
                       pad=(p["pad_lat"], p["pad_lon"]))
    x = (u * np.sqrt(p["c"])).astype(np.float32)
    st = G.field_stats(x, None, None, ml, censor=True)[0]
    rows.append(dict(member=m, seed=sd, source=src[i], **{f"meas_{k}": st[k] for k in ("var", "r1", "L_lat", "L_lon")},
                     **{f"tgt_{k}": tg[i][k] for k in ("var", "r1", "L_lat", "L_lon")}))
    xr.DataArray(np.transpose(x, np.argsort(order)).astype(np.float32), dims=dims, coords=coords, name="DECOY",
                 attrs=dict(case=case, member=m, seed=sd, source_stats=src[i])).to_netcdf(out / f"decoy_{m:02d}.nc")
    del u, x
V = pd.DataFrame(rows)
V.to_csv(out / "picks.csv", index=False)
chk = []
for k in ("var", "r1", "L_lat", "L_lon"):
    rel = (V[f"meas_{k}"] - V[f"tgt_{k}"]) / V[f"tgt_{k}"].abs()
    chk.append(dict(stat=k, mean_rel_err=float(rel.mean()), worst_abs_rel_err=float(rel.abs().max()), ok=bool(abs(rel.mean()) <= 0.10)))
C = pd.DataFrame(chk); C.to_csv(out / "validation.csv", index=False)
(out / "calib.json").write_text(json.dumps({src[i]: params[i] for i in need}, indent=1, default=float))
(out / "validation.md").write_text(f"# pooled decoy pool {case}: {a.n} realizations, statistics of a random final-set variable each\n\n"
                                   + V[["member", "source"]].to_string(index=False) + "\n\n" + C.to_string(index=False)
                                   + f"\n\nall within 10 %: {bool(C.ok.all())}\n")
(out / "sha256.txt").write_text("".join(f"{P.sha256(f)}  {f.name}\n" for f in sorted(out.glob("decoy_*.nc"))))
print(C.to_string(index=False), "\nPASS" if C.ok.all() else "\nFAIL")
