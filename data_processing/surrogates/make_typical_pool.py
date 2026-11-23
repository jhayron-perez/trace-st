"""Typical-field decoy pool (revision 1, night 8): 30 independent realizations of a linkless field with the case's
median statistics over its final-set variables (variance, lag-1 memory r1, spatial e-folding lengths L_lat, L_lon).

Targets = medians of the night-6 (PNW) / night-7 A_indep (Pinatubo, --max-lag 45) validation tables' source statistics
over the final-set variables. Fields live on the case's analysis grid and time axis (taken from the run script's own
case builder); Pinatubo realizations get the loader's per-cell time standardization. Fixed-point calibration of
(phi, s_lat, s_lon, c) with make_grf_surrogates.calibrate (2 calibration realizations), then member m is generated with
seed BASE + 1e6 * case_idx + m (BASE 20261123; case_idx pnw 1, pinatubo 2, debby 3, synthetic 4).
Validation: mean over the 30 realizations of each statistic within 10 % of the target (per-realization values listed).
Writes <outdir>/decoy_MM.nc (variable DECOY, case dims), targets.json, calib.json, validation.{csv,md}, seeds.csv, sha256.txt.

python make_typical_pool.py --case {pnw,pinatubo} --outdir DIR [--n 30]   (PBS only)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import make_grf_surrogates as G

BASE_SEED = 20261123
CASE_IDX = {"pnw": 1, "pinatubo": 2, "debby": 3, "synthetic": 4}
R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
FINAL = dict(pnw=["z500", "tcwv", "olr", "slhf", "sshf", "z10", "t2m"],
             pinatubo=["TMSO201", "BURDENSO401", "TMH2SO401", "AEROD_v", "TMSO203", "BURDENSO403", "T050"])
VAL = dict(pnw=R1 / "night6/surrogates/validation_pnw.csv", pinatubo=R1 / "night7/surrogates/validation_pinatubo_indep.csv")
MAX_LAG = dict(pnw=60, pinatubo=45)


def seed_for(case, member):
    return int(BASE_SEED + 1000000 * CASE_IDX[case] + int(member))


def targets(case):
    v = pd.read_csv(VAL[case])
    v = v[v["source"].isin(FINAL[case])]
    assert len(v) == len(FINAL[case]), (case, len(v))
    t = {k: float(np.median(v[f"{k}_src"])) for k in ("var", "r1", "L_lat", "L_lon")}
    t["n_types"] = int(len(v))
    t["types"] = list(v["source"])
    return t


def case_axes(case):
    if case == "pnw":
        m = G.load_script("8_HW_PNW21_Runs_v2.py")
        c = m.build_case_pnw(extra_vars=["t2m"], extra_path=str(R1 / "night5/inputs_pnw"))
    else:
        os.environ["TST_PINATUBO_EXTRA"] = "TMSO203,BURDENSO403,T050"
        m = G.load_script("6_PinatuboRuns.py")
        c = m.build_case_pinatubo()
    fd = c["full_data"]
    dims = [d for d in fd.dims if d != "var"]
    return fd, dims


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["pnw", "pinatubo"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    case, out = a.case, Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tgt = targets(case)
    fd, dims = case_axes(case)
    coords = {d: fd[d] for d in dims}
    shape = tuple(len(fd[d]) for d in dims)
    ti, yi, xi = dims.index("time"), None, None
    lat = "lat" if "lat" in dims else "latitude"
    lon = "lon" if "lon" in dims else "longitude"
    order = [dims.index("time"), dims.index(lat), dims.index(lon)]   # generator order (time, lat, lon)
    nt, ny, nx = (shape[i] for i in order)
    lonv = np.asarray(fd[lon].values, dtype=float)
    dlon = abs(float(lonv[1] - lonv[0]))
    periodic = bool(abs(nx * dlon - 360.0) < 1e-6)
    print(case, "targets", tgt, "grid", (nt, ny, nx), "periodic", periodic, flush=True)
    restd = case == "pinatubo"
    ml = MAX_LAG[case]
    tmask = np.ones(nt, bool)
    ones = np.ones((ny, nx))
    target = {k: tgt[k] for k in ("var", "r1", "L_lat", "L_lon")}
    params, hist = G.calibrate(target, (nt, ny, nx), tmask, ones, ones, np.ones((ny, nx), bool), seed_for(case, 9000),
                               periodic, False, max_lag=ml, n_real=2, censor=True)
    print("calibrated", params, flush=True)
    rows = []
    for m in range(a.n):
        u = G.unit_grf_ar1(nt, ny, nx, params["phi"], params["s_lat"], params["s_lon"], np.random.default_rng(seed_for(case, m)),
                           periodic_lon=periodic, pad=(params["pad_lat"], params["pad_lon"]))
        x = (u * np.sqrt(params["c"])).astype(np.float32)
        if restd:
            x = G.standardize_time(x)
        st = G.field_stats(x, None, None, ml, censor=True)[0]
        rows.append(dict(member=m, seed=seed_for(case, m), **st))
        da = xr.DataArray(np.transpose(x, np.argsort(order)).astype(np.float32), dims=dims, coords=coords, name="DECOY",
                          attrs=dict(case=case, member=m, seed=seed_for(case, m), targets=json.dumps(target)))
        da.to_netcdf(out / f"decoy_{m:02d}.nc")
        del u, x
    V = pd.DataFrame(rows)
    V.to_csv(out / "seeds.csv", index=False)
    chk = []
    for k in ("var", "r1", "L_lat", "L_lon"):
        mean = float(V[k].mean())
        chk.append(dict(stat=k, target=target[k], mean=mean, rel_err=(mean - target[k]) / abs(target[k]),
                        worst_member_rel_err=float(((V[k] - target[k]) / abs(target[k])).abs().max()), ok=abs(mean - target[k]) <= 0.10 * abs(target[k])))
    C = pd.DataFrame(chk)
    C.to_csv(out / "validation.csv", index=False)
    (out / "targets.json").write_text(json.dumps(dict(case=case, **tgt, periodic=periodic, grid=[nt, ny, nx], dims=dims), indent=1))
    (out / "calib.json").write_text(json.dumps(dict(params=params, hist=hist), indent=1, default=float))
    (out / "validation.md").write_text(f"# typical-field pool {case}: {a.n} realizations, targets = median over {tgt['types']}\n\n"
                                       + C.to_string(index=False) + f"\n\nall within 10 %: {bool(C.ok.all())}\n")
    (out / "sha256.txt").write_text("\n".join(f"{sha256(p)}  {p.name}" for p in sorted(out.glob("decoy_*.nc"))) + "\n")
    print(C.to_string(index=False), "\nPASS" if C.ok.all() else "\nFAIL", flush=True)


if __name__ == "__main__":
    main()
