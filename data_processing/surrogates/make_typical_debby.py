"""Debby typical field (revision 1, night 8, T1c): one linkless GRF realization with the median statistics of the six
final-set variables (vo, Tb, precip, CAPE, DIV200, MSLP): variance, lag-1 memory r1, e-folding lengths L_lat, L_lon.

Targets = medians of the *_src columns of night6/surrogates/validation_debby.csv (the source statistics of the six
variables, measured by the night-6 tooling, max-lag 60, censored). The field lives on the Debby analysis grid and time
axis (run script's own builder, night-6 final set), is stationary with unit sd map scaled by sqrt(c), per-time domain
mean removed (as the night-6 surrogates), and is written in the native layout of the CAPE file (int16 packed, name
DECOY) so the night-6 add-one wiring (--extra-inputs DECOY=<file>:DECOY) reads it. Fixed-point calibration with
make_grf_surrogates.calibrate (2 calibration realizations, seed BASE+3e6+9000); realizations m = 0..N-1 use seed
BASE+3e6+m (BASE 20261123); realization 0 is the one used in the runs. Validation: mean of the N realizations (as
the pools) within 10 % of the targets; realization 0's own values are reported.
PBS only:  python make_typical_debby.py --outdir DIR [--n 8]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

import make_grf_surrogates as G

BASE_SEED = 20261123
CASE_IDX = 3
R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
VAL = R1 / "night6/surrogates/validation_debby.csv"
TYPES = ["vo", "Tb", "precip", "CAPE", "DIV200", "MSLP"]
MAX_LAG = 60


def seed_for(member):
    return int(BASE_SEED + 1000000 * CASE_IDX + int(member))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--extras", type=int, nargs=2, default=None, metavar=("DRAWS", "K"), help="night 9 S11: write DRAWS x K independent fields draw<dd>/x_<jj>.nc on the analysis array")
    ap.add_argument("--analysis-grid", action="store_true", help="night 9 N1: write all n realizations as decoy_MM.nc on the analysis array of the case (pool for --decoy-pool)")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    v = pd.read_csv(VAL)
    v = v[v["source"].isin(TYPES)]
    assert len(v) == 6
    tgt = {k: float(np.median(v[f"{k}_src"])) for k in ("var", "r1", "L_lat", "L_lon")}
    print("targets", tgt, list(v["source"]), flush=True)
    m, c = G.build_case("debby", spec="night6")
    fd, periodic = G.analysis_arrays(c["full_data"])
    tmask, win = G.analysis_window(m, "debby", c, fd.time.values)
    shape = (fd.sizes["time"], fd.sizes["lat"], fd.sizes["lon"])
    print("analysis grid", shape, "periodic", periodic, "window", win, int(tmask.sum()), "steps", flush=True)
    ones = np.ones(shape[1:])
    cal = R1 / "night8/pools/debby/calib.json"
    if cal.exists():   # night 9: reuse the night-8 calibration (same targets, grid and seeds)
        params, hist = json.loads(cal.read_text())["params"], json.loads(cal.read_text())["hist"]
    else:
      params, hist = G.calibrate(tgt, shape, tmask, ones, ones, np.ones(shape[1:], bool), seed_for(9000), periodic, False,
                               max_lag=MAX_LAG, n_real=2, censor=True)
    print("calibrated", params, flush=True)
    if a.extras:   # seed BASE + 3e6 + 500000 + 1000 draw + slot
        nd, kk = a.extras
        fdc = c["full_data"]
        dims = [d for d in fdc.dims if d != "var"]
        stats = []
        for dd in range(nd):
            (out / f"draw{dd:02d}").mkdir(parents=True, exist_ok=True)
            for jj in range(1, kk + 1):
                sd = int(BASE_SEED + 1000000 * CASE_IDX + 500000 + 1000 * dd + jj)
                u = G.unit_grf_ar1(*shape, params["phi"], params["s_lat"], params["s_lon"], np.random.default_rng(sd),
                                   periodic_lon=periodic, pad=(params["pad_lat"], params["pad_lon"]))
                x = G.remove_domain_mean(u * np.float32(np.sqrt(params["c"])))
                st = G.field_stats(x, tmask, None, MAX_LAG, censor=True)[0]
                stats.append(dict(draw=dd, slot=jj, seed=sd, **st))
                xr.DataArray(x.astype(np.float32), dims=["time", "lat", "lon"]).transpose(*dims).assign_coords({d: fdc[d] for d in dims}).rename(f"X{jj:02d}").to_netcdf(out / f"draw{dd:02d}" / f"x_{jj:02d}.nc")
                del u, x
        S = pd.DataFrame(stats); S.to_csv(out / "extras_seeds.csv", index=False)
        print(S[["var", "r1", "L_lat", "L_lon"]].describe().loc[["mean", "std"]].to_string(), "targets", tgt, flush=True)
        return
    lay = G.native_layout("debby", dict(time=fd.time.values, lat=fd.lat.values, lon=fd.lon.values), periodic)
    rows = []
    for k in range(a.n):
        u = G.unit_grf_ar1(*shape, params["phi"], params["s_lat"], params["s_lon"], np.random.default_rng(seed_for(k)),
                           periodic_lon=periodic, pad=(params["pad_lat"], params["pad_lon"]))
        x = G.remove_domain_mean(u * np.float32(np.sqrt(params["c"])))
        del u
        q, scale = G.quantize(x)
        xq = q.astype(np.float32) * scale                       # values as read back from the file
        st = G.field_stats(xq, tmask, None, MAX_LAG, censor=True)[0]   # analysis window, as calibrated
        rows.append(dict(member=k, seed=seed_for(k), **st))
        if a.analysis_grid:   # night 9 N1: pool member on the case analysis array (dims and coordinates of full_data)
            fdc = c["full_data"]
            dims = [d for d in fdc.dims if d != "var"]
            sz = {"time": shape[0], "lat": shape[1], "lon": shape[2]}
            gen = [d for d in ("time", "lat", "lon")]
            da = xr.DataArray(xq, dims=gen).transpose(*dims).assign_coords({d: fdc[d] for d in dims}).rename("DECOY")
            da.attrs.update(case="debby", member=k, seed=seed_for(k), targets=json.dumps(tgt))
            da.to_netcdf(out / f"decoy_{k:02d}.nc")
        elif k == 0:
            G.write_surrogate(out / "decoy_00.nc", "DECOY", xq, lay,
                              dict(long_name="typical-field decoy of Debby (night 8 T1c)", seed=seed_for(0),
                                   targets=json.dumps(tgt), stationary=1, domain_mean_removed=1,
                                   generator="data_processing/surrogates/make_typical_debby.py"))
        del x, xq
    V = pd.DataFrame(rows)
    V.to_csv(out / "seeds.csv", index=False)
    chk = []
    for k in ("var", "r1", "L_lat", "L_lon"):
        mean = float(V[k].mean())
        chk.append(dict(stat=k, target=tgt[k], mean=mean, rel_err=(mean - tgt[k]) / abs(tgt[k]),
                        member0=float(V[k].iloc[0]), member0_rel_err=(float(V[k].iloc[0]) - tgt[k]) / abs(tgt[k]),
                        ok=abs(mean - tgt[k]) <= 0.10 * abs(tgt[k])))
    C = pd.DataFrame(chk)
    C.to_csv(out / "validation.csv", index=False)
    (out / "targets.json").write_text(json.dumps(dict(case="debby", **tgt, types=TYPES, grid=list(shape), window=list(win),
                                                      max_lag=MAX_LAG), indent=1))
    (out / "calib.json").write_text(json.dumps(dict(params=params, hist=hist), indent=1, default=float))
    (out / "validation.md").write_text(f"# Debby typical field: {a.n} validation realizations, targets = median over {TYPES}\n\n"
                                       + C.to_string(index=False) + f"\n\nall within 10 %: {bool(C.ok.all())}\n")
    (out / "sha256.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in sorted(out.glob("decoy_*.nc"))))
    print(C.to_string(index=False), "\nPASS" if C.ok.all() else "\nFAIL", flush=True)


if __name__ == "__main__":
    main()
