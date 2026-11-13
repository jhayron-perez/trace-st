"""Revision-1 night 4 (A2): extra PNW21 decoy inputs (2 m temperature, total cloud cover) built exactly like
the paper's PNW anomaly files (`DataCaStLeBTs/JointFiles/<stem>_daily{,_anoms}.nc`).

Recipe (copied from the paper's data-processing notebooks, PaperCausalBTs/ScriptsDataProcessing, read-only):
  1. daily means (`Process_tcwv.py`): ERA5 ds633.0 `e5.oper.an.sfc` monthly hourly files, nearest-neighbour
     selection of lat -30..90 and lon 0..359 at 1 deg, `resample(time="1D").mean()`, 1995-01-01..2024-12-31;
     joined into one file `<stem>_daily.nc` (`2_JoinFiles.ipynb`).
  2. anomalies (`3_ComputeAnoms.ipynb`): drop Feb 29; climatological mean per day of year = mean over years of the
     29-day centred running mean (min_periods 1) of the daily series; anomaly = daily - clim mean; linear trend
     over years removed per (cell, day of year); std over years (ddof 0) per day of year, smoothed with a circular
     29-day running mean; standardized anomaly = detrended anomaly / smoothed std. (OLR is sign-flipped there;
     not needed here.)
  Differences from the notebook, numerical only: no dask; the per-cell linear trend uses the closed-form
  least-squares line (same estimator as np.polyfit deg 1); float64 arithmetic, written as float32.
  `--validate` rebuilds the paper's TCWV anomaly file from `JointFiles/tcwv_daily.nc` with this code and reports
  the max |difference|, and re-extracts June 2021 TCWV daily means and compares them with `tcwv_daily.nc`.

python make_pnw_extra_anoms.py --out DIR [--vars t2m tcc] [--nproc 32] [--validate]   (PBS only)
Outputs: DIR/{t2m,tcc}_daily.nc, DIR/{t2m,tcc}_daily_anoms.nc (variables VAR_2T, TCC), DIR/inputs.sha256,
DIR/validate_tcwv.txt
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ERA5 = "/glade/campaign/collections/gdex/data/d633000/"
JOINT = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/JointFiles")
SPEC = {  # stem -> (ERA5 code, variable name in the ERA5 file)
    "t2m": ("128_167_2t.ll025sc", "VAR_2T"),
    "tcc": ("128_164_tcc.ll025sc", "TCC"),
    "tcwv": ("128_137_tcwv.ll025sc", "TCWV"),   # validation only
}
LAT = np.arange(-30, 91, 1)
LON = np.arange(0, 360, 1)


def month_path(code, y, m):
    last = calendar.monthrange(y, m)[1]
    ym = f"{y}{m:02d}"
    return f"{ERA5}e5.oper.an.sfc/{ym}/e5.oper.an.sfc.{code}.{ym}0100_{ym}{last:02d}23.nc"


def daily_month(args):
    code, name, y, m = args
    with xr.open_dataset(month_path(code, y, m)) as ds:
        d = ds.sel(latitude=LAT, longitude=LON, method="nearest").load()
        return d[[name]].resample(time="1D").mean()


def build_daily(stem, nproc, years=(1995, 2024)):
    code, name = SPEC[stem]
    months = [(code, name, y, m) for y in range(years[0], years[1] + 1) for m in range(1, 13)]
    with get_context("fork").Pool(nproc) as pool:
        parts = pool.map(daily_month, months, chunksize=1)
    return xr.concat(parts, dim="time").sortby("time")


def anoms(daily: xr.Dataset) -> xr.DataArray:
    name = list(daily.data_vars)[0]
    t = pd.to_datetime(daily.time.values)
    data = daily[name].isel(time=~((t.day == 29) & (t.month == 2)))
    tt = pd.to_datetime(data.time.values)
    ny = len(tt.year.unique())
    if data.sizes["time"] != 365 * ny:
        raise ValueError("expected whole years")
    x = data.transpose("time", "latitude", "longitude").values.astype(np.float64)
    # 29-day centred running mean over the continuous series, min_periods 1 (xarray rolling semantics)
    sm = data.astype(np.float64).rolling(time=29, center=True, min_periods=1).mean() \
        .transpose("time", "latitude", "longitude").values
    nl, no = x.shape[1], x.shape[2]
    xr_ = x.reshape(ny, 365, nl, no)
    clim = sm.reshape(ny, 365, nl, no).mean(axis=0)
    a = xr_ - clim[None]
    yrs = np.asarray(sorted(tt.year.unique()), dtype=np.float64)
    yc = yrs - yrs.mean()
    slope = np.tensordot(yc, a, axes=(0, 0)) / float((yc * yc).sum())
    icpt = a.mean(axis=0)
    trend = icpt[None] + slope[None] * yc[:, None, None, None]
    ad = a - trend
    sd = ad.std(axis=0)                                   # ddof 0, as xarray .std
    h = 14
    wrap = np.concatenate([sd[-h:], sd, sd[:h]], axis=0)
    cs = np.cumsum(np.concatenate([np.zeros((1, nl, no)), wrap], axis=0), axis=0)
    sds = (cs[29:] - cs[:-29]) / 29.0                     # centred 29-day mean, length 365
    z = (ad / sds[None]).reshape(ny * 365, nl, no)
    out = xr.DataArray(z.astype(np.float32), dims=("time", "latitude", "longitude"),
                       coords=dict(time=data.time.values, latitude=data.latitude.values,
                                   longitude=data.longitude.values), name=name)
    return out.transpose("latitude", "longitude", "time")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vars", nargs="+", default=["t2m", "tcc"])
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.validate:
        ref_d = xr.open_dataset(JOINT / "tcwv_daily.nc")
        mine = anoms(ref_d)
        ref = xr.open_dataset(JOINT / "tcwv_daily_anoms.nc")["TCWV"].transpose("latitude", "longitude", "time")
        mine = mine.sel(latitude=ref.latitude, longitude=ref.longitude)
        d = np.abs(mine.values.astype(np.float64) - ref.values.astype(np.float64))
        jun = daily_month((SPEC["tcwv"][0], "TCWV", 2021, 6))["TCWV"]
        refj = ref_d["TCWV"].sel(time=slice("2021-06-01", "2021-06-30"))
        dj = np.abs(jun.values - refj.transpose(*jun.dims).values)
        msg = (f"tcwv anomalies rebuilt vs JointFiles/tcwv_daily_anoms.nc: max|d| {np.nanmax(d):.3e}, "
               f"mean|d| {np.nanmean(d):.3e}, p99.99 {np.nanpercentile(d, 99.99):.3e}, "
               f"corr {np.corrcoef(mine.values.ravel(), ref.values.ravel())[0, 1]:.9f}\n"
               f"tcwv June 2021 daily means re-extracted vs tcwv_daily.nc: max|d| {np.nanmax(dj):.3e}\n")
        (out / "validate_tcwv.txt").write_text(msg)
        print(msg, flush=True)
    for stem in a.vars:
        daily = build_daily(stem, a.nproc)
        pd_ = out / f"{stem}_daily.nc"
        daily.to_netcdf(pd_)
        print("wrote", pd_, dict(daily.sizes), flush=True)
        an = anoms(daily)
        pa = out / f"{stem}_daily_anoms.nc"
        an.to_netcdf(pa)
        print("wrote", pa, dict(an.sizes), float(an.mean()), float(an.std()), flush=True)
    with open(out / "inputs.sha256", "a") as f:
        for stem in a.vars:
            for suf in ("_daily.nc", "_daily_anoms.nc"):
                p = out / f"{stem}{suf}"
                f.write(f"{sha256(p)}  {p}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
