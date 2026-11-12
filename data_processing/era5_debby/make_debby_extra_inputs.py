"""Build extra Debby 2006 input fields from ERA5 (RDA ds633.0 on GLADE), matching FilesDebby/*.

Revision-1 (R1 major 1, DECISIONS 2026-11-04). The original FilesDebby preprocessing
script is not on Derecho (K12), so this reconstructs it and must first reproduce
Vo_anomalies_TC_500-700.nc4 (validation mode).

Grid/domain as FilesDebby: 0.25 deg, lat 40 -> 0 (161, descending), lon -50 -> 40 (361).
Time: hourly; output window 2006-08-11 00:00 .. 2006-08-28 23:00 (432 steps).
Anomaly: per grid cell, standardized with mean and sd over JUL-SEP of the same year;
--stdz hour: separately for each hour of day (default), --stdz all: over all JAS hours.
Year shift (placebo): --year 2005 --relabel-year 2006 takes the same calendar dates
from 2005 (with 2005 JAS stats) and relabels the time coordinate to 2006.

Fields (--field):
  vo500700 : relative vorticity, mean over --levels (default 500 550 600 650 700 hPa)
  d200     : divergence at 200 hPa (--levels 200)
  cape     : CAPE (surface analysis)

Usage:
  python make_debby_extra_inputs.py --field vo500700 --year 2006 --out OUT.nc [--nproc 8]
"""
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RDA = Path("/glade/campaign/collections/rda/data/d633000")
LAT = np.arange(40.0, -0.001, -0.25)          # 161
LON = np.arange(-50.0, 40.001, 0.25)          # 361
WIN = ("08-11 00:00", "08-28 23:00")

SPEC = {
    "vo500700": dict(kind="pl", code="128_138_vo", var="VO", out="vo", levels=[500, 550, 600, 650, 700]),
    "d200": dict(kind="pl", code="128_155_d", var="D", out="d", levels=[200]),
    "cape": dict(kind="sfc", code="128_059_cape", var="CAPE", out="cape", levels=None),
}


def _subset(da):
    da = da.sel(latitude=slice(40.0, 0.0))
    lon = ((da.longitude + 180.0) % 360.0) - 180.0
    da = da.assign_coords(longitude=lon).sortby("longitude").sel(longitude=slice(-50.0, 40.0))
    return da


def _read_pl_day(args):
    code, var, levels, day = args
    f = RDA / "e5.oper.an.pl" / day.strftime("%Y%m") / (
        f"e5.oper.an.pl.{code}.ll025sc.{day:%Y%m%d}00_{day:%Y%m%d}23.nc")
    with xr.open_dataset(f) as ds:
        da = _subset(ds[var].sel(level=levels)).mean("level").load()
    return da.astype("float64")


def _read_sfc_month(args):
    code, var, month = args
    end = (month + pd.offsets.MonthEnd(0))
    f = RDA / "e5.oper.an.sfc" / month.strftime("%Y%m") / (
        f"e5.oper.an.sfc.{code}.ll025sc.{month:%Y%m}0100_{end:%Y%m%d}23.nc")
    with xr.open_dataset(f) as ds:
        da = _subset(ds[var]).load()
    return da.astype("float64")


def load_jas_cache(field, year, cache, levels=None):
    """Read JAS of `year` from the extract_cache.py cache (preferred; one read of ERA5 per day)."""
    s = SPEC[field]
    lev = levels if levels is not None else s["levels"]
    if s["kind"] == "pl":
        files = [Path(cache) / f"{s['code']}_{d:%Y%m%d}.nc" for d in pd.date_range(f"{year}-07-01", f"{year}-09-30", freq="D")]
    else:
        files = [Path(cache) / f"{s['code']}_{m:%Y%m}.nc" for m in pd.date_range(f"{year}-07-01", f"{year}-09-01", freq="MS")]
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            da = ds[s["var"]]
            if lev is not None:
                da = da.sel(level=lev).mean("level")
            parts.append(da.load().astype("float64"))
    da = xr.concat(parts, "time")
    assert da.sizes["latitude"] == 161 and da.sizes["longitude"] == 361, da.sizes
    return da


def load_jas(field, year, levels=None, nproc=8, cache=None):
    if cache is not None:
        return load_jas_cache(field, year, cache, levels=levels)
    s = SPEC[field]
    lev = levels if levels is not None else s["levels"]
    with mp.get_context("fork").Pool(nproc) as pool:
        if s["kind"] == "pl":
            days = pd.date_range(f"{year}-07-01", f"{year}-09-30", freq="D")
            parts = pool.map(_read_pl_day, [(s["code"], s["var"], lev, d) for d in days])
        else:
            months = pd.date_range(f"{year}-07-01", f"{year}-09-01", freq="MS")
            parts = pool.map(_read_sfc_month, [(s["code"], s["var"], m) for m in months])
    da = xr.concat(parts, "time")
    assert da.sizes["latitude"] == 161 and da.sizes["longitude"] == 361, da.sizes
    return da


def standardize(da, stdz="hour"):
    if stdz == "hour":
        g = da.groupby("time.hour")
        return (g - g.mean("time")).groupby("time.hour") / da.groupby("time.hour").std("time")
    return (da - da.mean("time")) / da.std("time")


def select_window(an, year, shift_days=0):
    """Event window of `year`; night-3 N2 (opt-in): shift_days != 0 takes the window K days later
    (K < 0: earlier) from the same JAS anomalies and relabels it back onto the event dates."""
    k = pd.Timedelta(days=int(shift_days))
    t0 = pd.Timestamp(f"{year}-{WIN[0]}") + k
    t1 = pd.Timestamp(f"{year}-{WIN[1]}") + k
    an = an.sel(time=slice(t0, t1))
    if int(shift_days) != 0:
        an = an.assign_coords(time=pd.to_datetime(an.time.values) - k)
    return an


def roll_lon(an, shift_lon=0.0):
    """Night-3 N2 (opt-in): shift the field east by shift_lon degrees, wrapping inside the domain
    (values leaving the east edge re-enter at the west edge); coordinates unchanged."""
    if not shift_lon:
        return an
    dlon = float(an.longitude[1] - an.longitude[0])
    return an.roll(longitude=int(round(float(shift_lon) / dlon)), roll_coords=False)


def build(field, year, relabel_year=2006, stdz="hour", levels=None, nproc=8, cache=None, shift_days=0, shift_lon=0.0):
    da = load_jas(field, year, levels=levels, nproc=nproc, cache=cache)
    an = standardize(da, stdz)
    an = roll_lon(select_window(an, year, shift_days), shift_lon)
    if relabel_year != year:
        t = pd.to_datetime(an.time.values)
        an = an.assign_coords(time=[ts.replace(year=relabel_year) for ts in t])
    an = an.drop_vars([c for c in an.coords if c not in ("time", "latitude", "longitude")])
    an = an.assign_coords(hour=("time", pd.to_datetime(an.time.values).hour.astype("int64")))
    out = an.astype("float32").rename(SPEC[field]["out"])
    extra = dict(shift_days=int(shift_days), shift_lon=float(shift_lon)) if (shift_days or shift_lon) else {}
    out.attrs = dict(source="ERA5 ds633.0 " + SPEC[field]["code"], year_source=int(year), **extra,
                     year_label=int(relabel_year), levels_hPa=str(levels or SPEC[field]["levels"]),
                     standardization=f"per grid cell, JAS {year}, by={stdz}")
    assert out.sizes["time"] == 432, out.sizes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True, choices=list(SPEC))
    ap.add_argument("--year", type=int, default=2006)
    ap.add_argument("--relabel-year", type=int, default=2006)
    ap.add_argument("--stdz", choices=["hour", "all"], default="hour")
    ap.add_argument("--levels", type=float, nargs="+", default=None)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--cache", default=None, help="extract_cache.py output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift-days", type=int, default=0, help="N2 null: window K days later (same JAS), relabelled")
    ap.add_argument("--shift-lon", type=float, default=0.0, help="N2 null: shift east by D degrees, wrap in domain")
    a = ap.parse_args()
    out = build(a.field, a.year, a.relabel_year, a.stdz, a.levels, a.nproc, a.cache, a.shift_days, a.shift_lon)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_dataset().to_netcdf(a.out)
    print("wrote", a.out, dict(out.sizes))


if __name__ == "__main__":
    main()
