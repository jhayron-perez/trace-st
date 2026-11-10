"""Stage 1: extract the Debby domain (lat 40..0, lon -50..40, 0.25 deg) from ERA5 ds633.0 once per
source file into a small cache, so validation and production read the 1.8 GB/day files only once.

python extract_cache.py CACHE_DIR NPROC
Writes CACHE_DIR/<code>_<YYYYMMDD>.nc (pl: all requested levels kept separately) and
CACHE_DIR/cape_<YYYYMM>.nc. Spawn pool (fork + netCDF/HDF5 can deadlock).
"""
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

RDA = Path("/glade/campaign/collections/rda/data/d633000")
JOBS = []
for year in (2006, 2005):
    for d in pd.date_range(f"{year}-07-01", f"{year}-09-30", freq="D"):
        JOBS.append(("pl", "128_138_vo", "VO", [500, 550, 600, 650, 700], d))
for d in pd.date_range("2006-07-01", "2006-09-30", freq="D"):
    JOBS.append(("pl", "128_155_d", "D", [200], d))
for m in pd.date_range("2006-07-01", "2006-09-01", freq="MS"):
    JOBS.append(("sfc", "128_059_cape", "CAPE", None, m))


def subset(da):
    da = da.isel(latitude=slice(200, 361))                      # 40.0 .. 0.0
    west = da.isel(longitude=slice(1240, 1440))                 # 310.0 .. 359.75 -> -50 .. -0.25
    east = da.isel(longitude=slice(0, 161))                     # 0.0 .. 40.0
    west = west.assign_coords(longitude=west.longitude - 360.0)
    return xr.concat([west, east], "longitude")


def work(job):
    kind, code, var, levels, t = job
    out = Path(CACHE) / (f"{code}_{t:%Y%m%d}.nc" if kind == "pl" else f"{code}_{t:%Y%m}.nc")
    if out.exists():
        return str(out), "exists"
    if kind == "pl":
        f = RDA / "e5.oper.an.pl" / f"{t:%Y%m}" / f"e5.oper.an.pl.{code}.ll025sc.{t:%Y%m%d}00_{t:%Y%m%d}23.nc"
    else:
        end = t + pd.offsets.MonthEnd(0)
        f = RDA / "e5.oper.an.sfc" / f"{t:%Y%m}" / f"e5.oper.an.sfc.{code}.ll025sc.{t:%Y%m}0100_{end:%Y%m%d}23.nc"
    with xr.open_dataset(f) as ds:
        da = ds[var]
        if levels is not None:
            da = da.sel(level=levels)
        da = subset(da).load()
    tmp = out.with_suffix(".tmp")
    da.to_dataset().to_netcdf(tmp)
    tmp.rename(out)
    return str(out), "ok"


def init(c):
    global CACHE
    CACHE = c


if __name__ == "__main__":
    CACHE = sys.argv[1]; Path(CACHE).mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[2])
    with mp.get_context("spawn").Pool(n, initializer=init, initargs=(CACHE,)) as pool:
        for i, (p, st) in enumerate(pool.imap_unordered(work, JOBS)):
            print(i + 1, len(JOBS), st, p, flush=True)
