"""Explore recipe variants for Vo_anomalies_TC_500-700.nc4 (diagnostic for the blocking validation)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
out, cache = Path(sys.argv[1]), Path(sys.argv[2])  # same order as validate_vo.py
REF = "/glade/work/jhayron/trace-st_data/original_submission/FilesDebby/Vo_anomalies_TC_500-700.nc4"
ref = xr.open_dataset(REF).vo.load(); ref = ref.assign_coords(time=ref.time.dt.round("1h"))
files = [cache / f"128_138_vo_{d:%Y%m%d}.nc" for d in pd.date_range("2006-07-01", "2006-09-30")]
raw = xr.concat([xr.open_dataset(f)["VO"].load() for f in files], "time").astype("float64")  # time, level, lat, lon
W = slice("2006-08-11 00:00", "2006-08-28 23:00")
def score(an, name):
    a = an.sel(time=W).values; b = ref.values
    m = np.isfinite(a) & np.isfinite(b); d = np.abs(a - b)[m]
    r = dict(corr=float(np.corrcoef(a[m], b[m])[0, 1]), max=float(d.max()), mean=float(d.mean()), p99=float(np.percentile(d, 99)))
    print(name, r, flush=True); return r
def hstd(da, sub=None):
    base = da if sub is None else da.sel(time=sub)
    mu = base.groupby("time.hour").mean("time"); sd = base.groupby("time.hour").std("time")
    return (da.groupby("time.hour") - mu).groupby("time.hour") / sd
res = {}
L5 = [500, 550, 600, 650, 700]
v5 = raw.sel(level=L5).mean("level")
res["lev5_hour_JAS"] = score(hstd(v5), "lev5_hour_JAS")



# standardize each level, then average
res["perlevel_hour_then_mean"] = score(xr.concat([hstd(raw.sel(level=l).drop_vars("level")) for l in L5], "level").mean("level"), "perlevel_hour_then_mean")
res["perlevel_hour_mean_restd"] = score(hstd(xr.concat([hstd(raw.sel(level=l).drop_vars("level")) for l in L5], "level").mean("level")), "perlevel_hour_mean_restd")
# daily-cycle-free: subtract hourly mean, divide by overall sd
an = (v5.groupby("time.hour") - v5.groupby("time.hour").mean("time")) / v5.std("time")
res["lev5_hourmean_allsd"] = score(an, "lev5_hourmean_allsd")
# time shifts
for s in (-1, 1):
    sh = hstd(v5).shift(time=s); res[f"lev5_hour_shift{s}"] = score(sh, f"lev5_hour_shift{s}")
# where are the large diffs?
a = hstd(v5).sel(time=W).values; d = np.abs(a - ref.values)
it, iy, ix = np.unravel_index(np.nanargmax(d), d.shape)
res["argmax"] = dict(time=str(ref.time.values[it]), lat=float(ref.latitude[iy]), lon=float(ref.longitude[ix]), new=float(a[it, iy, ix]), ref=float(ref.values[it, iy, ix]))
res["frac_gt_1sd"] = float(np.nanmean(d > 1)); res["by_hour_mean_absdiff"] = {int(h): float(np.nanmean(d[ref.hour.values == h])) for h in range(24)}
res["by_lat_band_mean_absdiff"] = {f"{lo}-{lo+10}": float(np.nanmean(d[:, (ref.latitude.values >= lo) & (ref.latitude.values < lo + 10), :])) for lo in (0, 10, 20, 30)}
print(json.dumps({k: res[k] for k in ("argmax", "frac_gt_1sd", "by_lat_band_mean_absdiff")}, indent=1))
json.dump(res, open(out / "vo_validation_variants.json", "w"), indent=1)
