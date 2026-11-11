"""Task 5 Pinatubo inventory: list variables (dims, frequency) in the extracted E3SMv2-SPA files,
flag tag-03 SO2/SO4 burdens, radiative fields, non-volcanic aerosols, and check that the Sandia/*
1991 fields are reproduced (raw 1 deg and after nearest-neighbour regrid to 4 deg as 6_PinatuboRuns.py).

python e3sm_inventory.py EXTRACT_DIR SANDIA_DIR OUT_PREFIX
"""
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ext, sandia, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
files = sorted(p for p in ext.rglob("*.nc"))
inv = []
for f in files:
    try:
        ds = xr.open_dataset(f, decode_times=False)
    except Exception as e:
        inv.append(dict(file=str(f), error=repr(e))); continue
    tl = ds.sizes.get("time", 0)
    freq = None
    if "time" in ds and tl > 1:
        freq = float(np.median(np.diff(ds["time"].values.astype(float))))
    for v, da in ds.data_vars.items():
        inv.append(dict(file=f.name, var=v, dims=list(da.dims), shape=list(da.shape),
                        long_name=str(da.attrs.get("long_name", "")), units=str(da.attrs.get("units", "")),
                        n_time=int(tl), dt_time_units=freq, time_units=str(ds["time"].attrs.get("units", "")) if "time" in ds else ""))
df = pd.DataFrame(inv); df.to_csv(out + "_variables.csv", index=False)
names = set(df.get("var", pd.Series(dtype=str)).dropna())
pat = {
    "tag03_SO2_SO4": r"(BURDENSO4.*03|TMSO2.*03|^SO2.*03|SO203|BURDENSO403|TMSO203)",
    "radiative": r"^(FLUT|FLUTC|QRS|QRL|FSNT|FSNTOA|FLNT|FSDS|FLDS)$",
    "dust": r"(DST|DUST|dst)", "sea_salt": r"(SSLT|ncl|SEASALT|_ss|BURDENSEASALT)", "black_carbon": r"(BC|bc_|BURDENBC)",
    "other_aerosol": r"(POM|SOA|MOM|BURDENPOM|BURDENSOA|AODDUST|AODSS|AODBC)",
}
flags = {k: sorted(n for n in names if re.search(p, n)) for k, p in pat.items()}
# Sandia reproduction
chk = {}
for sf in sorted(sandia.glob("*_199101_199112.nc")):
    v = sf.name.split("_1991")[0]
    hit = df[df.get("var") == v] if "var" in df else pd.DataFrame()
    if len(hit) == 0:
        chk[v] = "not found"; continue
    a = xr.open_dataset(sf, decode_times=False)[v]
    best = None
    for fn in hit["file"].unique():
        b = xr.open_dataset(next(ext.rglob(fn)), decode_times=False)[v]
        if b.shape[-2:] != a.shape[-2:]:
            continue
        # align on time values
        ta, tb = a["time"].values, b["time"].values
        common = np.intersect1d(ta, tb)
        if len(common) == 0:
            continue
        x = a.sel(time=common).values.astype(float); y = b.sel(time=common).values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        r = float(np.corrcoef(x[m], y[m])[0, 1]); d = float(np.nanmax(np.abs(x - y)))
        # 4-deg nearest neighbour as in 6_PinatuboRuns.regrid_global_nearest
        lat4 = np.arange(-90 + 2, 90, 4.0); lon4 = np.arange(0 + 2, 360, 4.0)
        xa = a.sel(time=common).interp(lat=lat4, lon=lon4, method="nearest").values
        yb = b.sel(time=common).interp(lat=lat4, lon=lon4, method="nearest").values
        d4 = float(np.nanmax(np.abs(xa - yb)))
        best = dict(file=fn, n_common_times=int(len(common)), sandia_times=int(len(ta)), corr=r, max_abs_diff_1deg=d,
                    max_abs_diff_4deg=d4, rel_max_diff=d / float(np.nanmax(np.abs(x))))
        break
    chk[v] = best or "found but grid/time did not align"
res = dict(n_files=len(files), n_vars=len(names), flags=flags, sandia_check=chk)
json.dump(res, open(out + "_summary.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
