"""Validation (blocking for Debby T8): rebuild the 2006 500-700 hPa vorticity anomaly and compare
with FilesDebby/Vo_anomalies_TC_500-700.nc4 for several candidate recipes.

python validate_vo.py OUTDIR CACHE_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import make_debby_extra_inputs as mk  # noqa: E402

REF = "/glade/work/jhayron/trace-st_data/original_submission/FilesDebby/Vo_anomalies_TC_500-700.nc4"
out = Path(sys.argv[1]); cache = sys.argv[2]
ref = xr.open_dataset(REF).vo.load()
ref = ref.assign_coords(time=ref.time.dt.round("1h"))
res = {}
for name, lev in {"lev5_500-700by50": [500, 550, 600, 650, 700], "lev3_500-600-700": [500, 600, 700]}.items():
    raw = mk.load_jas("vo500700", 2006, levels=lev, cache=cache)
    for stdz in ("hour", "all"):
        an = mk.standardize(raw, stdz).sel(time=slice("2006-08-11 00:00", "2006-08-28 23:00"))
        a = an.values.astype(float); b = ref.values.astype(float)
        assert a.shape == b.shape, (a.shape, b.shape)
        assert np.allclose(an.latitude, ref.latitude) and np.allclose(an.longitude, ref.longitude)
        m = np.isfinite(a) & np.isfinite(b)
        r = float(np.corrcoef(a[m], b[m])[0, 1]); d = np.abs(a[m] - b[m])
        res[f"{name}|{stdz}"] = dict(corr=r, max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                                     p99_abs_diff=float(np.percentile(d, 99)), n=int(m.sum()))
        print(name, stdz, res[f"{name}|{stdz}"], flush=True)
json.dump(res, open(out / "vo_validation.json", "w"), indent=1)
