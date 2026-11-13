"""Night-4 D: Debby specificity re-analysis (existing outputs; no new trajectories).

D1 path accuracy vs nulls. Metric = the paper's Debby track metric in results_jobs: per trial, the mean over
members of each member's mean distance to the IBTrACS track of Debby (km; `mean_of_member_mean_track_dist_km`,
also the median version). Real run = rebuilt-vorticity baseline T4r_debby_vorebuilt (M 30). Null runs: night-3
N2 (add design: vo + a time-/space-shifted copy; replace design: vo replaced by another year or a +-10 d shift;
M 10), night-2 A2 twins (vo + other-year vo, M 30), night-4 A1 (+ CAPE, DIV200) and B1 (ko0_10 + twins).
Statistic per run: d = dist(run) - dist(real) per config (positive = the real run is closer to the track),
mean and bootstrap CI over the 75 retained configs present in both. The mean distance over members does not
depend on M in expectation, so M 10 vs M 30 adds noise but no bias.

D2 wave-train check. Field: vo500700_2006_rebuilt.nc (hourly, 500-700 hPa mean relative vorticity, Debby
domain). Region: cells within 5 deg of the IBTrACS Debby positions (and the starting point) over the 15 days
before 2006-08-26 00 UTC (the precursor path). For each hour t in that window: spatial pattern correlation over the
region between vo(t) and vo(t + K) (K = +-5, +-10 d; the N2 time shifts) and between vo(t) and the field shifted
east by D = 10, 20 deg (value at lon x taken from x - D; the N2 space shifts); mean over t. Also the lag-
correlation curve (K = 0..15 d) and the dominant period of the region-mean vo (periodogram of JAS hourly series,
detrended, 2-10 d band).

python N4_debby_D.py   (PBS only). Outputs: revision1/night4/analysis/d1_path_accuracy.csv, d2_wavetrain.csv,
d2_lagcurve.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from A4_twins import boot_ci, retained_ids  # noqa: E402

R1 = C.R1
N3 = R1 / "night3"; N4 = R1 / "night4"
OUT = N4 / "analysis"
IBTRACS = "/glade/u/home/jhayron/Causality_CaStLe/PaperCausalBTs/Scripts_SciAdv_Paper/IBTrACS.NA.v04r01.nc"
DATE_END = pd.Timestamp("2006-08-26 00:00:00")
START = (28.0, -46.0)
METRICS = ["mean_of_member_mean_track_dist_km", "median_of_member_mean_track_dist_km"]


def jobs(d):
    for n in ("results_jobs_final.csv", "results_jobs_snapshot.csv"):
        if (d / n).exists():
            return pd.read_csv(d / n).drop_duplicates("trial_id", keep="last").set_index("trial_id")
    return None


def d1():
    ids = retained_ids("debby")
    real = jobs(R1 / "T4r_debby_vorebuilt")
    runs = []
    for n, fam in (("SM10", "add time -10 d"), ("SM5", "add time -5 d"), ("SP5", "add time +5 d"),
                   ("SP10", "add time +10 d"), ("L10", "add space +10 deg"), ("L20", "add space +20 deg")):
        runs.append((f"N2 debby_add_{n}", fam, N3 / "n2" / "base" / f"debby_add_{n}"))
    for y in (2003, 2004, 2005, 2007, 2008):
        runs.append((f"N2 debby_rep{y}", "replace other year", N3 / "n2" / "base" / f"debby_rep{y}"))
        runs.append((f"A2 vo_{y}", "add twin (other year)", R1 / "A2_debby_twin" / f"vo_{y}"))
    for n in ("SM10", "SP10"):
        runs.append((f"N2 debby_rep{n}", "replace time +-10 d", N3 / "n2" / "base" / f"debby_rep{n}"))
    for m in ("base", "ko0_10"):
        runs.append((f"A1 {m} +CAPE+DIV200", f"decoys ({m})", N4 / "a" / m / "debby_cape_div200"))
    for y in (2003, 2005, 2008):
        runs.append((f"B1 ko0_10 vo_{y}", "ko0_10 + twin", N4 / "b1" / "ko0_10" / f"debby_vo_{y}"))
    rows = []
    for name, fam, d in runs:
        j = jobs(d)
        if j is None:
            continue
        cf = sorted(set(ids) & set(j.index) & set(real.index))
        for met in METRICS:
            x = (j.loc[cf, met] - real.loc[cf, met]).astype(float).values
            m, lo, hi = boot_ci(x)
            rows.append(dict(run=name, family=fam, metric=met, n_cfg=len(cf), real_km=float(real.loc[cf, met].mean()),
                             run_km=float(j.loc[cf, met].mean()), null_minus_real_km=m, lo=lo, hi=hi,
                             frac_real_closer=float(np.mean(x > 0)) if len(x) else np.nan))
    S = pd.DataFrame(rows)
    # pooled by family (mean over the runs of a family per config, then CI)
    P = []
    for (fam, met), g in S.groupby(["family", "metric"]):
        xs = []
        for name in g["run"]:
            d = dict((r[0], r[2]) for r in runs)[name]
            j = jobs(d); cf = sorted(set(ids) & set(j.index) & set(real.index))
            xs.append(pd.Series(j.loc[cf, met].values - real.loc[cf, met].values, index=cf))
        x = pd.concat(xs, axis=1).mean(1).values
        m, lo, hi = boot_ci(x)
        P.append(dict(run="(family mean)", family=fam, metric=met, n_cfg=len(x), null_minus_real_km=m, lo=lo, hi=hi,
                      frac_real_closer=float(np.mean(x > 0))))
    S = pd.concat([S, pd.DataFrame(P)])
    S.to_csv(OUT / "d1_path_accuracy.csv", index=False)
    print(S.round(2).to_string(), flush=True)


def track_points():
    ds = xr.open_dataset(IBTRACS)
    nm = np.array([n.decode() if isinstance(n, bytes) else str(n) for n in ds["name"].values])
    season = ds["season"].values
    k = np.where((nm == "DEBBY") & (season == 2006))[0][0]
    t = pd.to_datetime(ds["time"].values[k]); lat = ds["lat"].values[k]; lon = ds["lon"].values[k]
    ok = pd.notna(t) & np.isfinite(lat) & np.isfinite(lon)
    t, lat, lon = t[ok], lat[ok], lon[ok]
    sel = (t <= DATE_END) & (t >= DATE_END - pd.Timedelta(days=15))
    return np.r_[lat[sel], START[0]], np.r_[lon[sel], START[1]]


def d2():
    v = xr.open_dataset(R1 / "inputs_debby" / "vo500700_2006_rebuilt.nc")["vo"]
    v = v.rename({k: n for k, n in (("latitude", "lat"), ("longitude", "lon")) if k in v.dims})
    if float(v.lon.max()) > 180:
        v = v.assign_coords(lon=((v.lon + 180) % 360) - 180).sortby("lon")
    v = v.sortby("lat")
    lat, lon = track_points()
    LA, LO = np.meshgrid(v.lat.values, v.lon.values, indexing="ij")
    reg = np.zeros(LA.shape, bool)
    for a, o in zip(lat, lon):
        reg |= (np.abs(LA - a) <= 5) & (np.abs(LO - o) <= 5)
    tt = pd.to_datetime(v.time.values)
    dt = float(np.median(np.diff(tt.values).astype("timedelta64[m]").astype(float))) / 60.0   # hours
    per_day = int(round(24 / dt))
    X = v.transpose("time", "lat", "lon").values.astype(float)
    win = np.where((tt <= DATE_END) & (tt >= DATE_END - pd.Timedelta(days=15)))[0]

    def pcorr(a, b):
        a = a[reg]; b = b[reg]; ok = np.isfinite(a) & np.isfinite(b)
        a = a[ok] - a[ok].mean(); b = b[ok] - b[ok].mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    rows = []
    for K in (-10, -5, 5, 10):
        cs = [pcorr(X[t], X[t + K * per_day]) for t in win if 0 <= t + K * per_day < len(tt)]
        rows.append(dict(null=f"time shift {K:+d} d", corr_mean=float(np.mean(cs)), corr_sd=float(np.std(cs)), n=len(cs)))
    dlon = float(np.median(np.diff(v.lon.values)))
    for D in (10, 20):
        s = int(round(D / dlon))
        cs = []
        for t in win:
            sh = np.full_like(X[t], np.nan); sh[:, s:] = X[t][:, :-s]
            cs.append(pcorr(X[t], sh))
        rows.append(dict(null=f"space shift +{D} deg east", corr_mean=float(np.mean(cs)), corr_sd=float(np.std(cs)), n=len(cs)))
    curve = []
    for Kh in range(0, 15 * per_day + 1, max(per_day // 4, 1)):
        cs = [pcorr(X[t], X[t + Kh]) for t in win if t + Kh < len(tt)]
        curve.append(dict(lag_days=Kh / per_day, corr_mean=float(np.mean(cs))))
    m = np.nanmean(np.where(reg[None], X, np.nan), axis=(1, 2))
    m = m[np.isfinite(m)]
    m = m - np.polyval(np.polyfit(np.arange(m.size), m, 1), np.arange(m.size))
    f = np.fft.rfftfreq(m.size, d=dt / 24.0); pw = np.abs(np.fft.rfft(m * np.hanning(m.size))) ** 2
    band = (f >= 1 / 10) & (f <= 1 / 2)
    peak = 1.0 / f[band][np.argmax(pw[band])]
    rows.append(dict(null="dominant period region-mean vo (2-10 d band)", corr_mean=float(peak), corr_sd=np.nan, n=m.size))
    pd.DataFrame(rows).to_csv(OUT / "d2_wavetrain.csv", index=False)
    pd.DataFrame(curve).to_csv(OUT / "d2_lagcurve.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(), "\n", pd.DataFrame(curve).round(3).to_string(), flush=True)
    print("region cells", int(reg.sum()), "track points", len(lat), "dt hours", dt, flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d1()
    d2()
