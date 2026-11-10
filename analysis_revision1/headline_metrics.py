"""Headline numbers for the comparison sheet (task 9c), from run outputs only.

python headline_metrics.py --debby DIR --pinatubo DIR --pnw DIR  (any subset)
Debby: retained (trace_st.retention), mean track distance = mean of member track_dist_km_mean
       pooled over members of retained trials (paper 825.4 km); density shares in the stored
       density_range window (96-360 h) pooled over retained trials, and mean of trial fractions.
Pinatubo: retained, mean over retained trials of mean_final_dist_to_volcano_km (paper 2135.9 km);
       density_range shares (window stored in density_range_start/end_day).
PNW: kept runs, mean over kept runs of window_stats[days_1_10].integrated_density_by_var_norm.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_st import retention  # noqa: E402


def shares(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix + "::")]
    v = df[cols].astype(float)
    names = [c.split("::", 1)[1] for c in cols]
    pooled = v.sum() / v.sum().sum()
    trial = v.div(v.sum(axis=1), axis=0).mean()
    return {n: dict(pooled=round(float(pooled[c]), 4), mean_of_trials=round(float(trial[c]), 4)) for n, c in zip(names, cols)}


def debby(d):
    j = pd.read_csv(f"{d}/results_jobs_final.csv"); m = pd.read_csv(f"{d}/results_members_final.csv")
    k = j[retention.debby_retained(j)]
    mm = m[m["trial_id"].isin(k["trial_id"]) & (m["ok"] == 1)]
    return dict(n_trials=len(j), retained=len(k), mean_track_dist_km=round(float(mm["track_dist_km_mean"].mean()), 1),
                window_h=[float(k["density_range_start_hour"].iloc[0]), float(k["density_range_end_hour"].iloc[0])],
                density_range_share=shares(k, "density_range"))


def pinatubo(d):
    j = pd.read_csv(f"{d}/results_jobs_final.csv"); k = j[retention.pinatubo_retained(j)]
    return dict(n_trials=len(j), retained=len(k), mean_endpoint_dist_km=round(float(k["mean_final_dist_to_volcano_km"].mean()), 1),
                window_day=[float(k["density_range_start_day"].iloc[0]), float(k["density_range_end_day"].iloc[0])],
                density_range_share=shares(k, "density_range"))


def pnw(d):
    s = pd.read_csv(f"{d}/hyperparam_search_summary.csv"); k = s[retention.pnw_retained(s)]
    out = {}
    for w in ("days_1_10", "days_11_20", "days_21_30"):
        rows = [json.load(open(f"{d}/run_{int(r):04d}/summary.json"))["window_stats"][w]["integrated_density_by_var_norm"] for r in k["run_id"]]
        out[w] = {v: round(float(np.nanmean([r[v] for r in rows])), 4) for v in rows[0]}
    return dict(n_runs=len(s), kept=len(k), mean_norm_density=out)


ap = argparse.ArgumentParser(); ap.add_argument("--debby"); ap.add_argument("--pinatubo"); ap.add_argument("--pnw"); ap.add_argument("--out")
a = ap.parse_args(); res = {}
if a.debby: res["debby"] = debby(a.debby)
if a.pinatubo: res["pinatubo"] = pinatubo(a.pinatubo)
if a.pnw: res["pnw"] = pnw(a.pnw)
print(json.dumps(res, indent=1))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
