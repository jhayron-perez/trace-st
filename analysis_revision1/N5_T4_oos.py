"""Night-5 T4 out-of-sample check: scale parameters from the rules alone (T1 r 3, box 30, W 2 d; T2 r 2, box 27.3,
W 2 d), other parameters drawn as in the paper, linear rule; vs the paper ensembles on the same trial ids.

T1: retained fraction (t1_retained), median mean track distance (score_mean_distance) and wrong-variable fraction
    (score_parent2_frac); reference = T1r_univariate (night-1, paper scale draws) same trial ids, all and linear subset.
T2: median |alpha err| (estimated_alpha_density), r and slope of the median recovered alpha; reference = T2r
    (paper draws, all 4 alphas) restricted to alpha 0.2/0.6, same trial ids, linear subset, and night-5 p03 (50 ids).
python N5_T4_oos.py  (PBS only) -> night5/analysis/T4/T4_oos.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from A4_twins import boot_ci  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = C.R1
N5 = R1 / "night5"
OUT = N5 / "analysis" / "T4"


def t1(d, ids, label, linear_only=False):
    j = pd.read_csv(d / "results_jobs_final.csv")
    j = j[(j["method"] == "granger") & j["trial_id"].isin(ids)]
    if linear_only:
        j = j[j["prob_rule"] == "linear"]
    tr = pd.read_csv(d / "results_tracks_final.csv", usecols=["method", "trial_id", "raw_len"])
    ok = retention.t1_retained(tr).reindex(list(zip(j["method"], j["trial_id"]))).fillna(False).to_numpy()
    g = j[ok]
    m, lo, hi = boot_ci(g["score_mean_distance"].to_numpy())
    return dict(case="T1", run=label, n=len(j), retained_frac=float(ok.mean()),
                median_track_dist=float(g["score_mean_distance"].median()), mean_track_dist=m, lo=lo, hi=hi,
                parent2_frac=float(g["score_parent2_frac"].mean()))


def t2(d, ids, label, linear_only=False):
    f = d / "results_alphas_final.csv"
    x = pd.read_csv(f if f.exists() else d / "results_alphas_snapshot.csv")
    x = x[x["trial_id"].isin(ids) & x["alpha_mix"].round(3).isin([0.2, 0.6]) & x["estimated_alpha_density"].notna()]
    if linear_only:
        x = x[x["prob_rule"] == "linear"]
    e = (x["estimated_alpha_density"] - x["alpha_mix"]).abs()
    pc = e.groupby(x["trial_id"]).mean().to_numpy()
    m, lo, hi = boot_ci(pc)
    med = x.groupby("alpha_mix")["estimated_alpha_density"].median()
    r = dict(case="T2", run=label, n=int(x["trial_id"].nunique()), median_abs_alpha_err=float(e.median()),
             mean_abs_alpha_err=m, lo=lo, hi=hi, r_pooled=float(np.corrcoef(x["alpha_mix"], x["estimated_alpha_density"])[0, 1]),
             valid_frac=float(x["is_valid_physical"].mean()) if "is_valid_physical" in x else np.nan)
    for a, v in med.items():
        r[f"median_alpha_{a:g}"] = float(v)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids1 = list(range(100)); ids2 = list(range(100))
    rows = [t1(N5 / "t4" / "T1_oos", ids1, "rules (night 5, linear)"),
            t1(R1 / "T1r_univariate", ids1, "paper draws (night 1), all"),
            t1(R1 / "T1r_univariate", ids1, "paper draws (night 1), linear", True),
            t2(N5 / "t4" / "T2_oos", ids2, "rules (night 5, linear)"),
            t2(R1 / "T2r_multivariate_en", ids2, "paper draws (night 1), linear", True),
            t2(N5 / "t3" / "p03", ids2, "paper draws (night 5 p03, linear, ids 0-49)")]
    rows.append(t2(N5 / "t4" / "T2_oos", list(range(50)), "rules (night 5, linear), ids 0-49"))
    df = pd.DataFrame(rows); df.to_csv(OUT / "T4_oos.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.round(3).to_string())


if __name__ == "__main__":
    main()
