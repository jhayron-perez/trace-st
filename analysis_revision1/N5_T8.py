"""Night-5 T8: refit table, T1 and T2 x Elastic-Net / PCMCI / DYNOTEARS x own vs refit coefficients (+ timing).

Sources (existing runs; night 1 T1r/T2r and T9 refit; the probability rule was drawn in those runs, so every cell
is reported for all configs and for the prob_rule == "linear" subset, the night-5 method):
  T1 own   : T1r_univariate (3 engines x 1000)        T1 refit : T9_refit/T1_refit (3 engines x 1000)
  T2-EN    : own T9_refit/T2_en_native_a06, refit T9_refit/T2_en_refit_a06 (50 configs, alpha 0.6, M 30);
             night-5 p03 (linear, alpha 0.2/0.6, M 20) as the linear own reference
  T2-PCMCI : own T2r_multivariate_pcmci, refit T9_refit/T2_pcmci_refit (30 configs x 4 alpha, M 20)
  T2-DYNOTEARS: no runs (night-1 timing: about 2 h per trajectory at the T2 geometry) -> missing.
Metrics: T1 = retained fraction (t1_retained), median mean track distance (score_mean_distance) and parent-2
(wrong variable) fraction; T2 = slope / r of median recovered alpha vs prescribed, median |alpha err|;
timing = median elapsed_sec per config.

python N5_T8.py   (PBS only) -> night5/analysis/T8_refit_table.csv / .md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = C.R1
OUT = R1 / "night5" / "analysis"


def t1_cells(d, mode):
    j = pd.read_csv(d / "results_jobs_final.csv")
    tr = pd.read_csv(d / "results_tracks_final.csv", usecols=["method", "trial_id", "raw_len"])
    ok = retention.t1_retained(tr)
    rows = []
    for m in ("granger", "pcmci", "dynotears"):
        for sub in ("all", "linear"):
            g = j[(j["method"] == m) & ((j["prob_rule"] == "linear") if sub == "linear" else True)]
            if not len(g):
                continue
            keep = ok.reindex(list(zip(g["method"], g["trial_id"]))).fillna(False).to_numpy()
            gk = g[keep]
            rows.append(dict(case="T1", engine=m, strength=mode, subset=sub, n_configs=len(g),
                             retained_frac=float(keep.mean()), median_track_dist=float(gk["score_mean_distance"].median()),
                             mean_parent2_frac=float(gk["score_parent2_frac"].mean()),
                             median_sec_per_config=float(g["elapsed_sec"].median()), source=str(d)))
    return rows


def t2_cells(d, engine, mode):
    f = d / "results_alphas_final.csv"
    if not f.exists():
        f = d / "results_alphas_snapshot.csv"
    x = pd.read_csv(f)
    jf = d / "results_jobs_final.csv"
    j = pd.read_csv(jf) if jf.exists() else None
    rows = []
    for sub in ("all", "linear"):
        g = x[(x["prob_rule"] == "linear") if sub == "linear" else np.ones(len(x), bool)]
        g = g[g["estimated_alpha_density"].notna()]
        if g["trial_id"].nunique() < 3:
            continue
        med = g.groupby("alpha_mix")["estimated_alpha_density"].median()
        a = med.index.values.astype(float)
        slope, r = (np.polyfit(a, med.values, 1)[0], np.corrcoef(a, med.values)[0, 1]) if len(a) > 1 else (np.nan, np.nan)
        row = dict(case="T2", engine=engine, strength=mode, subset=sub, n_configs=int(g["trial_id"].nunique()),
                   alphas=",".join(f"{v:g}" for v in a), slope=float(slope), r_median=float(r),
                   median_abs_alpha_err=float((g["estimated_alpha_density"] - g["alpha_mix"]).abs().median()), source=str(d))
        if j is not None and "elapsed_sec" in j:
            jj = j[j["trial_id"].isin(g["trial_id"].unique())]
            row["median_sec_per_config"] = float(jj["elapsed_sec"].median())
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = t1_cells(R1 / "T1r_univariate", "own") + t1_cells(R1 / "T9_refit" / "T1_refit", "refit")
    rows += t2_cells(R1 / "T9_refit" / "T2_en_native_a06", "granger", "own")
    rows += t2_cells(R1 / "T9_refit" / "T2_en_refit_a06", "granger", "refit")
    p03 = R1 / "night5" / "t3" / "p03"
    if (p03 / "results_alphas_final.csv").exists() or (p03 / "results_alphas_snapshot.csv").exists():
        rows += [dict(r, strength="own (night-5 p03, linear)") for r in t2_cells(p03, "granger", "own")]
    rf = R1 / "night5" / "t8" / "T2en_refit_linear"
    if (rf / "results_alphas_final.csv").exists():
        rows += [dict(r, strength="refit (night-5, linear, same settings as p03)") for r in t2_cells(rf, "granger", "refit")]
    rows += t2_cells(R1 / "T2r_multivariate_pcmci", "pcmci", "own")
    rows += t2_cells(R1 / "T9_refit" / "T2_pcmci_refit", "pcmci", "refit")
    for mode in ("own", "refit"):
        rows.append(dict(case="T2", engine="dynotears", strength=mode, subset="-", n_configs=0,
                         source="MISSING: no T2-DYNOTEARS runs; ~2 h per trajectory at the T2 geometry (night-1 T3r "
                                "8129551/8135293), i.e. far above the 500 core-h allowance for missing cells"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "T8_refit_table.csv", index=False)
    cols = [c for c in ["case", "engine", "strength", "subset", "n_configs", "retained_frac", "median_track_dist",
                        "mean_parent2_frac", "alphas", "slope", "r_median", "median_abs_alpha_err",
                        "median_sec_per_config"] if c in df]
    (OUT / "T8_refit_table.md").write_text(C.to_markdown(df[cols], "{:.3g}"))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df[cols].round(3).to_string())


if __name__ == "__main__":
    main()
