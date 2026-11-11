"""Task H (R2 l240): Fig. 2E-G of the paper recomputed for relaxed completion thresholds.

A T1 configuration passes threshold q if every one of its 3 tracks completed >= q of the 30 backward
steps, i.e. raw_len >= 1 + ceil(30 q) (q = 1.0 -> 31 = trace_st.retention.t1_retained; 0.8 -> 25; 0.5 -> 16).
Errors are computed over the completed part only, step-aligned with the prescribed pathway:
backward step k (centers_raw[k], k = 0 .. raw_len-1) is compared with truth[T-1-k] (T = 40),
which is exactly what 1_SyntheticRuns_Univariate.summarize_track does for complete tracks.
  E  origin distance  = distance at the last completed backward step (k = raw_len-1)
  F  track distance   = mean distance over k = 0 .. raw_len-1
  G  V2 fraction      = mean(parents_raw[:raw_len] == 2)
Per configuration: mean over the 3 tracks (as in 4and2_Fig2.ipynb df_trial). Only jobs with ok == 1.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from trace_st import retention  # noqa: E402

TASK = "H_relaxed_success"
THRESH = [1.0, 0.8, 0.5]
ENGINES = ["granger", "pcmci", "dynotears"]


def truth_tracks():
    spec = importlib.util.spec_from_file_location("t1script", C.REPO / "Scripts_Paper" / "1_SyntheticRuns_Univariate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _, specs, _ = m.build_synthetic_case()
    return {i + 1: np.asarray(s["centers"], float) for i, s in enumerate(specs)}


def track_metrics(tracks, truth):
    rows = []
    for _, r in tracks.iterrows():
        c = np.asarray(r["centers_raw"], float).reshape(-1, 2)
        p = np.asarray(r["parents_raw"], int)
        L = len(c)
        tr = truth[int(r["track_idx"])]
        T = len(tr)
        if L == 0:
            rows.append(dict(origin_cp=np.nan, track_cp=np.nan, v2_cp=np.nan))
            continue
        k = np.arange(L)
        d = np.hypot(*(c - tr[T - 1 - k]).T)
        rows.append(dict(origin_cp=d[-1], track_cp=d.mean(), v2_cp=float(np.mean(p[:L] == 2))))
    return pd.concat([tracks.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def main():
    C.set_style()
    out = C.outdir(TASK)
    d = C.RUNS["t1"]
    tr = pd.read_json(d / "results_tracks.jsonl", lines=True)
    jobs = pd.read_csv(d / "results_jobs_final.csv")
    okjobs = jobs.loc[jobs["ok"] == 1, ["method", "trial_id"]]
    tr = tr.merge(okjobs, on=["method", "trial_id"])
    truth = truth_tracks()
    tm = track_metrics(tr, truth)

    # validation: complete tracks must reproduce the stored metrics exactly
    full = tm["raw_len"] == 31
    val = dict(n_full_tracks=int(full.sum()),
               max_abs_diff_track=float(np.nanmax(np.abs(tm.loc[full, "track_cp"] - tm.loc[full, "mean_distance"]))),
               max_abs_diff_origin=float(np.nanmax(np.abs(tm.loc[full, "origin_cp"] - tm.loc[full, "origin_distance"]))),
               max_abs_diff_v2=float(np.nanmax(np.abs(tm.loc[full, "v2_cp"] - tm.loc[full, "parent2_frac"]))))
    # retention check vs trace_st.retention.t1_retained
    ret100 = retention.t1_retained(tm[["method", "trial_id", "raw_len"]])
    cfg = tm.groupby(["method", "trial_id"]).agg(min_raw_len=("raw_len", "min"), n_tracks=("raw_len", "size"),
                                                origin=("origin_cp", "mean"), track=("track_cp", "mean"),
                                                v2=("v2_cp", "mean"),
                                                origin_stored=("origin_distance", "mean"),
                                                track_stored=("mean_distance", "mean")).reset_index()
    cfg = cfg[cfg["n_tracks"] == 3]
    val["t1_retained_matches_q100"] = bool(
        (cfg.set_index(["method", "trial_id"])["min_raw_len"].eq(31) == ret100.reindex(cfg.set_index(["method", "trial_id"]).index)).all())

    rows, summ = [], []
    for q in THRESH:
        need = 1 + math.ceil(30 * q - 1e-9)
        sel = cfg[cfg["min_raw_len"] >= need].copy()
        sel["threshold"] = q
        rows.append(sel)
        for e in ENGINES:
            s = sel[sel["method"] == e]
            n_all = int((cfg["method"] == e).sum())
            summ.append(dict(threshold=f"{int(q*100)}%", min_raw_len=need, engine=C.ENGINE_LABELS[e],
                             n_pass=len(s), n_ok_configs=n_all, frac_pass=len(s) / n_all,
                             origin_median=s["origin"].median(), origin_mean=s["origin"].mean(),
                             track_median=s["track"].median(), track_mean=s["track"].mean(),
                             v2_median=s["v2"].median(), v2_mean=s["v2"].mean()))
    allsel = pd.concat(rows)
    allsel.to_csv(out / "H_config_metrics_by_threshold.csv", index=False)
    S = pd.DataFrame(summ)
    S.to_csv(out / "H_counts_and_medians.csv", index=False)
    (out / "H_counts_and_medians.md").write_text(C.to_markdown(S))
    json.dump(val, open(out / "H_validation.json", "w"), indent=1)
    print(json.dumps(val, indent=1))
    print(S.round(3).to_string())

    # ---------------- figure: rows = threshold, cols = E/F/G
    specs = [("origin", "E. Origin distance", "Distance (grid units)"),
             ("track", "F. Mean track distance", "Distance (grid units)"),
             ("v2", "G. Fraction of $V_2$ steps", "Fraction of steps")]
    fig, axes = plt.subplots(3, 3, figsize=(19.2, 10.8), sharex="col")
    for i, q in enumerate(THRESH):
        sel = allsel[allsel["threshold"] == q]
        for jcol, (col, title, xl) in enumerate(specs):
            ax = axes[i, jcol]
            v_all = allsel[col].dropna()
            bins = np.linspace(0, 1, 26) if col == "v2" else np.linspace(0, np.nanpercentile(v_all, 99.5), 26)
            for e in ENGINES:
                v = sel.loc[sel["method"] == e, col].dropna()
                if len(v) == 0:
                    continue
                ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", lw=2.4,
                        color=C.ENGINE_COLORS[e], label=f"{C.ENGINE_LABELS[e]} (n={len(v)})")
                ax.axvline(np.median(v), color=C.ENGINE_COLORS[e], lw=1.6, ls="--")
            if i == 0:
                ax.set_title(title, loc="left", fontweight="bold")
            if i == 2:
                ax.set_xlabel(xl)
            if jcol == 0:
                ax.set_ylabel(f"≥{int(q*100)}% complete\nConfigurations")
            ax.legend(fontsize=12, loc="upper right", frameon=True)
    fig.suptitle("Relaxing completion to 80 % or 50 % adds configurations but barely moves the error medians",
                 fontsize=20, y=1.0)
    fig.text(0.5, -0.01, "Errors over the completed part of each track only; dashed lines = medians; "
             "rows: configurations whose 3 tracks all completed ≥ 100 %, 80 %, 50 % of the 30 steps.",
             ha="center", fontsize=14)
    fig.tight_layout()
    print(C.savefig(fig, TASK, "fig2EFG_relaxed_thresholds"))


if __name__ == "__main__":
    main()
