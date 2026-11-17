"""Night-5 T7: supplementary figures redrawn with the FINAL method (linear probability rule only, harmonized
retention for the real cases), in the figures_v5 style (N5_style). Existing scripts are imported, not edited.

  --which E  winner-grouping options (night-2 EG_summary E). T1: E_grouping/T1_<g>, EN, first 200 configs of the
             paper draw; restricted to prob_rule == linear; completion = trace_st.retention.t1_retained; errors over
             linear AND t1_retained configs. T2-EN: E_grouping/T2_<g> (50 configs x alpha 0.2/0.6), linear only
             (no retention, as night 2).                                  -> figures_v5/S_DBSCAN_grouping
  --which G  longer N (night-2 EG_summary G). G_T1_N40, G_T1_N49 (EN, the 993 T1-retained configs), linear only.
                                                                          -> figures_v5/S_longer_N
  --which H  relaxed success (night-2 H_relaxed_success). T1r_univariate, ok jobs, prob_rule == linear; the 100 %
             threshold = t1_retained.                                     -> figures_v5/S_relaxed_success
  --which F  exact enumeration vs Monte Carlo: REDRAW ONLY from analysis_v2/F (night-2 job 8147105 already forced
             prob_rule = linear on trials 0-4; no recompute).             -> figures_v5/S_exact_vs_MC
  --which K  linearity check: K_linearity_check.analyse_step re-run on 100 steps per case (seed 0) sampled from
             linear AND harmonized-retained configs (real cases); T2-EN reference = T2r_multivariate_en linear configs.
                                                                          -> figures_v5/S_linearity
  --which J  existing J robustness table (headline holds in what share of configs), linear AND harmonized, from the
             analysis_v2/J_sensitivity per-config tables (filter + replot; the 15 partial-dependence panels are not
             redrawn).                                                    -> figures_v5/S_sensitivity_existing
  --source night1|night5 (or env TST_T7_SOURCE; default night1): only K uses it (real-case run dirs). night5 =
             revision1/night5/t2/{debby_orig/part*, pin_orig, pnw_orig} (all forced linear). E, G, H, F have no
             night-5 re-run, so they always read the night-1/2 outputs.
Tables: revision1/night5/analysis/T7_supp/<X>/ (K night5: T7_supp/K_night5/).

python N5_T7_supp.py --which E G H F K J [--source night1] [--nproc 16]     (PBS develop only)
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import N5_style as S  # noqa: E402  (sets Agg)
import rev1_common as C  # noqa: E402
from trace_st import retention  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R1 = C.R1
OUT = R1 / "night5" / "analysis" / "T7_supp"
V2 = R1 / "analysis_v2"
GROUPINGS = ["dbscan", "strongest", "variable_sign", "sectors8", "kmeans"]
GCOL = dict(zip(GROUPINGS, ["#000000", "#E69F00", "#56B4E9", "#CC79A7", "#D55E00"]))
GLAB = {"dbscan": "DBSCAN (paper)", "strongest": "Strongest", "variable_sign": "Variable + sign",
        "sectors8": "8 sectors", "kmeans": "k-means"}
ENGINES = ["granger", "pcmci", "dynotears"]
ENG_COL = {"granger": "#0072B2", "pcmci": "#CC79A7", "dynotears": "#E69F00"}
ENG_LAB = {"granger": "Elastic-Net", "pcmci": "PCMCI", "dynotears": "DYNOTEARS"}


def outdir(x):
    d = OUT / x
    d.mkdir(parents=True, exist_ok=True)
    return d


def md(df, path):
    Path(path).write_text(C.to_markdown(df))


# ============================================================== E
def run_E():
    from EG_summary import load_tracks
    out = outdir("E")
    rows = []
    for g in GROUPINGS:
        t = load_tracks(R1 / "E_grouping" / f"T1_{g}")
        if t is None:
            continue
        t = t[t["method"] == "granger"]
        rule = t.groupby("trial_id")["prob_rule"].first()
        ret = retention.t1_retained(t).droplevel(0)  # index trial_id
        per = t.groupby("trial_id")[["mean_distance", "origin_distance", "parent2_frac"]].mean()
        for subset in ("all", "linear"):
            ids = rule.index if subset == "all" else rule.index[rule == "linear"]
            r_ids = [i for i in ids if ret.get(i, False)]
            p, pr = per.loc[ids], per.loc[r_ids]
            rows.append(dict(grouping=g, subset=subset, n_cfg=len(ids), n_retained=len(r_ids),
                             completion=len(r_ids) / len(ids),
                             track_err_median_retained=pr["mean_distance"].median(),
                             origin_err_median_retained=pr["origin_distance"].median(),
                             wrong_var_frac_mean_retained=pr["parent2_frac"].mean(),
                             track_err_median_allcfg=p["mean_distance"].median(),
                             origin_err_median_allcfg=p["origin_distance"].median(),
                             wrong_var_frac_mean_allcfg=p["parent2_frac"].mean()))
    E1 = pd.DataFrame(rows)
    rows = []
    for g in GROUPINGS:
        f = R1 / "E_grouping" / f"T2_{g}" / "results_alphas_final.csv"
        if not f.exists():
            f = f.with_name("results_alphas_snapshot.csv")
        if not f.exists():
            continue
        x = pd.read_csv(f)
        x["abs_alpha_err"] = (x["estimated_alpha_density"] - x["alpha_mix"]).abs()
        for subset in ("all", "linear"):
            xs = x if subset == "all" else x[x["prob_rule"] == "linear"]
            for a, gg in xs.groupby("alpha_mix"):
                rows.append(dict(grouping=g, subset=subset, alpha=a, n_cfg=len(gg),
                                 abs_alpha_err_median=gg["abs_alpha_err"].median(),
                                 valid_physical=gg["is_valid_physical"].mean(), ok_frac=gg["valid_ok_frac"].mean()))
    E2 = pd.DataFrame(rows)
    E1.to_csv(out / "E_T1_grouping_linear_vs_all.csv", index=False)
    E2.to_csv(out / "E_T2_grouping_linear_vs_all.csv", index=False)
    md(E1.round(4), out / "E_T1_grouping_linear_vs_all.md"); md(E2.round(4), out / "E_T2_grouping_linear_vs_all.md")
    print(E1.round(3).to_string(), "\n", E2.round(3).to_string(), flush=True)

    S.set_style()
    fig, axs = plt.subplots(2, 3, figsize=(S.DOUBLE, 11 * S.CM))
    axs = axs.ravel()
    L = E1[E1.subset == "linear"].set_index("grouping").reindex(GROUPINGS)
    A = E1[E1.subset == "all"].set_index("grouping").reindex(GROUPINGS)
    x = np.arange(len(GROUPINGS))
    n1 = int(L['n_cfg'].iloc[0])
    specs = [("track_err_median_retained", "Track error, median\n(grid cells)", "T1: track error"),
             ("origin_err_median_retained", "Origin error, median\n(grid cells)", "T1: origin error"),
             ("wrong_var_frac_mean_retained", "Steps on $V_2$, mean\n(fraction)", "T1: wrong-variable steps"),
             ("completion", "Configs with all 3 tracks\nat 30 steps (fraction)", "T1: completion")]
    for k, (c, yl, tt) in enumerate(specs):
        ax = axs[k]
        ax.bar(x, L[c], color=[GCOL[g] for g in GROUPINGS], width=0.7)
        ax.plot(x, A[c], "D", mfc="white", mec="#555555", ms=3.2, mew=0.8, ls="none")
        ax.set_ylabel(yl); ax.set_title(tt)
    axs[3].set_ylim(0, 1.05)
    for k, a in enumerate([0.2, 0.6]):
        ax = axs[4 + k]
        g2 = E2[(E2.subset == "linear") & np.isclose(E2.alpha, a)].set_index("grouping").reindex(GROUPINGS)
        g2a = E2[(E2.subset == "all") & np.isclose(E2.alpha, a)].set_index("grouping").reindex(GROUPINGS)
        n2 = int(g2['n_cfg'].iloc[0])
        ax.bar(x, g2["abs_alpha_err_median"], color=[GCOL[g] for g in GROUPINGS], width=0.7)
        ax.plot(x, g2a["abs_alpha_err_median"], "D", mfc="white", mec="#555555", ms=3.2, mew=0.8, ls="none")
        ax.set_ylabel("|recovered − true α|,\nmedian (unitless)")
        ax.set_title(fr"T2-EN: true $\alpha$ = {a}")
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels([GLAB[g] for g in GROUPINGS], rotation=35, ha="right",
                                             rotation_mode="anchor", fontsize=6)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hd = [Patch(color="#777777", label=f"linear rule only (T1 n = {n1} configs, T2-EN n = {n2})"),
          Line2D([], [], marker="D", ls="none", mfc="white", mec="#555555", ms=3.2,
                 label="all configs, both rules (night-2 design: T1 n = 200, T2-EN n = 50)")]
    fig.legend(handles=hd, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=6)
    S.panel_labels(axs, x=-0.3, y=1.06)
    fig.tight_layout(h_pad=1.0, w_pad=1.6, rect=(0, 0, 1, 0.96))
    print(S.save(fig, "S_DBSCAN_grouping", "S_DBSCAN_grouping"), flush=True)


# ============================================================== G
def run_G():
    import importlib.util
    out = outdir("G")
    spec = importlib.util.spec_from_file_location("t1s", C.REPO / "Scripts_Paper" / "1_SyntheticRuns_Univariate.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    _, specs, _ = m.build_synthetic_case()
    origins = {i + 1: np.asarray(s["centers"], float)[0] for i, s in enumerate(specs)}
    prof = []
    ncfg = {}
    for N in (40, 49):
        t = pd.read_csv(R1 / f"G_T1_N{N}" / "results_tracks_final.csv")
        ncfg[(N, "all")] = t.trial_id.nunique()
        ncfg[(N, "linear")] = t.loc[t.prob_rule == "linear", "trial_id"].nunique()
        for r in t.itertuples():
            c = np.asarray(ast.literal_eval(r.centers_raw), float)
            p = np.asarray(ast.literal_eval(r.parents_raw), int)
            o = origins[int(r.track_idx)]
            for k in range(1, len(c)):
                prof.append((N, r.trial_id, r.prob_rule, k, float(np.hypot(*(c[k] - c[k - 1]))),
                             float(np.hypot(*(c[k] - o))), int(p[k]) if k < len(p) else -1))
    P = pd.DataFrame(prof, columns=["N", "trial_id", "prob_rule", "step", "displacement", "dist_to_origin", "parent"])
    P["block"] = np.where(P["step"] <= 30, "steps 1-30", np.where(P["step"] <= 39, "steps 31-39", "steps 40+"))
    prof_t, blk = [], []
    for subset in ("all", "linear"):
        Q = P if subset == "all" else P[P.prob_rule == "linear"]
        G = Q.groupby(["N", "step"]).agg(disp_mean=("displacement", "mean"), dist_origin_median=("dist_to_origin", "median"),
                                         dist_origin_std=("dist_to_origin", "std"),
                                         frac_parent_v1=("parent", lambda x: float((x == 1).mean())),
                                         frac_parent_v2=("parent", lambda x: float((x == 2).mean())),
                                         n=("parent", "size")).reset_index()
        G["subset"] = subset; prof_t.append(G)
        B = Q.groupby(["N", "block"]).agg(disp_mean=("displacement", "mean"), dist_origin_median=("dist_to_origin", "median"),
                                          dist_origin_std=("dist_to_origin", "std"),
                                          frac_v1=("parent", lambda x: float((x == 1).mean()))).reset_index()
        B["subset"] = subset
        B["n_cfg"] = [ncfg[(n_, subset)] for n_ in B["N"]]
        blk.append(B)
    G = pd.concat(prof_t); B = pd.concat(blk)
    G.to_csv(out / "G_profiles_linear_vs_all.csv", index=False)
    B.to_csv(out / "G_blocks_linear_vs_all.csv", index=False)
    md(B.round(3), out / "G_blocks_linear_vs_all.md")
    print(B.round(3).to_string(), flush=True)

    S.set_style()
    GL = G[G.subset == "linear"]
    fig, ax = plt.subplots(1, 3, figsize=(S.DOUBLE, 5.8 * S.CM))
    for N, g in GL.groupby("N"):
        ls = "-" if N == 40 else "--"
        ax[0].plot(g["step"], g["disp_mean"], ls, color="#0072B2", label=f"N = {N}")
        ax[1].plot(g["step"], g["dist_origin_median"], ls, color="#D55E00", label=f"N = {N}")
        ax[1].fill_between(g["step"], g["dist_origin_median"] - g["dist_origin_std"],
                           g["dist_origin_median"] + g["dist_origin_std"], color="#D55E00", alpha=0.12, lw=0)
        ax[2].plot(g["step"], g["frac_parent_v1"], ls, color="#000000", label=f"$V_1$ (true parent), N = {N}")
        ax[2].plot(g["step"], g["frac_parent_v2"], ls, color="#999999", label=f"$V_2$, N = {N}")
    for a_ in ax:
        a_.axvline(30, color="#999999", lw=0.7, ls=":")
        a_.axvline(39, color="#CC79A7", lw=0.9, ls=":")
        a_.set_xlabel("Backward tracing step (time steps)")
    ax[1].text(39.5, 0.97, "true origin", color="#CC79A7", fontsize=6, transform=ax[1].get_xaxis_transform(), va="top")
    ax[0].text(29.5, 0.55, "paper N = 30", color="#777777", fontsize=6, transform=ax[0].get_xaxis_transform(),
               va="top", ha="right", rotation=90)
    ax[0].set_ylabel("Mean step displacement (grid cells)")
    ax[1].set_ylabel("Distance to true origin,\nmedian ± sd (grid cells)")
    ax[2].set_ylabel("Fraction of steps by parent")
    ax[0].legend(loc="upper left"); ax[2].legend(loc="lower left", fontsize=5.3)
    n40 = ncfg[(40, "linear")]
    print(f"G linear configs: N40 {n40}", flush=True)
    S.panel_labels(ax, x=-0.24)
    fig.tight_layout(w_pad=1.2)
    print(S.save(fig, "S_longer_N", "S_longer_N"), flush=True)


# ============================================================== H
def run_H():
    import H_relaxed_success as H
    out = outdir("H")
    d = C.RUNS["t1"]
    tr = pd.read_json(d / "results_tracks.jsonl", lines=True)
    jobs = pd.read_csv(d / "results_jobs_final.csv")
    okjobs = jobs.loc[jobs["ok"] == 1, ["method", "trial_id", "prob_rule"]]
    tr = tr.drop(columns=[c for c in ("prob_rule",) if c in tr.columns]).merge(okjobs, on=["method", "trial_id"])
    tm = H.track_metrics(tr, H.truth_tracks())
    cfg = tm.groupby(["method", "trial_id"]).agg(min_raw_len=("raw_len", "min"), n_tracks=("raw_len", "size"),
                                                origin=("origin_cp", "mean"), track=("track_cp", "mean"),
                                                v2=("v2_cp", "mean"), prob_rule=("prob_rule", "first")).reset_index()
    cfg = cfg[cfg["n_tracks"] == 3]
    ret100 = retention.t1_retained(tm[["method", "trial_id", "raw_len"]])
    chk = bool((cfg.set_index(["method", "trial_id"])["min_raw_len"].eq(31)
                == ret100.reindex(cfg.set_index(["method", "trial_id"]).index)).all())
    rows, summ = [], []
    for subset in ("all", "linear"):
        cs = cfg if subset == "all" else cfg[cfg.prob_rule == "linear"]
        for q in H.THRESH:
            need = 1 + math.ceil(30 * q - 1e-9)
            sel = cs[cs["min_raw_len"] >= need].copy()
            sel["threshold"] = q; sel["subset"] = subset
            rows.append(sel)
            for e in ENGINES:
                s = sel[sel["method"] == e]
                n_all = int((cs["method"] == e).sum())
                summ.append(dict(subset=subset, threshold=f"{int(q*100)}%", min_raw_len=need, engine=ENG_LAB[e],
                                 n_pass=len(s), n_ok_configs=n_all, frac_pass=len(s) / n_all,
                                 origin_median=s["origin"].median(), track_median=s["track"].median(),
                                 v2_median=s["v2"].median(), origin_mean=s["origin"].mean(),
                                 track_mean=s["track"].mean(), v2_mean=s["v2"].mean()))
    allsel = pd.concat(rows)
    allsel.to_csv(out / "H_config_metrics_by_threshold_linear_vs_all.csv.gz", index=False)
    Sm = pd.DataFrame(summ)
    Sm.to_csv(out / "H_counts_and_medians_linear_vs_all.csv", index=False)
    md(Sm.round(3), out / "H_counts_and_medians_linear_vs_all.md")
    json.dump(dict(t1_retained_matches_q100=chk), open(out / "H_validation.json", "w"))
    print("t1_retained matches q=100%:", chk); print(Sm.round(3).to_string(), flush=True)

    S.set_style()
    lin = allsel[allsel.subset == "linear"]
    specs = [("origin", "Origin distance (grid cells)"), ("track", "Mean track distance (grid cells)"),
             ("v2", "Fraction of steps on $V_2$")]
    fig, axes = plt.subplots(3, 3, figsize=(S.DOUBLE, 13 * S.CM), sharex="col")
    for i, q in enumerate(H.THRESH):
        sel = lin[lin["threshold"] == q]
        for j, (col, xl) in enumerate(specs):
            ax = axes[i, j]
            v_all = lin[col].dropna()
            bins = np.linspace(0, 1, 26) if col == "v2" else np.linspace(0, np.nanpercentile(v_all, 99.5), 26)
            for e in ENGINES:
                v = sel.loc[sel["method"] == e, col].dropna()
                if len(v) == 0:
                    continue
                ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, histtype="step", lw=1.0, color=ENG_COL[e],
                        label=f"{ENG_LAB[e]} (n = {len(v)})")
                ax.axvline(np.median(v), color=ENG_COL[e], lw=0.8, ls="--")
            if i == 2:
                ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel(f"≥ {int(q*100)} % complete\nConfigurations (count)")
            ax.legend(loc="upper right", fontsize=5.3)
    for j, t in enumerate(["Origin distance", "Mean track distance", "Fraction of $V_2$ steps"]):
        axes[0, j].set_title(t)
    S.panel_labels(axes.ravel(), x=-0.18)
    fig.tight_layout(h_pad=0.8, w_pad=1.0)
    print(S.save(fig, "S_relaxed_success", "S_relaxed_success"), flush=True)


# ============================================================== F (redraw only)
def run_F():
    out = outdir("F")
    Fs = pd.read_csv(V2 / "F" / "F_summary.csv")
    cfgs = pd.read_csv(V2 / "F" / "F_configs.csv")
    Fs.to_csv(out / "F_summary_copy_from_analysis_v2.csv", index=False)
    cfgs.to_csv(out / "F_configs_copy_from_analysis_v2.csv", index=False)
    (out / "README.txt").write_text("Redraw only. Source: analysis_v2/F (night-2 job 8147105, F_exact_vs_mc.py, "
                                    "prob_rule forced to linear for trials 0-4; used_prob_rule column). No recompute.\n")
    assert (cfgs["used_prob_rule"] == "linear").all()
    S.set_style()
    colors = {"tv_var": "#0072B2", "tv_joint": "#D55E00", "tv_density": "#CC79A7"}
    labels = {"tv_var": "Variable shares", "tv_joint": "Joint (variable, exact location)",
              "tv_density": "Box density (T2 metric)"}
    mk = {"tv_var": "o", "tv_joint": "s", "tv_density": "^"}
    fig, ax = plt.subplots(figsize=(S.SINGLE, 7 * S.CM))
    for met in ["tv_var", "tv_joint", "tv_density"]:
        s = Fs[Fs.metric == met]
        ax.fill_between(s.M, s["min"], s["max"], color=colors[met], alpha=0.18, lw=0)
        ax.plot(s.M, s["median"], marker=mk[met], ms=3.5, color=colors[met],
                label=f"{labels[met]} (slope {s.slope_median.iloc[0]:.2f})")
    Ms = np.array([10, 300])
    ref0 = Fs[(Fs.metric == "tv_var") & (Fs.M == 10)]["median"].iloc[0]
    ax.plot(Ms, ref0 * np.sqrt(10 / Ms), "k--", lw=0.8, label=r"$1/\sqrt{M}$ reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([10, 30, 100, 300]); ax.set_xticklabels(["10", "30", "100", "300"]); ax.minorticks_off()
    ax.set_yticks([0.03, 0.1, 0.3, 1.0]); ax.set_yticklabels(["0.03", "0.1", "0.3", "1"])
    ax.set_xlabel("Ensemble size M (members)")
    ax.set_ylabel("Total-variation distance to exact (unitless)")
    ax.legend(loc="lower left", fontsize=5.5)
    ax.set_title(r"T2-EN, $\alpha$ = 0.6, N = 4, linear rule, 5 configs", fontsize=7)
    fig.tight_layout()
    print(S.save(fig, "S_exact_vs_MC", "S_exact_vs_MC"), flush=True)


# ============================================================== K
def k_real_dirs(source):
    if source == "night5":
        b = R1 / "night5" / "t2"
        return dict(debby=sorted(b.glob("debby_orig/part*")) or [b / "debby_orig"], pinatubo=[b / "pin_orig"],
                    pnw=[b / "pnw_orig"])
    return dict(debby=[R1 / "T4r_debby_vorebuilt"], pinatubo=[R1 / "T5r_pinatubo"], pnw=[R1 / "T6r_pnw"])


def k_pool(case, source):
    """Linear AND harmonized-retained configs. Returns list of (config_id, loader-info) and counts."""
    if case == "t2":
        df = pd.read_csv(R1 / "T2r_multivariate_en/results_alphas_final.csv")
        n_tot = df.trial_id.nunique()
        df = df[(df.n_ok > 0) & (df.prob_rule == "linear")]
        pool = [(int(t), float(a), None) for t, a in zip(df.trial_id, df.alpha_mix)]
        return pool, dict(case=case, n_total=n_tot, n_linear=int(df.trial_id.nunique()), n_pool=int(df.trial_id.nunique()),
                          retention="none (T2 synthetic; n_ok > 0 as night 2)", pool_rows=len(pool))
    dirs = k_real_dirs(source)[case]
    if case in ("debby", "pinatubo"):
        parts = []
        for d in dirs:
            f = d / "results_jobs_final.csv"
            if not f.exists():
                f = d / "results_jobs_snapshot.csv"
            j = pd.read_csv(f); j["_dir"] = str(d); parts.append(j)
        j = pd.concat(parts, ignore_index=True).drop_duplicates("trial_id")
        h = retention.harmonized_case(case, j)
        lin = j["prob_rule"] == "linear"
        sel = j[lin & h]
        pool = [(int(t), None, dd) for t, dd in zip(sel.trial_id, sel._dir)]
        n_old = int((retention.debby_retained(j) if case == "debby" else retention.pinatubo_retained(j))[lin].sum())
    else:
        d = dirs[0]
        p = pd.read_csv(d / "hyperparam_search_summary.csv")
        p["prob_rule"] = [json.load(open(d / f"run_{int(i):04d}" / "params.json"))["prob_rule"] for i in p.run_id]
        h = retention.harmonized_case("pnw", p)
        lin = p["prob_rule"] == "linear"
        sel = p[lin & h]
        pool = [(int(t), None, str(d)) for t in sel.run_id]
        j = p
        n_old = int(retention.pnw_retained(p)[lin].sum())
    return pool, dict(case=case, n_total=len(j), n_linear=int(lin.sum()), n_pool=len(pool),
                      n_linear_old_retention=n_old, retention="linear & harmonized_case", dirs=";".join(map(str, dirs)))


def k_sample(case, pool, n, rng):
    import pickle
    import K_linearity_check as K
    rows, cache = [], {}
    while len(rows) < n:
        tid, am, d = pool[int(rng.integers(len(pool)))]
        key = (tid, am)
        if key not in cache:
            if case == "t2":
                stem = f"trial_{tid:05d}_alpha_{str(am).replace('.', 'p')}"
                b = R1 / "T2r_multivariate_en" / "bundles"
                cache[key] = (json.load(open(b / f"{stem}.json"))["params"], K._members_from_npz(b / f"{stem}.npz"))
            elif case == "debby":
                b = Path(d) / "bundles"
                cache[key] = (json.load(open(b / f"debby_trial_{tid:05d}.json"))["params"],
                              K._members_from_npz(b / f"debby_trial_{tid:05d}.npz"))
            elif case == "pinatubo":
                b = sorted((Path(d) / "bundles").glob(f"*trial_{tid:05d}.json"))[0]
                cache[key] = (json.load(open(b))["params"], K._members_from_npz(b.with_suffix(".npz")))
            else:
                rd = Path(d) / f"run_{tid:04d}"
                res = pickle.load(open(rd / "ensemble_results.pkl", "rb"))
                mem = [(int(r["member_id"]), np.asarray(r["centers"], float), np.asarray(r["parents"], int))
                       for r in res if r["success"] and r["centers"] is not None and len(r["centers"]) >= 2]
                cache[key] = (json.load(open(rd / "params.json")), mem)
        params, mem = cache[key]
        if not mem:
            continue
        m, centers, parents = mem[int(rng.integers(len(mem)))]
        k = int(rng.integers(len(parents) - 1))
        rows.append(dict(case=case, config=tid, alpha_mix=am, member=m, k=k, lat=float(centers[k, 0]),
                         lon=float(centers[k, 1]), child=int(parents[k]), params=params))
    return rows


def run_K(source, nproc, n_steps=100, plot_only=False):
    import K_linearity_check as K
    out = outdir("K" if source == "night1" else "K_night5")
    cases = ["t2", "debby", "pinatubo", "pnw"]
    if not plot_only:
        pools = []
        for case in cases:
            t0 = time.time()
            pool, info = k_pool(case, source)
            pools.append(info)
            print(f"[K {case}] pool {info}", flush=True)
            if not pool:
                print(f"[K {case}] empty pool, skipped", flush=True); continue
            jobs = k_sample(case, pool, n_steps, np.random.default_rng(0))
            fd, de = K.load_case(case)
            if case != "t2":
                fd = {None: fd[None].load()}
            K.WG["full_data"] = fd; K.WG["date_end"] = de
            with mp.get_context("fork").Pool(nproc) as p:
                res = p.map(K.analyse_step, jobs, chunksize=1)
            df = pd.DataFrame(res)
            df.to_csv(out / f"K_steps_{case}.csv", index=False)
            print(f"[K {case}] {time.time()-t0:.0f}s status {df.status.value_counts().to_dict()}", flush=True)
        pd.DataFrame(pools).to_csv(out / "K_pools.csv", index=False)
    # summary via the night-2 function (its old-style figure goes to a scratch subdir of the T7 table dir)
    K.summarize_and_plot(out, out / "_night2_style_fig_unused", cases)
    Sm = pd.read_csv(out / "K_summary.csv")
    S2 = pd.read_csv(V2 / "K" / "K_summary.csv")
    keys = ["n_ok", "median_adjr2_ols", "median_d_adjr2_reset", "median_d_adjr2_spline", "median_d_adjr2_group",
            "frac_bic_reset_gt10", "frac_bic_spline_gt10"]
    cmp_ = Sm[["case"] + keys].merge(S2[["case"] + keys], on="case", suffixes=("_linear_harm", "_night2_all_oldret"))
    cmp_.to_csv(out / "K_compare_night2.csv", index=False); md(cmp_.round(4), out / "K_compare_night2.md")
    print(cmp_.round(4).T.to_string(), flush=True)

    S.set_style()
    df = pd.concat([pd.read_csv(out / f"K_steps_{c}.csv") for c in cases if (out / f"K_steps_{c}.csv").exists()])
    have = [c for c in cases if (df.case == c).any()]
    cols = {"t2": "#999999", "debby": "#0072B2", "pinatubo": "#D55E00", "pnw": "#CC79A7"}
    names = {"t2": "T2-EN\n(linear ref.)", "debby": "Debby", "pinatubo": "Pinatubo", "pnw": "PNW\n2021"}
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, 6.5 * S.CM))
    for ax, met, ttl in [(axes[0], "d_adjr2_reset", "RESET (+ fitted$^2$, fitted$^3$)"),
                         (axes[1], "d_adjr2_spline", "+ group splines and products"),
                         (axes[2], "d_adjr2_group", "Group-level model")]:
        data = [df[df.case == c][met].dropna().values for c in have]
        pos = np.arange(len(data))
        vp = ax.violinplot([d if len(d) else [np.nan] for d in data], positions=pos, showextrema=False, widths=0.8)
        for b, c in zip(vp["bodies"], have):
            b.set_facecolor(cols[c]); b.set_alpha(0.5); b.set_edgecolor("#333333"); b.set_linewidth(0.4)
        ax.boxplot(data, positions=pos, widths=0.22, showfliers=False, medianprops=dict(color="k", lw=1.2),
                   boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xticks(pos); ax.set_xticklabels([names[c] for c in have], fontsize=6)
        ax.set_ylabel("Adjusted $R^2$ gain over\nlinear OLS (unitless)")
        ax.set_title(ttl)
    src = "night-1 baselines" if source == "night1" else "night-5 linear runs"
    print("K figure source:", src, flush=True)
    S.panel_labels(axes, x=-0.3)
    fig.tight_layout(w_pad=1.2)
    print(S.save(fig, "S_linearity", "S_linearity" if source == "night1" else "S_linearity_night5"), flush=True)


# ============================================================== J (existing robustness table, filtered)
def run_J():
    from J_sensitivity import wilson
    out = outdir("J")
    J = V2 / "J_sensitivity"
    specs = [("debby", "Debby (rebuilt vo)", "vo_top_120_360", "Vorticity top, 5–15 d"),
             ("debby_orig", "Debby (original vo)", "vo_top_120_360", "Vorticity top, 5–15 d"),
             ("pinatubo", "Pinatubo", "so2_top_4_16", "SO$_2$ top, 4–16 d (stored)"),
             ("pnw", "PNW 2021", "z500_top_1_10", "Z500 top, 1–10 d (stored)"),
             ("pnw", "PNW 2021", "z500_top_1_10_excl_start", "Z500 top, 1–10 d excl. step 0")]
    rows = []
    for case, exp, col, head in specs:
        t = pd.read_csv(J / f"J_configs_{case}.csv")
        hcase = "pnw" if case == "pnw" else ("debby" if case.startswith("debby") else "pinatubo")
        harm = retention.harmonized_case(hcase, t).values
        lin = (t["prob_rule"] == "linear").values
        hold = t[col].astype(bool).values
        for lab, mask in (("night 2: all configs, old retention", t["retained"].astype(bool).values),
                          ("linear + old retention", lin & t["retained"].astype(bool).values),
                          ("linear + harmonized (final)", lin & harm)):
            k, n = int(hold[mask].sum()), int(mask.sum())
            lo, hi = wilson(k, n) if n else (np.nan, np.nan)
            rows.append(dict(experiment=exp, headline=head, subset=lab, n=n, n_holds=k,
                             pct=100 * k / n if n else np.nan, wilson95_lo=100 * lo, wilson95_hi=100 * hi))
    R = pd.DataFrame(rows)
    R.to_csv(out / "J_robustness_linear_harmonized.csv", index=False)
    md(R.round(1), out / "J_robustness_linear_harmonized.md")
    print(R.round(1).to_string(), flush=True)
    S.set_style()
    fig, ax = plt.subplots(figsize=(S.SINGLE, 7.5 * S.CM))
    subs = ["night 2: all configs, old retention", "linear + old retention", "linear + harmonized (final)"]
    sc = dict(zip(subs, ["#999999", "#56B4E9", "#0072B2"]))
    labs = [f"{e}\n{h}" for _, e, _, h in specs]
    y = np.arange(len(specs))[::-1]
    for k, s_ in enumerate(subs):
        g = R[R.subset == s_].reset_index(drop=True)
        yy = y + (1 - k) * 0.25
        ax.errorbar(g.pct, yy, xerr=[g.pct - g.wilson95_lo, g.wilson95_hi - g.pct], fmt="o", ms=3, color=sc[s_],
                    elinewidth=0.8, capsize=1.5, label=s_)
        for xi, yi, n in zip(g.pct, yy, g.n):
            ax.text(106, yi, f"n = {n}", fontsize=5, va="center", color=sc[s_])
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=5.8)
    ax.set_xlim(0, 122); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Configs in which the headline holds (%)")
    ax.axvline(50, color="#999999", lw=0.5, ls=":")
    ax.legend(loc="lower left", fontsize=5.3, bbox_to_anchor=(0, 1.0), ncol=1)
    fig.tight_layout()
    print(S.save(fig, "S_sensitivity_existing", "S_sensitivity_existing_robustness"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["E", "G", "H", "F", "K", "J"])
    ap.add_argument("--source", default=os.environ.get("TST_T7_SOURCE", "night1"), choices=["night1", "night5"])
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--k-plot-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("source:", a.source, "(only K uses it)", flush=True)
    for w in a.which:
        t0 = time.time()
        print(f"===== {w}", flush=True)
        {"E": run_E, "G": run_G, "H": run_H, "F": run_F, "J": run_J,
         "K": lambda: run_K(a.source, a.nproc, plot_only=a.k_plot_only)}[w]()
        print(f"===== {w} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
