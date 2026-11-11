"""Task J (R1 major 2): hyperparameter sensitivity, guidance numbers, robustness, N_samples/N_features.

Per experiment, one row per configuration (all configurations with ok == 1; error metrics only where defined):
  T1r (3 engines)  completion = fraction of the 3 tracks with raw_len == 31; errors over retained configs
                   (trace_st.retention.t1_retained): mean track distance, origin distance (stored);
                   share = fraction of V2 steps (V2 is not a true parent).
  T2r EN (+T10 T2_wide)  alpha_mix = 0.6 rows only (T2_wide ran only 0.6): completion = fraction of members
                   with 31 centres; |alpha error| = |estimated_alpha_density - 0.6|;
                   share = true-source density share (V1 + V3) = 1 - density_frac_var2.
  Debby (rebuilt vo; T4r_debby_vorebuilt)  completion = fraction of the 30 members with 358 centres (the
                   longest stored); error = mean_of_member_mean_track_dist_km (vs IBTrACS, as stored);
                   share = vorticity share 120-360 h (recomputed, I_step_densities.py).
  Pinatubo         completion = fraction of members with 17 centres; error = mean_final_dist_to_volcano_km;
                   share = SO2 share in the stored density_range (4-16 d; steps [4,16) of the 1-d axis).
  PNW (+T10 pnw_wide)  completion = fraction_full_length; no reference track; share = Z500 days 1-10 (stored).
Robustness (retained configs): Debby vo top at 120-360 h (and 96-360 h stored); Pinatubo SO2 top 4-16 d;
PNW Z500 top 1-10 d (stored, and without the start step).
Table S1 (derived): for Elastic-Net (castle_core.mv_CaStLe_GRANGER / _lasso_single_child_lag1):
  N_samples = (ny - 2r)(nx - 2r)(tw - 1), N_features = n_vars (2r + 1)^2, with ny, nx = grid points inside the
  box [c - b/2, c + b/2] at the starting point (data_io.get_data_mcastle) and tw = timewindow / timeres.
  Same formula as the stored t7_samples_per_feature (3_SyntheticRuns_Multivariate.py), checked against it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from trace_st import retention  # noqa: E402

TASK = "J_sensitivity"
OUT = C.outdir(TASK)
PARAMS = ["timewindow", "box_size", "radius", "eps_dbscan", "score_mode", "alpha", "prob_rule", "beta_softmax",
          "lambda_a", "l1_ratio"]
CONT = {"lambda_a", "l1_ratio"}


def _step(case):
    z = np.load(C.OUT / "I_significance" / f"step_density_{case}.npz", allow_pickle=True)
    return pd.Series(range(len(z["ids"])), index=z["ids"]), z["D"]


def _shares_window(case, ids, a, b):
    pos, D = _step(case)
    M = D[pos.loc[ids].values, a:b, :].sum(axis=1)
    tot = M.sum(1, keepdims=True)
    return np.where(tot > 0, M / np.where(tot > 0, tot, 1), np.nan)


# ---------------------------------------------------------------- per-experiment tables
def table_t1():
    d = C.RUNS["t1"]
    tr = pd.read_csv(d / "results_tracks_final.csv", usecols=["method", "trial_id", "raw_len", "mean_distance",
                                                              "origin_distance", "parent2_frac"])
    jobs = pd.read_csv(d / "results_jobs_final.csv")
    jobs = jobs[jobs.ok == 1]
    g = tr.groupby(["method", "trial_id"]).agg(completion=("raw_len", lambda x: np.mean(x == 31)),
                                               min_raw_len=("raw_len", "min"), track_err=("mean_distance", "mean"),
                                               origin_err=("origin_distance", "mean"), share=("parent2_frac", "mean")).reset_index()
    t = jobs.merge(g, on=["method", "trial_id"])
    t["retained"] = t["min_raw_len"] == 31
    for c in ("track_err", "origin_err", "share"):
        t.loc[~t.retained, c] = np.nan
    t["source"] = "main"
    return t


def table_t2(case):
    d = C.RUNS[case]
    jobs = pd.read_csv(d / "results_jobs_final.csv")
    al = pd.read_csv(d / "results_alphas_final.csv")
    al = al[np.isclose(al.alpha_mix, 0.6)]
    m = pd.read_csv(d / "results_members_final.csv", usecols=["trial_id", "alpha_mix", "ok", "n_steps"])
    m = m[np.isclose(m.alpha_mix, 0.6)]
    comp = m.groupby("trial_id").apply(lambda g: np.mean((g.ok == 1) & (g.n_steps == 31))).rename("completion")
    a = al[["trial_id", "estimated_alpha_density", "density_frac_var2"]].copy()
    a["alpha_err"] = (a.estimated_alpha_density - 0.6).abs()
    a["share"] = 1 - a.density_frac_var2
    t = jobs[jobs.ok == 1].merge(a, on="trial_id").merge(comp.reset_index(), on="trial_id")
    t["retained"] = True
    t["source"] = "main" if case == "t2" else "T10 wide"
    return t


def table_debby(case="debby"):
    j = C.load_jobs(case)
    m = pd.read_csv(C.RUNS[case] / "results_members_final.csv", usecols=["trial_id", "ok", "n_steps"])
    comp = m.groupby("trial_id").apply(lambda g: np.mean((g.ok == 1) & (g.n_steps >= 358))).rename("completion")
    t = j.merge(comp.reset_index(), on="trial_id")
    t["error"] = t["mean_of_member_mean_track_dist_km"]
    S = _shares_window(case, t.trial_id.values, 120, 360)
    t["share"] = S[:, C.VARS[case].index("vo")]
    t["vo_top_120_360"] = np.argmax(np.nan_to_num(S, nan=-1), 1) == C.VARS[case].index("vo")
    st = t[[f"density_range::{v}" for v in C.VARS[case]]].values
    t["vo_top_96_360"] = np.argmax(st, 1) == C.VARS[case].index("vo")
    t["retained"] = retention.debby_retained(t)
    t["lambda_a"], t["l1_ratio"] = t["cd_kwargs.lambda_a"], t["cd_kwargs.l1_ratio"]
    t["source"] = "main"
    return t


def table_pinatubo():
    j = C.load_jobs("pinatubo")
    m = pd.read_csv(C.RUNS["pinatubo"] / "results_members_final.csv", usecols=["trial_id", "ok", "n_steps"])
    comp = m.groupby("trial_id").apply(lambda g: np.mean((g.ok == 1) & (g.n_steps >= 17))).rename("completion")
    t = j.merge(comp.reset_index(), on="trial_id")
    t["error"] = t["mean_final_dist_to_volcano_km"]
    st = t[[f"density_range::{v}" for v in C.VARS["pinatubo"]]].values.astype(float)
    tot = st.sum(1, keepdims=True)
    t["share"] = np.where(tot[:, 0] > 0, st[:, 0] / np.where(tot[:, 0] > 0, tot[:, 0], 1), np.nan)
    t["so2_top_4_16"] = (np.argmax(st, 1) == 0) & (tot[:, 0] > 0)
    t["retained"] = retention.pinatubo_retained(t)
    t["lambda_a"], t["l1_ratio"] = t["cd_kwargs.lambda_a"], t["cd_kwargs.l1_ratio"]
    t["source"] = "main"
    return t


def table_pnw(case):
    s = C.load_pnw_summary(case)
    rows = []
    for rid in s.run_id:
        p = json.load(open(C.RUNS[case] / f"run_{int(rid):04d}" / "params.json"))
        rows.append(dict(run_id=int(rid), **{k: p[k] for k in PARAMS if k in p},
                         lambda_a=p["cd_kwargs"]["lambda_a"], l1_ratio=p["cd_kwargs"]["l1_ratio"]))
    t = s.merge(pd.DataFrame(rows), on="run_id")
    t["completion"] = t["fraction_full_length"]
    names = C.VARS["pnw"]
    st = t[[f"days_1_10_integrated_norm_{v}" for v in names]].values
    t["share"] = st[:, 0]
    t["z500_top_1_10"] = np.argmax(np.nan_to_num(st, nan=-1), 1) == 0
    S = _shares_window(case, t.run_id.values, 1, 10)
    t["share_excl_start"] = S[:, 0]
    t["z500_top_1_10_excl_start"] = np.argmax(np.nan_to_num(S, nan=-1), 1) == 0
    t["retained"] = retention.pnw_retained(t)
    t["source"] = "main" if case == "pnw" else "T10 wide"
    return t


# ---------------------------------------------------------------- PD plots
def _levels(t_main, t_wide, p):
    if p in CONT:
        v = np.log10(pd.concat([t_main[p], t_wide[p]] if t_wide is not None else [t_main[p]]).astype(float))
        edges = np.arange(np.floor(v.min() * 2) / 2, np.ceil(v.max() * 2) / 2 + 0.5, 0.5)
        return "bin", edges
    vals = pd.concat([t_main[p], t_wide[p]] if t_wide is not None else [t_main[p]]).dropna().unique()
    try:
        vals = sorted(vals, key=lambda x: float(str(x).rstrip("dh")))
    except ValueError:
        vals = sorted(vals, key=str)
    return "cat", vals


def _cat_of(t, p, kind, lv):
    if kind == "bin":
        idx = np.digitize(np.log10(t[p].astype(float)), lv) - 1
        return idx
    m = {v: i for i, v in enumerate(lv)}
    return t[p].map(m).values


def _labels(kind, lv):
    if kind == "bin":
        return [f"{lv[i]:.1f}–{lv[i+1]:.1f}" for i in range(len(lv) - 1)]
    return [str(v) for v in lv]


def pd_figure(t_main, metric, ylabel, title, name, t_wide=None, hue=None, params=PARAMS):
    params = [p for p in params if p in t_main.columns and t_main[p].nunique() > 1]
    nc = 5
    nr = int(np.ceil(len(params) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(24, 13.5 * nr / 2), squeeze=False, sharey=True)
    rows = []
    groups = [(None, t_main)] if hue is None else [(h, t_main[t_main[hue] == h]) for h in hue_order(t_main, hue)]
    for ax, p in zip(axes.flat, params):
        kind, lv = _levels(t_main, t_wide, p)
        labs = _labels(kind, lv)
        nlev = len(labs)
        ng = len(groups)
        w = 0.8 / ng
        main_levels = set(pd.Series(_cat_of(t_main, p, kind, lv)).dropna().astype(int))
        for i in range(nlev):
            if i not in main_levels:
                ax.axvspan(i - 0.5, i + 0.5, color="0.9", zorder=0)
        for gi, (h, g) in enumerate(groups):
            cat = _cat_of(g, p, kind, lv)
            data, pos = [], []
            for i in range(nlev):
                v = g.loc[cat == i, metric].dropna().values
                if len(v):
                    data.append(v); pos.append(i - 0.4 + w / 2 + gi * w)
                    rows.append(dict(param=p, level=labs[i], group=h if h is not None else "main", n=len(v),
                                     median=np.median(v), q25=np.percentile(v, 25), q75=np.percentile(v, 75), mean=v.mean()))
            col = C.ENGINE_COLORS.get(h, "#0072B2") if h is not None else "#0072B2"
            if data:
                bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True, showfliers=False,
                                medianprops=dict(color="0.1", lw=1.6))
                for b in bp["boxes"]:
                    b.set_facecolor(col); b.set_alpha(0.5); b.set_edgecolor(col)
        if t_wide is not None and p in t_wide.columns:
            cat = _cat_of(t_wide, p, kind, lv)
            v = t_wide[metric].values
            ok = np.isfinite(v) & np.isfinite(cat.astype(float))
            rng = np.random.default_rng(0)
            ax.scatter(cat[ok] + rng.uniform(-0.25, 0.25, ok.sum()), v[ok], s=22, facecolors="none",
                       edgecolors="#D55E00", lw=1.2, zorder=5)
            for i in range(nlev):
                vv = v[(cat == i) & ok]
                if len(vv):
                    rows.append(dict(param=p, level=labs[i], group="T10 wide", n=len(vv), median=np.median(vv),
                                     q25=np.percentile(vv, 25), q75=np.percentile(vv, 75), mean=vv.mean(),
                                     outside_main_range=i not in main_levels))
        ax.set_xticks(range(nlev))
        ax.set_xticklabels(labs, rotation=45 if nlev > 4 or kind == "bin" else 0, ha="right" if nlev > 4 or kind == "bin" else "center", fontsize=13)
        ax.set_xlim(-0.6, nlev - 0.4)
        ax.set_title(p + (" (log$_{10}$)" if kind == "bin" else ""), fontsize=17)
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(params):]:
        ax.axis("off")
    for r in range(nr):
        axes[r, 0].set_ylabel(ylabel)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hs = [Patch(facecolor=C.ENGINE_COLORS.get(h, "#0072B2") if h else "#0072B2", alpha=0.5,
                label=C.ENGINE_LABELS.get(h, "Main ensemble") if h else "Main ensemble") for h, _ in groups]
    if t_wide is not None:
        hs += [Line2D([], [], marker="o", ls="", mfc="none", mec="#D55E00", label="T10 wide screening"),
               Patch(facecolor="0.9", label="outside the main sampled range")]
    fig.legend(handles=hs, loc="upper right", ncol=len(hs), fontsize=15, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=21)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    C.savefig(fig, TASK, name)
    plt.close(fig)
    return pd.DataFrame(rows).assign(metric=metric, figure=name)


def hue_order(t, hue):
    return [h for h in ["granger", "pcmci", "dynotears"] if h in set(t[hue])]


# ---------------------------------------------------------------- robustness / Table S1
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def npts(grid, c, b):
    return int(((grid >= c - b / 2) & (grid <= c + b / 2)).sum())


def table_s1(tabs):
    ys_syn, xs_syn = np.linspace(0, 200, 121), np.linspace(0, 200, 241)
    b = np.load(C.load_jobs("debby").bundle_npz[0]); dxs, dys = np.asarray(b["xs"], float), np.asarray(b["ys"], float)
    b = np.load(C.load_jobs("pinatubo").bundle_npz[0]); pxs, pys = np.asarray(b["xs"], float), np.asarray(b["ys"], float)
    exp = {
        "T1r (univariate; 3 events)": (tabs["t1"], 2, [(ys_syn, xs_syn, (100.0, 50.0)), (ys_syn, xs_syn, (110.0, 105.0)), (ys_syn, xs_syn, (70.0, 100.0))], "d"),
        "T2r EN (multivariate)": (tabs["t2"], 3, [(ys_syn, xs_syn, (100.0, 100.0))], "d"),
        "T10 T2_wide": (tabs["t2_wide"], 3, [(ys_syn, xs_syn, (100.0, 100.0))], "d"),
        "Debby (T4r_debby_vorebuilt)": (tabs["debby"], 3, [(dys, dxs, (28.0, -46.0 % 360))], "h"),
        "Pinatubo (T5r)": (tabs["pinatubo"], 4, [(pys, pxs, (15.0, -160.0 % 360))], "d"),
        "PNW (T6r)": (tabs["pnw"], 6, [(C.PNW_LATS, C.PNW_LONS, (52.5, 240.0))], "d"),
        "T10 pnw_wide": (tabs["pnw_wide"], 6, [(C.PNW_LATS, C.PNW_LONS, (52.5, 240.0))], "d"),
    }
    rows, per = [], []
    for name, (t, nv, grids, unit) in exp.items():
        tt = t.drop_duplicates(subset=["timewindow", "box_size", "radius"] + (["method", "trial_id"] if "method" in t else []))
        for _, r in t.iterrows():
            tw = int(str(r["timewindow"]).rstrip("dh"))
            rr = int(r["radius"]); bb = float(r["box_size"])
            ns = np.mean([max(npts(ys, c[0], bb) - 2 * rr, 0) * max(npts(xs, c[1], bb) - 2 * rr, 0) * (tw - 1) for ys, xs, c in grids])
            nf = nv * (2 * rr + 1) ** 2
            per.append(dict(experiment=name, retained=bool(r.get("retained", True)), n_samples=ns, n_features=nf, ratio=ns / nf))
    P = pd.DataFrame(per)
    for (name, ret), g in P.groupby(["experiment", "retained"]):
        rows.append(dict(experiment=name, configs="retained" if ret else "not retained", n=len(g),
                         n_samples_median=g.n_samples.median(), n_samples_min=g.n_samples.min(), n_samples_max=g.n_samples.max(),
                         n_features_median=g.n_features.median(), n_features_min=g.n_features.min(), n_features_max=g.n_features.max(),
                         ratio_median=g.ratio.median(), ratio_min=g.ratio.min(), ratio_max=g.ratio.max()))
    for name, g in P.groupby("experiment"):
        rows.append(dict(experiment=name, configs="all", n=len(g), n_samples_median=g.n_samples.median(), n_samples_min=g.n_samples.min(),
                         n_samples_max=g.n_samples.max(), n_features_median=g.n_features.median(), n_features_min=g.n_features.min(),
                         n_features_max=g.n_features.max(), ratio_median=g.ratio.median(), ratio_min=g.ratio.min(), ratio_max=g.ratio.max()))
    S = pd.DataFrame(rows).sort_values(["experiment", "configs"])
    # check against stored T7 values (same formula in 3_SyntheticRuns_Multivariate.py)
    chk = []
    for f in sorted((C.R1 / "T7_decoy_sweep").glob("*/results_alphas_final.csv")) + sorted((C.R1 / "T7_distractors").glob("*/results_alphas_final.csv")):
        a = pd.read_csv(f)
        if "t7_p" not in a.columns:
            continue
        a = a.drop_duplicates(subset=["box_size", "radius", "timewindow", "t7_p"])
        for _, r in a.iterrows():
            tw = int(str(r.timewindow).rstrip("d")); rr = int(r.radius)
            ns = max(npts(ys_syn, 100.0, r.box_size) - 2 * rr, 0) * max(npts(xs_syn, 100.0, r.box_size) - 2 * rr, 0) * (tw - 1)
            chk.append(dict(file=str(f.parent.name), ok_samples=ns == r.t7_n_samples, ok_features=int(r.t7_p) * (2 * rr + 1) ** 2 == r.t7_n_features))
    chk = pd.DataFrame(chk)
    return S, dict(n_checked=len(chk), all_samples_match=bool(chk.ok_samples.all()), all_features_match=bool(chk.ok_features.all()))


def main():
    C.set_style()
    tabs = dict(t1=table_t1(), t2=table_t2("t2"), t2_wide=table_t2("t2_wide"), debby=table_debby("debby"),
                debby_orig=table_debby("debby_orig"), pinatubo=table_pinatubo(), pnw=table_pnw("pnw"), pnw_wide=table_pnw("pnw_wide"))
    for k, t in tabs.items():
        t.to_csv(OUT / f"J_configs_{k}.csv", index=False)
    pdrows = []
    t1 = tabs["t1"]
    pdrows += [pd_figure(t1, "completion", "Fraction of 3 tracks complete", "T1 synthetic: completion by hyperparameter", "T1_completion", hue="method"),
               pd_figure(t1, "track_err", "Mean track distance (grid units)", "T1 synthetic: track error (retained configs)", "T1_track_error", hue="method"),
               pd_figure(t1, "origin_err", "Origin distance (grid units)", "T1 synthetic: origin error (retained configs)", "T1_origin_error", hue="method"),
               pd_figure(t1, "share", "Fraction of V$_2$ steps", "T1 synthetic: fraction of steps on the non-parent V$_2$", "T1_V2_share", hue="method")]
    t2, w2 = tabs["t2"], tabs["t2_wide"]
    pdrows += [pd_figure(t2, "completion", "Fraction of members complete", "T2 Elastic-Net (α$_{mix}$ = 0.6): completion", "T2_completion", t_wide=w2),
               pd_figure(t2, "alpha_err", "|α error| (density estimate)", "T2 Elastic-Net: |recovered − prescribed α$_{mix}$| (0.6)", "T2_alpha_error", t_wide=w2),
               pd_figure(t2, "share", "V$_1$+V$_3$ density share", "T2 Elastic-Net: density share on the true sources", "T2_source_share", t_wide=w2)]
    d = tabs["debby"]
    pdrows += [pd_figure(d, "completion", "Fraction of members reaching 358 h", "Debby (rebuilt vo): completion", "Debby_completion"),
               pd_figure(d, "error", "Mean track distance to IBTrACS (km)", "Debby (rebuilt vo): track distance to IBTrACS", "Debby_track_error"),
               pd_figure(d, "share", "Vorticity share, 120–360 h", "Debby (rebuilt vo): vorticity share at 5–15 d", "Debby_vo_share")]
    p = tabs["pinatubo"]
    pdrows += [pd_figure(p, "completion", "Fraction of members reaching 16 d", "Pinatubo: completion", "Pinatubo_completion"),
               pd_figure(p, "error", "Final distance to volcano (km)", "Pinatubo: final distance to the volcano", "Pinatubo_final_distance"),
               pd_figure(p, "share", "SO$_2$ share, 4–16 d", "Pinatubo: SO$_2$ share at 4–16 d", "Pinatubo_SO2_share")]
    n, nw = tabs["pnw"], tabs["pnw_wide"]
    pdrows += [pd_figure(n, "completion", "Fraction of members full length (30 d)", "PNW 2021: completion", "PNW_completion", t_wide=nw),
               pd_figure(n, "share", "Z500 share, 1–10 d (stored)", "PNW 2021: Z500 share at 1–10 d", "PNW_Z500_share", t_wide=nw)]
    PD = pd.concat(pdrows, ignore_index=True)
    PD.to_csv(OUT / "J_partial_dependence_levels.csv", index=False)

    # robustness
    rob = []
    def add(exp, headline, mask, ret):
        k, nn = int(mask[ret].sum()), int(ret.sum())
        lo, hi = wilson(k, nn)
        rob.append(dict(experiment=exp, headline=headline, n_retained=nn, n_holds=k, pct=100 * k / nn, wilson95_lo=100 * lo, wilson95_hi=100 * hi))
    for c, lab in (("debby", "Debby rebuilt vo (primary)"), ("debby_orig", "Debby original vo")):
        t = tabs[c]
        add(lab, "vorticity top parent 120–360 h (5–15 d)", t.vo_top_120_360.values, t.retained.values)
        add(lab, "vorticity top parent 96–360 h (stored window)", t.vo_top_96_360.values, t.retained.values)
    add("Pinatubo", "SO2 top 4–16 d (stored, 1-d steps [4,16))", p.so2_top_4_16.values, p.retained.values)
    add("PNW", "Z500 top 1–10 d (stored)", n.z500_top_1_10.values, n.retained.values)
    add("PNW", "Z500 top 1–10 d excl. start step", n.z500_top_1_10_excl_start.values, n.retained.values)
    add("T10 pnw_wide", "Z500 top 1–10 d (stored)", nw.z500_top_1_10.values, nw.retained.values)
    add("T10 pnw_wide", "Z500 top 1–10 d excl. start step", nw.z500_top_1_10_excl_start.values, nw.retained.values)
    R = pd.DataFrame(rob)
    R.to_csv(OUT / "J_robustness.csv", index=False)
    (OUT / "J_robustness.md").write_text(C.to_markdown(R))

    # guidance: completion by single level, with retained fraction
    G = []
    for k, t in tabs.items():
        for prm in PARAMS:
            if prm not in t.columns or t[prm].nunique() < 2 or prm in CONT:
                continue
            for h, g in (t.groupby("method") if k == "t1" else [("", t)]):
                for lvl, gg in g.groupby(prm):
                    G.append(dict(experiment=k, engine=h, param=prm, level=lvl, n=len(gg), completion_mean=gg.completion.mean(),
                                  retained_frac=gg.retained.mean()))
    pd.DataFrame(G).to_csv(OUT / "J_guidance_completion_by_level.csv", index=False)

    S1, chk = table_s1(tabs)
    S1.to_csv(OUT / "J_tableS1_samples_per_feature.csv", index=False)
    (OUT / "J_tableS1_samples_per_feature.md").write_text(
        "Derived (not stored): Elastic-Net design size at the starting point; see J_sensitivity.py docstring.\n"
        f"Formula check against stored T7 t7_n_samples/t7_n_features: {chk}\n\n" + C.to_markdown(S1))
    print(R.round(1).to_string())
    print(S1[S1.configs != "not retained"].round(2).to_string())
    print(chk)


if __name__ == "__main__":
    main()
