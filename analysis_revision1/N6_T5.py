"""Night-6 T5: supplementary and response figures and tables at the final tag, in figures_v6 (N5_style).

Final method: linear probability rule only, harmonized retention (trace_st.retention.harmonized_case). The night-5
scripts are imported, not copied; their figures are redirected with N5_style.set_output() and renamed to the paper
figure numbers (RENAME), their tables go to night6/analysis/T5_supp/<X>/ (night-5 tables untouched).

  --which S13  truth errors (T1 mean track distance, T2 |alpha err|) vs every parameter, linear + admissible configs
               (N5_T4 tables, source night5: T1/T2 = night-1 linear configs); also S13b/S13c = the night-5 T4
               sensitivity figures (completion, share, truth error for all five cases) regenerated.
          S14  retention sweep and counts (N5_T4_figs.fig_retention, source night5).
          S15  winner-grouping alternatives (N5_T7_supp E) + the regression-stencil schematic (castle_core
               get_MV_reduced_space / mv_CaStLe_GRANGER, movement._add_delta_cols / snap_delta_to_stencil).
          S16  exact tree vs Monte Carlo (N5_T7_supp F, redraw).
          S17  AGL vs Elastic-Net (night5/analysis/T5_agl_vs_en.csv; N5_T5.py).
          S18  linearity check (N5_T7_supp K, source night5 = linear-forced original-set runs; recomputed).
          S19  longer N (N5_T7_supp G) and relaxed success (N5_T7_supp H) -> S19a, S19b.
          R1   T2 alpha slope by probability rule (T2r_multivariate_en, N5_T0.alpha_fit).
          R2   refit vs own coefficients by engine, T1 and T2 (N5_T8 table; T2-DYNOTEARS not run).
          TS1  Table S1: samples per feature for the original and final variable sets and the synthetic cases.
          T8   refit table (N5_T8.main with OUT = night6 tables).  T5  AGL table (copy of the night-5 table).
python N6_T5.py --which S13 S14 ... [--nproc 16]      (PBS develop only)
Figures: revision1/figures_v6/<folder>/; tables: revision1/night6/analysis/tables/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import N5_style as S  # noqa: E402  (sets Agg)
import rev1_common as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

R1 = C.R1
N5 = R1 / "night5"
N6 = R1 / "night6"
FIG6 = R1 / "figures_v6"
TAB = N6 / "analysis" / "tables"
SUPP = N6 / "analysis" / "T5_supp"
RENAME = {
    ("guidelines", "T4_sensitivity_scalefree"): ("S13_sensitivity", "S13b_sensitivity_scalefree_all_cases"),
    ("guidelines", "T4_sensitivity_scale"): ("S13_sensitivity", "S13c_sensitivity_scale_all_cases"),
    ("guidelines", "T4_retention"): ("S14_retention", "S14_retention_sweep_counts"),
    ("S_DBSCAN_grouping", "S_DBSCAN_grouping"): ("S15_grouping", "S15_grouping_stencil"),
    ("S_exact_vs_MC", "S_exact_vs_MC"): ("S16_exact_vs_MC", "S16_exact_vs_MC"),
    ("S_linearity", "S_linearity_night5"): ("S18_linearity", "S18_linearity"),
    ("S_longer_N", "S_longer_N"): ("S19_longer_N_relaxed", "S19a_longer_N"),
    ("S_relaxed_success", "S_relaxed_success"): ("S19_longer_N_relaxed", "S19b_relaxed_success"),
}
P_FINAL = dict(debby=6, pinatubo=7, pnw=7)      # final sets (DECISIONS 2026-11-17)
P_ORIG = dict(t1=2, t2=3, debby=3, pinatubo=4, pnw=6)
SETS = dict(debby="vo, Tb, precip, CAPE, DIV200, MSLP", pinatubo="TMSO201, BURDENSO401, AEROD_v, TMH2SO401, TMSO203, "
            "BURDENSO403, T050", pnw="Z500, TCWV, OLR, SLHF, SSHF, Z10, T2m")


def md(df, path, fmt="{:.4g}", head=""):
    Path(path).write_text(head + C.to_markdown(df, fmt))


# ============================================================== S13
def run_S13():
    import N5_T4_figs as F
    from N5_T4_analysis import EDGES, lin, load
    T = load("night5")
    S.set_style()
    F.fig_free(T); F.fig_scale_pd(T)                     # S13b, S13c (all cases)
    params = ["eps_dbscan", "alpha", "score_mode", "lambda_a", "l1_ratio", "timewindow", "box_size", "radius", "np_ratio"]
    xl = dict(eps_dbscan="DBSCAN ε", alpha="Score exponent α", score_mode="Score mode", lambda_a="log$_{10}$ λ$_a$",
              l1_ratio="log$_{10}$ l1 ratio", timewindow="Window tw (steps)", box_size="Box (grid units)",
              radius="Stencil radius r (cells)", np_ratio="N/p (samples per feature)")
    off = dict(eps_dbscan=0.006, alpha=0.12, score_mode=0.08, lambda_a=0.05, l1_ratio=0.05)
    lab = dict(t1="T1 (EN): mean track distance", t2="T2-EN: |recovered − true α|")
    unit = dict(t1="grid cells", t2="")
    fig, axs = plt.subplots(3, 3, figsize=(S.DOUBLE, 14 * S.CM), sharey=True)
    rows, hd = [], []
    for c, sgn in (("t1", -1), ("t2", 1)):
        L = lin(T[c])
        if "method" in L:
            L = L[L["method"] == "granger"]
        med = float(L[L.adm].err.median())
        col = F.CASE_COL[c]
        for k, p in enumerate(params):
            ax = axs.ravel()[k]
            if p == "np_ratio":
                b = pd.cut(L.np_ratio, EDGES, right=False)
                lv = [(float(L.np_ratio[b == q].median()), L.index[b == q]) for q in b.cat.categories if (b == q).sum() >= 5]
            else:
                lv = F._levels(c, L, p)
            grp = [L.loc[ix][L.loc[ix, "adm"]].err.dropna().values for _, ix in lv]
            F.ci_points(ax, [x for x, _ in lv], [g / med for g in grp], col, off=sgn * off.get(p, 0.0), ms=3)
            for (x, _), g in zip(lv, grp):
                m, lo, hi = F.boot_ci(g)
                rows.append(dict(case=c, param=p, level=x, n_admissible=len(g), err_mean=m, err_lo=lo, err_hi=hi,
                                 rel_to_case_median=m / med, case_median=med))
        n = int(L.adm.sum())
        hd.append(Line2D([], [], color=col, marker="o", ms=3, label=f"{lab[c]} (n = {n}; median {med:.3g}"
                         + (f" {unit[c]})" if unit[c] else ")")))
    for k, p in enumerate(params):
        ax = axs.ravel()[k]
        ax.axhline(1, color="0.6", lw=0.5, ls=":")
        ax.set_xlabel(xl[p])
        if p == "alpha":
            ax.set_xticks(range(9)); ax.set_xticklabels(["0", ".5", "1", "2", "4", "8", "16", "32", "64"], fontsize=5.5)
        if p == "score_mode":
            ax.set_xticks([0, 1]); ax.set_xticklabels(["mean", "sum"]); ax.set_xlim(-0.6, 1.6)
        if p == "np_ratio":
            ax.set_xscale("log")
        if p in ("timewindow", "radius"):
            ax.set_xticks(sorted({int(v) for v in pd.DataFrame(rows).query("param == @p").level}))
    for a in axs[:, 0]:
        a.set_ylabel("Truth error / case median\n(admissible configs)")
    fig.legend(handles=hd, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    S.panel_labels(axs.ravel(), x=-0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.8, h_pad=1.0)
    print(S.save(fig, "S13_sensitivity", "S13_truth_error_vs_parameters"), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(TAB / "S13_truth_error_by_level.csv", index=False)
    md(R, TAB / "S13_truth_error_by_level.md", head="S13: truth error (T1 mean track distance, grid cells; T2 mean "
       "|alpha err| over 4 alphas) per parameter level, linear + admissible configs, mean and 95 % bootstrap CI "
       "(2000, seed 0). Source: night5/analysis/T4/night5/configs_t{1,2}.csv (night-1 runs).\n\n")
    print(R.round(3).to_string(), flush=True)


# ============================================================== S14
def run_S14():
    import N5_T4_figs as F
    from N5_T4_analysis import T4
    S.set_style()
    print(F.fig_retention(T4 / "night5"), flush=True)


# ============================================================== S15 stencil schematic
def draw_stencil(ax, ax_tx, n=11, r=2):
    """Schematic of one TraCE-ST step for a box of n x n cells and stencil radius r (code definitions); text in ax_tx."""
    ax.set_aspect("equal"); ax.axis("off"); ax_tx.axis("off")
    for i in range(n):
        for j in range(n):
            inner = r <= i < n - r and r <= j < n - r
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor="white" if inner else "#DDDDDD", edgecolor="#AAAAAA", lw=0.3))
    ci, cj = 6, 4                                          # example child cell (interior)
    ax.add_patch(Rectangle((cj - r, ci - r), 2 * r + 1, 2 * r + 1, fill=False, edgecolor="#0072B2", lw=1.3, zorder=3))
    ax.add_patch(Rectangle((cj, ci), 1, 1, facecolor="#D55E00", edgecolor="#D55E00", zorder=4))
    links = [(ci + 2, cj - 1), (ci + 1, cj - 2), (ci + 2, cj - 2)]  # parent cells of the winning cluster (example)
    for pi, pj in links:
        ax.add_patch(Rectangle((pj, pi), 1, 1, facecolor="#56B4E9", edgecolor="#0072B2", lw=0.5, zorder=3))
        ax.annotate("", xy=(cj + 0.5, ci + 0.5), xytext=(pj + 0.5, pi + 0.5), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="#0072B2", lw=0.7, mutation_scale=6))
    di = float(np.mean([p[0] - ci for p in links])); dj = float(np.mean([p[1] - cj for p in links]))
    si, sj = int(np.clip(np.round(di), -r, r)), int(np.clip(np.round(dj), -r, r))
    c0 = n / 2
    ax.add_patch(Rectangle((sj, si), n, n, fill=False, edgecolor="#555555", lw=0.8, ls="--", zorder=2))
    ax.plot(c0, c0, "+", color="k", ms=6, mew=1.0, zorder=6)
    ax.annotate("", xy=(c0 + sj, c0 + si), xytext=(c0, c0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2, mutation_scale=8))
    ax.set_xlim(min(0, sj) - 0.3, max(n, n + sj) + 0.3); ax.set_ylim(min(0, si) - 0.3, max(n, n + si) + 0.3)
    ax.set_title(f"Stencil and box move (r = {r})", fontsize=6.3, loc="right")
    txt = [(f"Box: {n} × {n} cells (box size / Δ).", "k"),
           (f"Grey: margin of r = {r} cells (no full stencil).", "#555555"),
           ("Orange: child cell (target variable) at t.", "#D55E00"),
           (f"Blue frame: its (2r+1)$^2$ = {(2 * r + 1) ** 2} neighbours", "#0072B2"),
           ("  at t−1, in each of the p variables.", "#0072B2"),
           ("One regression sample per interior cell", "k"),
           ("and time pair:", "k"),
           ("  N = (n$_y$ − 2r)(n$_x$ − 2r)(tw − 1) samples,", "k"),
           ("  p(2r+1)$^2$ features.", "k"),
           ("Light blue: parent links of the winning", "#0072B2"),
           ("  cluster. Arrow: box move = their weighted", "k"),
           ("  mean offset × Δ, snapped to |Δi|, |Δj| ≤ r.", "k"),
           ("Dashed: box at the next (earlier) step.", "#555555")]
    for k, (t, cc) in enumerate(txt):
        ax_tx.text(0.0, 1.0 - 0.075 * k, t, fontsize=5.5, color=cc, va="top", ha="left", transform=ax_tx.transAxes)


def run_S15():
    import N5_T7_supp as T7
    T7.OUT = SUPP
    T7.run_E(stencil=draw_stencil)


# ============================================================== S17 AGL vs EN
def run_S17():
    A = pd.read_csv(N5 / "analysis" / "T5_agl_vs_en.csv")
    A.to_csv(TAB / "T5_agl_vs_en.csv", index=False)
    cols = ["case", "run", "n", "retained_frac", "median_track_dist", "mean_track_dist", "lo", "hi", "parent2_frac",
            "median_abs_alpha_err", "mean_abs_alpha_err", "r_pooled", "valid_frac", "median_alpha_0.2", "median_alpha_0.6",
            "median_sec_per_config"]
    md(A[cols], TAB / "T5_agl_vs_en.md", head="AGL (adaptive group lasso, variable-block groups, opt-in fit=agl) vs "
       "Elastic-Net, same configs, linear rule. T1: 200 configs (night5/t5/{en,agl}/T1_200); T2-EN: 50 configs x alpha "
       "0.2/0.6, M 20 (EN = night5/t3/p03, AGL = night5/t5/agl/T2en). lo/hi = 95 % bootstrap CI of the mean track "
       "distance (T1) or of the mean |alpha err| (T2). Script: analysis_revision1/N5_T5.py (night 5); copied here.\n\n")
    S.set_style()
    cc = {"Elastic-Net": "#0072B2", "AGL": "#D55E00"}
    t1, t2 = A[A.case == "T1"].set_index("run"), A[A.case == "T2"].set_index("run")
    runs = ["Elastic-Net", "AGL"]
    fig, ax = plt.subplots(1, 4, figsize=(S.DOUBLE, 5.6 * S.CM))
    for k, r in enumerate(runs):
        g = t1.loc[r]
        ax[0].bar(k, g.mean_track_dist, color=cc[r], width=0.6)
        ax[0].errorbar(k, g.mean_track_dist, yerr=[[g.mean_track_dist - g.lo], [g.hi - g.mean_track_dist]], color="k",
                       capsize=2, lw=0.8)
        ax[0].text(k, g.hi + 0.3, f"median {g.median_track_dist:.1f}", ha="center", fontsize=5.5)
        ax[1].bar(k, g.parent2_frac, color=cc[r], width=0.6)
        ax[1].text(k, g.parent2_frac + 0.01, f"{g.parent2_frac:.3f}", ha="center", fontsize=5.5)
        h = t2.loc[r]
        ax[2].plot([0.2, 0.6], [h["median_alpha_0.2"], h["median_alpha_0.6"]], "o-", color=cc[r], ms=3.5, label=r)
        ax[3].bar(k, h.mean_abs_alpha_err, color=cc[r], width=0.6)
        ax[3].errorbar(k, h.mean_abs_alpha_err, yerr=[[h.mean_abs_alpha_err - h.lo], [h.hi - h.mean_abs_alpha_err]],
                       color="k", capsize=2, lw=0.8)
        ax[3].text(k, h.hi + 0.004, f"valid {h.valid_frac:.2f}", ha="center", fontsize=5.5)
    ax[2].plot([0.1, 0.7], [0.1, 0.7], ls=":", color="0.5", lw=0.7, label="1:1")
    for a in (ax[0], ax[1], ax[3]):
        a.set_xticks([0, 1]); a.set_xticklabels(runs); a.set_xlim(-0.6, 1.6)
    ax[0].set_ylabel("Mean track distance,\nmean ± 95 % CI (grid cells)"); ax[0].set_title(f"T1 ({int(t1.n.iloc[0])} configs)")
    ax[0].set_ylim(0, float(t1.hi.max()) * 1.25)
    ax[1].set_ylabel("Steps on $V_2$ (wrong variable),\nmean fraction"); ax[1].set_title("T1: wrong-variable steps")
    ax[1].set_ylim(0, float(t1.parent2_frac.max()) * 1.2)
    ax[2].set_xlabel("Prescribed α"); ax[2].set_ylabel("Recovered α, median"); ax[2].set_xticks([0.2, 0.6])
    ax[2].set_title(f"T2-EN ({int(t2.n.iloc[0])} configs)"); ax[2].legend(loc="upper left", fontsize=5.5)
    ax[3].set_ylabel("|recovered − true α|,\nmean ± 95 % CI"); ax[3].set_title("T2-EN: α error")
    ax[3].set_ylim(0, float(t2.hi.max()) * 1.3)
    S.panel_labels(ax, x=-0.32)
    fig.tight_layout(w_pad=1.2)
    print(S.save(fig, "S17_AGL_vs_EN", "S17_AGL_vs_EN"), flush=True)


# ============================================================== S16, S18, S19 (N5_T7_supp)
def run_S16():
    import N5_T7_supp as T7
    T7.OUT = SUPP
    T7.run_F()


def run_S18(nproc):
    import N5_T7_supp as T7
    T7.OUT = SUPP
    T7.run_K("night5", nproc)


def run_S19():
    import N5_T7_supp as T7
    T7.OUT = SUPP
    T7.run_G(); T7.run_H()


# ============================================================== R1
def run_R1():
    from N5_T0 import alpha_fit
    x = pd.read_csv(R1 / "T2r_multivariate_en" / "results_alphas_final.csv")
    subs = [("Linear rule (final)", x[x.prob_rule == "linear"], "#0072B2", "o", "-"),
            ("Mixed rules (all configs)", x, "#000000", "s", "--"),
            ("Softmax, β ≤ 1", x[(x.prob_rule == "softmax") & (x.beta_softmax <= 1.0)], "#D55E00", "^", "-"),
            ("Softmax, β 2–8", x[(x.prob_rule == "softmax") & (x.beta_softmax > 1.5) & (x.beta_softmax <= 8)], "#E69F00", "v", "-"),
            ("Softmax, β 16–64", x[(x.prob_rule == "softmax") & (x.beta_softmax > 15)], "#CC79A7", "D", "-")]
    rows, per = [], []
    for nm, s, *_ in subs:
        f = alpha_fit(s, nm); f["source"] = "T2r_multivariate_en (night 1)"; rows.append(f)
        s = s[s.estimated_alpha_density.notna()]
        for a, g in s.groupby("alpha_mix"):
            q = g.estimated_alpha_density.quantile([0.25, 0.5, 0.75]).values
            per.append(dict(subset=nm, alpha_mix=a, n=len(g), q25=q[0], median=q[1], q75=q[2]))
    Fz, P = pd.DataFrame(rows), pd.DataFrame(per)
    Fz.to_csv(TAB / "R1_alpha_slope_by_rule.csv", index=False); P.to_csv(TAB / "R1_alpha_by_rule_levels.csv", index=False)
    md(Fz[["subset", "n_configs", "slope", "intercept", "r_median", "median_abs_err"]], TAB / "R1_alpha_slope_by_rule.md",
       head="R1: slope of the median recovered alpha (density) vs prescribed alpha (0.2, 0.4, 0.6, 0.8), T2-EN night-1 "
       "re-run, by probability rule (N5_T0.alpha_fit). Paper value 0.84.\n\n")
    print(Fz.round(3).to_string(), flush=True)
    S.set_style()
    fig, ax = plt.subplots(figsize=(S.SINGLE, 7.5 * S.CM))
    ax.plot([0.15, 0.85], [0.15, 0.85], ls=":", color="0.6", lw=0.7, label="1:1")
    for k, (nm, s, c, m, ls) in enumerate(subs):
        g = P[P.subset == nm]; f = Fz[Fz.subset == nm].iloc[0]
        o = (k - 2) * 0.008
        ax.errorbar(g.alpha_mix + o, g["median"], yerr=[g["median"] - g.q25, g.q75 - g["median"]], color=c, marker=m,
                    ms=3.2, ls=ls, lw=0.9, capsize=1.2, elinewidth=0.5,
                    label=f"{nm}: slope {f.slope:.3f} (n = {int(f.n_configs)})")
    ax.set_xticks([0.2, 0.4, 0.6, 0.8]); ax.set_xlim(0.12, 0.88); ax.set_ylim(0.1, 0.9)
    ax.set_xlabel("Prescribed α (share of V$_1$)"); ax.set_ylabel("Recovered α (density), median and IQR")
    ax.legend(loc="upper left", fontsize=5.3)
    ax.set_title("T2-EN: α recovery by probability rule (paper slope 0.84)", fontsize=6.8)
    fig.tight_layout()
    print(S.save(fig, "R1_alpha_by_rule", "R1_alpha_slope_by_rule"), flush=True)


# ============================================================== T8 refit table and R2
def run_T8():
    import N5_T8
    N5_T8.OUT = TAB
    N5_T8.main()
    T = pd.read_csv(TAB / "T8_refit_table.csv")
    T["status"] = np.where(T.n_configs == 0, "not run", "ok")
    T.to_csv(TAB / "T8_refit_table.csv", index=False)
    cols = [c for c in ["case", "engine", "strength", "subset", "n_configs", "status", "retained_frac", "median_track_dist",
                        "mean_parent2_frac", "alphas", "slope", "r_median", "median_abs_alpha_err",
                        "median_sec_per_config"] if c in T]
    md(T[cols], TAB / "T8_refit_table.md", "{:.3g}", head="Refit table (N5_T8.py): own (engine) vs refit (common "
       "standardized least-squares) coefficients. T2-DYNOTEARS: not run (about 2 h per trajectory at the T2 geometry). "
       "Final method = rows with subset linear; the T2-EN final pair = night-5 p03 (own) vs night-5 t8 refit (same "
       "settings, linear, alpha 0.2/0.6).\n\n")


def run_R2():
    f = TAB / "T8_refit_table.csv"
    if not f.exists():
        run_T8()
    T = pd.read_csv(f)
    eng = ["granger", "pcmci", "dynotears"]
    ec = {"granger": "#0072B2", "pcmci": "#CC79A7", "dynotears": "#E69F00"}
    el = {"granger": "Elastic-Net", "pcmci": "PCMCI", "dynotears": "DYNOTEARS"}

    def cell(case, e, mode):
        g = T[(T.case == case) & (T.engine == e)]
        if case == "T2" and e == "granger":
            g = g[g.strength.str.startswith("own (night-5" if mode == "own" else "refit (night-5")]
        else:
            g = g[g.strength == mode]
        g = g[g.subset == "linear"]
        return g.iloc[0] if len(g) else None

    S.set_style()
    specs = [("T1", "median_track_dist", "Median track distance\n(grid cells)", "T1 track error"),
             ("T1", "mean_parent2_frac", "Steps on $V_2$ (wrong variable),\nmean fraction", "T1 wrong var."),
             ("T1", "retained_frac", "Retained configs (fraction)", "T1 retained"),
             ("T2", "slope", "Slope, recovered vs\nprescribed α (median)", "T2 α slope"),
             ("T2", "median_abs_alpha_err", "|recovered − true α|,\nmedian", "T2 α error")]
    fig, axs = plt.subplots(1, 5, figsize=(S.DOUBLE, 6.2 * S.CM))
    for ax, (case, col, yl, tt) in zip(axs, specs):
        vals = []
        for k, e in enumerate(eng):
            for j, mode in enumerate(("own", "refit")):
                r = cell(case, e, mode)
                xk = k + (j - 0.5) * 0.36
                if r is None or not np.isfinite(r.get(col, np.nan)):
                    if j == 1:
                        ax.text(k, 0.02, "not run", rotation=90, ha="center", va="bottom", fontsize=5.5, color="#555555",
                                transform=ax.get_xaxis_transform())
                    continue
                v = float(r[col]); vals.append(v)
                ax.bar(xk, v, width=0.34, color=ec[e] if mode == "own" else "white", edgecolor=ec[e], lw=0.8,
                       hatch=None if mode == "own" else "////")
                if case == "T2":
                    ax.text(xk, v if v >= 0 else 0, f"n {int(r.n_configs)}", ha="center", va="bottom", fontsize=4.8)
        ax.axhline(0, color="k", lw=0.5)
        if col == "slope":
            ax.axhline(0.84, color="0.5", lw=0.6, ls=":"); ax.text(2.45, 0.86, "paper 0.84", fontsize=5, ha="right", color="0.4")
        ax.set_xticks(range(3)); ax.set_xticklabels([el[e] for e in eng], rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylabel(yl); ax.set_title(tt, loc="right")
        lo = min(0, min(vals)) if vals else 0
        ax.set_ylim(lo * 1.15, max(vals) * 1.2 if vals else 1)
    hd = [Patch(facecolor="0.4", edgecolor="0.4", label="own coefficients (default)"),
          Patch(facecolor="white", edgecolor="0.4", hatch="////", label="common refit (opt-in)")]
    fig.legend(handles=hd, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    S.panel_labels(axs, x=-0.38, y=1.13)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.9)
    print(S.save(fig, "R2_refit", "R2_refit_engines"), flush=True)


# ============================================================== Table S1
def run_TS1():
    from N5_T4_analysis import T4, lin
    rows = []

    def summ(name, vset, p, ns, r, sub, n_cfg, note=""):
        nf = p * (2 * r + 1) ** 2
        q = ns / nf
        rows.append(dict(experiment=name, variable_set=vset, p=p, configs=sub, n=n_cfg,
                         n_samples_median=float(np.median(ns)), n_features_median=float(np.median(nf)),
                         ratio_median=float(np.median(q)), ratio_p10=float(np.percentile(q, 10)),
                         ratio_p90=float(np.percentile(q, 90)), ratio_min=float(q.min()), ratio_max=float(q.max()),
                         frac_ratio_ge1=float(np.mean(q >= 1)), note=note))

    names = dict(t1="T1 (univariate)", t2="T2-EN (multivariate)", debby="Debby", pinatubo="Pinatubo", pnw="PNW 2021")
    for c in ("t1", "t2", "debby", "pinatubo", "pnw"):
        f = T4 / "night5" / f"configs_{c}.csv"
        if not f.exists():
            f = T4 / "night1" / f"configs_{c}.csv"
        t = lin(pd.read_csv(f, low_memory=False))
        if "method" in t:
            t = t[t["method"] == "granger"]
        t["adm"] = t["adm_ret"].astype(bool)
        r = t["radius"].astype(int).values
        p0 = t["n_features"].values / (2 * r + 1) ** 2
        assert np.allclose(p0, P_ORIG[c]), (c, np.unique(p0))
        src = f"{f.parent.name}/{f.name}"
        for sub, m in (("all linear", np.ones(len(t), bool)), ("linear + harmonized", t["adm"].values)):
            summ(names[c], "original (paper)", P_ORIG[c], t["n_samples"].values[m], r[m], sub, int(m.sum()), src)
            if c in P_FINAL:
                summ(names[c], f"final: {SETS[c]}", P_FINAL[c], t["n_samples"].values[m], r[m], sub, int(m.sum()),
                     "same configs (same trial seeds); retention of the original-set run" if sub != "all linear"
                     else "same configs (same trial seeds as the night-6 final-set runs)")
    for pdir in sorted((N5 / "t3").glob("p*")):
        f = pdir / "results_alphas_final.csv"
        if not f.exists():
            continue
        x = pd.read_csv(f).drop_duplicates("trial_id")
        if "t7_n_samples" in x:
            p = int(x.t7_p.iloc[0]); ns = x.t7_n_samples.values.astype(float)
            assert np.allclose(x.t7_n_features.values, p * (2 * x.radius.values + 1) ** 2)
            note = f"night5/t3/{pdir.name} (stored t7_n_samples / t7_n_features)"
        else:   # p = 3 run (no T7 columns): same formula, T2 start point (100, 100) on the synthetic grid
            from N5_T4_data import SYN_GRID, n_over_p, tw_steps
            p = int(pdir.name.lstrip("p"))
            ns = np.array([n_over_p(float(b), int(r_), tw_steps(w), p, [(SYN_GRID[0], SYN_GRID[1], (100.0, 100.0))])[0]
                           for b, r_, w in zip(x.box_size, x.radius, x.timewindow)])
            note = f"night5/t3/{pdir.name} (formula; no stored T7 columns)"
        summ("T2-EN synthetic extension (S10)", f"p = {p}", p, ns, x.radius.values.astype(int),
             "all (linear forced)", len(x), note)
    D = pd.DataFrame(rows)
    D.to_csv(TAB / "TableS1_samples_per_feature.csv", index=False)
    md(D, TAB / "TableS1_samples_per_feature.md", "{:.3g}",
       head="Table S1: Elastic-Net design size per local fit at the starting point, N = (n_y − 2r)(n_x − 2r)(tw − 1) "
       "samples and p(2r + 1)^2 features (castle_core.mv_CaStLe_GRANGER; formula of J_sensitivity.table_s1 / "
       "N5_T4_data.n_over_p). Linear-rule configs. The final sets use the same configs as the original sets (the "
       "night-6 final-set runs use the same trial seeds), so N is unchanged and only p changes. Synthetic extension "
       "values are the stored per-config numbers.\n\n")
    print(D.round(3).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["TS1", "T8", "R1", "R2", "S13", "S14", "S15", "S16", "S17", "S19"])
    ap.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()
    for d in (FIG6, TAB, SUPP):
        d.mkdir(parents=True, exist_ok=True)
    S.set_output(FIG6, RENAME)
    for w in a.which:
        t0 = time.time()
        print(f"===== {w}", flush=True)
        fn = {"S13": run_S13, "S14": run_S14, "S15": run_S15, "S16": run_S16, "S17": run_S17,
              "S18": lambda: run_S18(a.nproc), "S19": run_S19, "R1": run_R1, "R2": run_R2, "TS1": run_TS1,
              "T8": run_T8, "T5": run_S17}[w]
        try:
            fn()
        except Exception as e:  # one failing item must not block the others
            import traceback; traceback.print_exc()
            print("FAILED", w, e, flush=True)
        print(f"===== {w} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
