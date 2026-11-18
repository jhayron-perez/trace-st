"""Night-5 T4 figures (R1 Major 2): N/p bins, sensitivity (partial dependence), scale rules, LOCO, retention.

Reads revision1/night5/analysis/T4/<source>/ (N5_T4_data.py, N5_T4_analysis.py) and night5/T0_retention_*.csv.
Style: N5_style (PDF + 300-dpi PNG, no green). Linear configs only; share / error over admissible configs
(harmonized retention), completion over all linear configs. Error bars: 95 % bootstrap CI over configs (2000, seed 0).

python N5_T4_figs.py [--source night1|night5]      (PBS only)
Figures: revision1/figures_v5/guidelines/T4_*.pdf/png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from N5_T4_analysis import CASES, NAMES, EDGES, load, lin, boot_ci, T4  # noqa: E402

CASE_COL = dict(t1="#56B4E9", t2="#0072B2", debby="#CC79A7", pinatubo="#D55E00", pnw="#000000")
COMP_COL = "#999999"
SHARE_LAB = dict(t1="A (true parent)", t2="V$_1$+V$_3$ (true)", debby="vo, 5–15 d", pinatubo="SO$_2$, 4–16 d",
                 pnw="Z500+TCWV, 1–10 d")
UNIT = dict(t1="grid units", t2="days", debby="0.25° (h)", pinatubo="4° (d)", pnw="2° (d)")
FOLDER = "guidelines"
LOGB = np.arange(-4.0, 1.01, 0.5)


def ci_points(ax, x, groups, col, marker="o", ls="-", label=None, ms=3, off=0.0):
    xs, m, lo, hi = [], [], [], []
    for xi, v in zip(x, groups):
        if len(v) == 0:
            continue
        a, b, c = boot_ci(v)
        if not np.isfinite(a):
            continue
        xs.append(xi + off); m.append(a); lo.append(a - b if np.isfinite(b) else 0); hi.append(c - a if np.isfinite(c) else 0)
    if xs:
        ax.errorbar(xs, m, yerr=[lo, hi], color=col, marker=marker, ms=ms, lw=0.9, ls=ls, capsize=1.2, elinewidth=0.6,
                    label=label, mfc=col if marker != "s" else "white")


# ------------------------------------------------------------------ figure 1: N/p bins
def fig_np(T, od):
    B = pd.read_csv(od / "T4_np_bins.csv"); tgt = json.load(open(od / "T4_np_target.json"))
    fig, axs = plt.subplots(2, 3, figsize=(S.DOUBLE, 10.5 * S.CM))
    ax = axs.ravel()
    for c in CASES:
        g = B[(B.case == c) & (B.n >= 5)].sort_values("np_median")
        if not len(g):
            continue
        col = CASE_COL[c]; x = g.np_median
        ax[0].plot(x, g.adm_rate, "o-", color=col, ms=3, label=NAMES[c])
        ax[1].errorbar(x, g.share, yerr=[g.share - g.share_lo, g.share_hi - g.share], color=col, marker="o", ms=3,
                       capsize=1.2, elinewidth=0.6)
        ax[2].errorbar(x, g.frac_rank1, yerr=[g.frac_rank1 - g.rank1_lo, g.rank1_hi - g.frac_rank1], color=col,
                       marker="o", ms=3, capsize=1.2, elinewidth=0.6)
        ax[3].plot(x, g.completion, "o-", color=col, ms=3)
        if c in ("t1", "t2"):
            a = ax[4] if c == "t1" else ax[5]
            a.errorbar(x, g.err, yerr=[g.err - g.err_lo, g.err_hi - g.err], color=col, marker="o", ms=3, capsize=1.2,
                       elinewidth=0.6)
    yl = ["Admissible fraction\n(harmonized retention)", "Dominant-variable share\n(admissible configs)",
          "Fraction of configs with the\ndominant variable at rank 1", "Completion (members at full length)",
          "T1 track error vs true path\n(grid units)", "T2 |α$_{est}$ − α|\n(mean over 4 α)"]
    for i, a in enumerate(ax):
        a.set_xscale("log"); a.set_ylabel(yl[i]); a.set_xlabel("Sample-to-feature ratio N/p")
        if np.isfinite(tgt.get("target", np.nan)):
            a.axvline(tgt["target"], color="0.4", ls="--", lw=0.7)
        if i in (0, 1, 2, 3):
            a.set_ylim(-0.03, 1.05)
    h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03))
    ax[0].text(tgt["target"] * 1.15, 0.05, f"target\nN/p = {tgt['target']:g}", fontsize=5.5, color="0.3")
    S.panel_labels(ax, x=-0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.0, h_pad=1.0)
    return S.save(fig, FOLDER, "T4_np_bins")


# ------------------------------------------------------------------ figure 2: scale-free parameters (main)
def _levels(c, L, p):
    if p in ("lambda_a", "l1_ratio"):
        v = np.log10(L[p].astype(float))
        idx = np.digitize(v, LOGB) - 1
        cen = (LOGB[:-1] + LOGB[1:]) / 2
        return [(cen[i], L.index[idx == i]) for i in sorted(set(idx)) if (idx == i).sum() >= 5]
    if p == "score_mode":
        return [(i, L.index[L.score_mode == m]) for i, m in enumerate(["mean", "sum"])]
    if p == "alpha":
        lv = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        return [(i, L.index[np.isclose(L.alpha.astype(float), v)]) for i, v in enumerate(lv)]
    vals = sorted(L[p].astype(float).unique())
    return [(v, L.index[L[p].astype(float) == v]) for v in vals]


def fig_free(T):
    params = ["eps_dbscan", "alpha", "score_mode", "lambda_a", "l1_ratio"]
    xl = dict(eps_dbscan="DBSCAN ε", alpha="Score exponent α", score_mode="Score mode", lambda_a="log$_{10}$ λ$_a$",
              l1_ratio="log$_{10}$ l1 ratio")
    fig, axs = plt.subplots(3, 5, figsize=(S.DOUBLE, 12.5 * S.CM), sharey="row")
    offs = dict(zip(CASES, np.linspace(-0.2, 0.2, 5)))
    for j, p in enumerate(params):
        for c in CASES:
            if c not in T:
                continue
            L = lin(T[c]); col = CASE_COL[c]
            lv = _levels(c, L, p)
            sc = 0.06 if p == "eps_dbscan" else (0.25 if p in ("lambda_a", "l1_ratio") else 1.0)
            o = offs[c] * sc
            x = [a for a, _ in lv]
            ci_points(axs[0, j], x, [L.loc[ix, "completion"].dropna().values for _, ix in lv], col, off=o, label=NAMES[c])
            ci_points(axs[1, j], x, [L.loc[ix][L.loc[ix, "adm"]].share.dropna().values for _, ix in lv], col, off=o)
            if c in ("t1", "t2"):
                med = L[L.adm].err.median()
                ci_points(axs[2, j], x, [L.loc[ix][L.loc[ix, "adm"]].err.dropna().values / med for _, ix in lv], col, off=o)
        for i in range(3):
            a = axs[i, j]
            if p == "alpha":
                a.set_xticks(range(9)); a.set_xticklabels(["0", ".5", "1", "2", "4", "8", "16", "32", "64"], fontsize=5.5)
            if p == "score_mode":
                a.set_xticks([0, 1]); a.set_xticklabels(["mean", "sum"]); a.set_xlim(-0.6, 1.6)
        axs[2, j].set_xlabel(xl[p])
    axs[0, 0].set_ylabel("Completion\n(all linear configs)"); axs[1, 0].set_ylabel("Dominant-variable share\n(admissible)")
    axs[2, 0].set_ylabel("Truth error / case median\n(T1, T2; admissible)")
    axs[0, 0].set_ylim(-0.03, 1.05); axs[1, 0].set_ylim(0.2, 1.02)
    for a in axs[2]:
        a.axhline(1, color="0.6", lw=0.5, ls=":")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    S.panel_labels(axs.ravel(), x=-0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=0.6, h_pad=0.8)
    return S.save(fig, FOLDER, "T4_sensitivity_scalefree")


# ------------------------------------------------------------------ figure 3: scale parameters (supplement)
def fig_scale_pd(T):
    params = ["timewindow", "box_size", "radius", "np_ratio"]
    rows = [c for c in CASES if c in T] + ["t1_err", "t2_err"]
    fig, axs = plt.subplots(len(rows), 4, figsize=(S.DOUBLE, 3.0 * len(rows) * S.CM))
    for i, rname in enumerate(rows):
        c = rname.split("_")[0]
        L = lin(T[c]); col = CASE_COL[c]
        for j, p in enumerate(params):
            a = axs[i, j]
            if p == "np_ratio":
                b = pd.cut(L.np_ratio, EDGES, right=False)
                lv = [(float(L.np_ratio[b == k].median()), L.index[b == k]) for k in b.cat.categories if (b == k).sum() >= 5]
                a.set_xscale("log")
            else:
                lv = _levels(c, L, p)
            x = [q for q, _ in lv]
            if rname.endswith("_err"):
                ci_points(a, x, [L.loc[ix][L.loc[ix, "adm"]].err.dropna().values for _, ix in lv], col, marker="D")
            else:
                ci_points(a, x, [L.loc[ix, "completion"].dropna().values for _, ix in lv], COMP_COL, marker="s", ls="--",
                          label="Completion")
                ci_points(a, x, [L.loc[ix][L.loc[ix, "adm"]].share.dropna().values for _, ix in lv], col,
                          label="Dominant share")
                a.set_ylim(-0.03, 1.05)
            if j == 0:
                a.set_ylabel(f"{NAMES[c]}\n" + ("truth error" if rname.endswith("_err") else SHARE_LAB[c]), fontsize=6.5)
            if i == len(rows) - 1 or True:
                a.set_xlabel({"timewindow": f"Window (steps)", "box_size": "Box (deg; synthetic: grid units)",
                              "radius": "Stencil radius r (cells)", "np_ratio": "N/p"}[p], fontsize=6)
            if p in ("timewindow", "radius"):
                a.set_xticks(x)
    axs[0, 0].legend(loc="lower left", fontsize=5.5)
    S.panel_labels(axs[:, 0], x=-0.42, y=1.02)   # one label per row (case)
    fig.tight_layout(w_pad=0.6, h_pad=0.5)
    return S.save(fig, FOLDER, "T4_sensitivity_scale")


# ------------------------------------------------------------------ figure 4: scale rules
def fig_rules(T, od):
    SR = pd.read_csv(od / "T4_scale_rules.csv"); FIT = pd.read_csv(od / "T4_scale_fit.csv")
    SD = json.load(open(T4 / "scale_diagnostics.json"))
    cases = [c for c in CASES if c in set(SR.case)]
    fig, axs = plt.subplots(len(cases), 4, figsize=(S.DOUBLE, 3.1 * len(cases) * S.CM))
    for i, c in enumerate(cases):
        s = SR[SR.case == c].iloc[0]; col = CASE_COL[c]
        # displacement (primary series not stored as a vector except PNW path; show summary bars)
        a = axs[i, 0]
        d = SD[c]["displacement"]
        a.barh([2, 1, 0], [d["max"], d["p90"], d["median"]], color=col, height=0.6)
        if "displacement_check" in SD[c]:
            dc = SD[c]["displacement_check"]
            a.plot([dc["max"], dc["p90"], dc["median"]], [2, 1, 0], "x", color="0.4", ms=4, label="check method")
        a.axvline(s.r_rule, color="0.2", ls="--", lw=0.8)
        a.set_yticks([2, 1, 0]); a.set_yticklabels(["max", "p90", "median"])
        a.set_xlabel("Displacement per step (cells)"); a.set_title(f"{NAMES[c]}: r$_{{rule}}$ = {int(s.r_rule)}", fontsize=6.5)
        for j, p in enumerate(["radius", "box_size", "timewindow"]):
            a = axs[i, j + 1]
            g = FIT[(FIT.case == c) & (FIT.param == p)].sort_values("level")
            x = np.arange(len(g))
            a.bar(x - 0.2, g.frac_of_drawn, width=0.4, color="0.8", label="drawn")
            a.bar(x + 0.2, g.frac_of_adm, width=0.4, color=col, label="admissible")
            a.set_xticks(x); a.set_xticklabels([f"{v:g}" for v in g.level], fontsize=5.5)
            rv = g.rule_value.iloc[0]
            lv = g.level.values
            if np.isfinite(rv):
                if rv < lv.min():
                    xr = -0.6; txt = f"rule {rv:.3g} <"
                elif rv > lv.max():
                    xr = len(lv) - 0.4; txt = f"> rule {rv:.3g}"
                else:
                    xr = np.interp(rv, lv, x); txt = f"rule {rv:.3g}"
                a.axvline(xr, color="0.2", ls="--", lw=0.8)
                a.text(0.02, 0.95, txt, transform=a.transAxes, fontsize=5.5, va="top")
            a.set_ylim(0, 1.0)
            a.set_xlabel({"radius": "r (cells)", "box_size": "Box (deg; synthetic: grid units)", "timewindow": "Window (steps)"}[p])
            if j == 0:
                a.set_ylabel("Fraction of configs")
        if i == 0:
            axs[0, 1].legend(loc="upper right", fontsize=5.5)
    S.panel_labels(axs[:, 0], x=-0.42, y=1.02)   # one label per row (case)
    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    return S.save(fig, FOLDER, "T4_scale_rules")


# ------------------------------------------------------------------ figure 5: LOCO
def fig_loco(od):
    LO = pd.read_csv(od / "T4_loco.csv")
    fig, axs = plt.subplots(1, 2, figsize=(S.DOUBLE, 5.5 * S.CM), gridspec_kw=dict(width_ratios=[1.4, 1]))
    y = np.arange(len(LO))[::-1]
    for k, (_, r) in enumerate(LO.iterrows()):
        col = CASE_COL[r.left_out]
        axs[0].errorbar(r.share_all, y[k] + 0.12, xerr=[[r.share_all - r.share_all_lo], [r.share_all_hi - r.share_all]],
                        fmt="o", color=col, mfc="white", ms=3.5, capsize=1.2, label="all admissible" if k == 0 else None)
        if np.isfinite(r.share_in):
            axs[0].errorbar(r.share_in, y[k] - 0.12, xerr=[[r.share_in - r.share_in_lo], [r.share_in_hi - r.share_in]],
                            fmt="o", color=col, ms=3.5, capsize=1.2, label="inside 4-case box" if k == 0 else None)
        axs[0].text(1.01, y[k], f"n {int(r.n_in)}/{int(r.n_adm)}", fontsize=5.5, va="center",
                    transform=axs[0].get_yaxis_transform())
        if "err_all" in LO and np.isfinite(r.get("err_all", np.nan)):
            axs[1].errorbar(r.err_all / r.err_all, y[k] + 0.12, xerr=[[(r.err_all - r.err_all_lo) / r.err_all],
                                                                     [(r.err_all_hi - r.err_all) / r.err_all]],
                            fmt="o", color=col, mfc="white", ms=3.5, capsize=1.2)
            if np.isfinite(r.err_in):
                axs[1].errorbar(r.err_in / r.err_all, y[k] - 0.12, xerr=[[(r.err_in - r.err_in_lo) / r.err_all],
                                                                        [(r.err_in_hi - r.err_in) / r.err_all]],
                                fmt="o", color=col, ms=3.5, capsize=1.2)
    for a in axs:
        a.set_yticks(y); a.set_yticklabels([f"{NAMES[c]} left out" for c in LO.left_out])
    axs[1].set_ylim(axs[0].get_ylim()); keep = [k for k, c in enumerate(LO.left_out) if c in ("t1", "t2")]
    axs[1].set_yticks(y[keep]); axs[1].set_yticklabels([f"{NAMES[LO.left_out.iloc[k]]} left out" for k in keep])
    axs[0].set_xlabel("Dominant-variable share (left-out case)"); axs[0].legend(loc="lower left", fontsize=5.5)
    axs[1].set_xlabel("Truth error / full-ensemble mean (T1, T2)"); axs[1].axvline(1, color="0.6", ls=":", lw=0.6)
    S.panel_labels(axs, x=-0.32)
    fig.tight_layout(w_pad=1.5)
    return S.save(fig, FOLDER, "T4_loco")


# ------------------------------------------------------------------ figure 6: retention
def fig_retention(od):
    CNT = pd.read_csv(od / "T4_retention_counts_all.csv"); SW = pd.read_csv(od / "T4_retention_sweep_all.csv")
    fig = plt.figure(figsize=(S.DOUBLE, 11.5 * S.CM))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.9, bottom=0.09, top=0.88, wspace=0.45, hspace=0.6)
    a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(CNT))
    a.bar(x - 0.3, CNT.n_old / CNT.n_total, 0.2, color="0.75", label="paper rule, all")
    a.bar(x - 0.1, CNT.n_harmonized / CNT.n_total, 0.2, color="#0072B2", label="harmonized, all")
    a.bar(x + 0.1, CNT.n_old_linear / CNT.n_linear, 0.2, color="0.75", hatch="////", edgecolor="0.4", lw=0.3,
          label="paper rule, linear")
    a.bar(x + 0.3, CNT.n_harmonized_linear / CNT.n_linear, 0.2, color="#0072B2", hatch="////", edgecolor="white", lw=0.3,
          label="harmonized, linear")
    for k, r in CNT.iterrows():
        pass
    a.set_xticks(x); a.set_xticklabels([f"{NAMES[c]}\n{int(r.n_old)}→\n{int(r.n_harmonized)}" for c, (_, r) in zip(CNT.case, CNT.iterrows())], fontsize=5)
    a.set_ylabel("Fraction of configs retained"); a.set_ylim(0, 1.5); a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a.legend(loc="upper left", fontsize=5, ncol=2, columnspacing=0.6, handlelength=1.4)   # inside: no overlap with b
    axs = [a]
    for k, c in enumerate([c for c in CASES if c in set(SW.case)]):
        ax = fig.add_subplot(gs[(k + 1) // 3, (k + 1) % 3])
        g = SW[SW.case == c].pivot_table(index="min_member_frac", columns="min_len_frac", values="n_retained_linear")
        n = SW[SW.case == c].n_linear.iloc[0]
        im = ax.imshow(g.values / n, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for ii in range(g.shape[0]):
            for jj in range(g.shape[1]):
                v = g.values[ii, jj]
                ax.text(jj, ii, f"{int(v)}", ha="center", va="center", fontsize=5.5,
                        color="white" if v / n > 0.6 else "black",
                        fontweight="bold" if (abs(g.index[ii] - 0.667) < 0.01 and abs(g.columns[jj] - 0.8) < 1e-6) else None)
        ax.set_xticks(range(g.shape[1])); ax.set_xticklabels([f"{int(v * 100)}" for v in g.columns])
        ax.set_yticks(range(g.shape[0])); ax.set_yticklabels([f"{v * 100:.0f}" for v in g.index])
        ax.set_xlabel("Median length ≥ % of N"); ax.set_ylabel("Members successful ≥ %")
        ax.set_title(f"{NAMES[c]} (linear, n = {int(n)})", fontsize=6.5)
        axs.append(ax)
    cax = fig.add_axes([0.93, 0.15, 0.012, 0.3]); cb = fig.colorbar(im, cax=cax); cb.set_label("Fraction retained")
    S.panel_labels(axs, x=-0.3)
    return S.save(fig, FOLDER, "T4_retention")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="night1")
    a = ap.parse_args()
    S.set_style()
    od = T4 / a.source
    T = load(a.source)
    for f in (lambda: fig_np(T, od), lambda: fig_free(T), lambda: fig_scale_pd(T), lambda: fig_rules(T, od),
              lambda: fig_loco(od), lambda: fig_retention(od)):
        try:
            print("saved", f(), flush=True)
        except Exception as e:  # keep going: one failing figure must not block the others
            import traceback; traceback.print_exc()
            print("FAILED", e, flush=True)


if __name__ == "__main__":
    main()
