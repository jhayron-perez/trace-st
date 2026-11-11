"""Task I (R1 minor 2): significance of the variable shares in the real cases.

Per retained configuration, the normalized integrated density (share) of each variable in a window
= integrated box-footprint density of that variable / sum over variables (as Fig. 6B of the paper,
9_AnalysisHW_Figs45_v2 compute_variable_importance_by_run; Debby / Pinatubo density_range columns as
used by Figs 4C / 5C in 7_Analysis_DebbyPinatubo_Fig3_v2, normalized per configuration).
- 95 % percentile bootstrap CI of the mean share over configurations (10,000 resamples, seed 0;
  the same resamples are used for all variables of a case x window, so paired differences are valid).
- Paired two-sided Wilcoxon signed-rank test for every pair of variables (per-config shares),
  Holm correction within case x window.
Windows: stored ones (Debby density_range 96-360 h, Pinatubo 4-16 d, PNW days 1-10/11-20/21-30) read
from the run outputs; sub-windows recomputed from bundles by I_step_densities.py
(Debby 0-24 / 24-96 h as in the supplementary time-split figure, with step 0 skipped, and 120-360 h;
Pinatubo 1-3 / 3-8 / 8-16 d; PNW days 1-10 without the start step, see below).
PNW: step 0 of every member is the starting box with parent = Z500 (the child variable), so the stored
days 1-10 shares include one Z500 step per member by construction; the "excl. start" variant drops it.
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

TASK = "I_significance"
OUT = C.outdir(TASK)


def _shares(M):
    M = np.asarray(M, float)
    tot = M.sum(axis=1, keepdims=True)
    S = np.where(tot > 0, M / np.where(tot > 0, tot, 1), np.nan)
    return S


def load_step(case):
    z = np.load(OUT / f"step_density_{case}.npz", allow_pickle=True)
    return pd.Series(range(len(z["ids"])), index=z["ids"]), z["D"], list(z["vars"])


def case_windows():
    """Return {case: [(window_label, cfg_ids, raw_matrix (n_cfg x n_vars), source), ...]}."""
    W = {}
    for case in ["debby", "debby_orig", "pinatubo"]:
        k = C.retained_jobs(case)
        names = C.VARS[case]
        ids = k["trial_id"].astype(int).values
        stored = k[[f"density_range::{v}" for v in names]].values.astype(float)
        pos, D, dnames = load_step(case)
        assert dnames == names
        Dk = D[pos.loc[ids].values]
        if case.startswith("debby"):
            lab = "96–360 h"
            sub = [("0–24 h", 1, 24), ("24–96 h", 24, 96), ("120–360 h", 120, 360)]
        else:
            lab = "4–16 d"
            sub = [("1–3 d", 1, 3), ("3–8 d", 3, 8), ("8–16 d", 8, 16)]
        W[case] = [(lab, ids, stored, "stored density_range")]
        for l, a, b in sub:
            W[case].append((l, ids, Dk[:, a:b, :].sum(axis=1), f"recomputed from bundles, steps [{a},{b})"))
    s = C.load_pnw_summary("pnw")
    k = s[retention.pnw_retained(s)]
    ids = k["run_id"].astype(int).values
    names = C.VARS["pnw"]
    W["pnw"] = []
    for lab, w in (("1–10 d", "days_1_10"), ("11–20 d", "days_11_20"), ("21–30 d", "days_21_30")):
        W["pnw"].append((lab, ids, k[[f"{w}_integrated_{v}" for v in names]].values.astype(float),
                         f"stored window_stats {w}"))
    pos, D, _ = load_step("pnw")
    Dk = D[pos.loc[ids].values]
    W["pnw"].append(("1–10 d excl. start", ids, Dk[:, 1:10, :].sum(axis=1), "recomputed, steps [1,10)"))
    return W


def main():
    C.set_style()
    W = case_windows()
    ci_rows, pair_rows, long_rows = [], [], []
    boots_store = {}
    for case, wins in W.items():
        names = C.VARS[case]
        for lab, ids, M, src in wins:
            S = _shares(M)
            good = np.isfinite(S).all(axis=1)
            S, idsg = S[good], ids[good]
            mean, lo, hi, boots = C.bootstrap_mean_ci(S, n_boot=10000, seed=0)
            boots_store[(case, lab)] = (S, boots)
            top = np.argmax(S, axis=1)
            for j, v in enumerate(names):
                ci_rows.append(dict(case=case, window=lab, var=v, n_configs=len(S), mean_share=mean[j],
                                    ci95_lo=lo[j], ci95_hi=hi[j], median_share=np.median(S[:, j]),
                                    frac_configs_top=np.mean(top == j), source=src))
            pr = C.wilcoxon_pairs(S, names)
            pr.insert(0, "window", lab)
            pr.insert(0, "case", case)
            pair_rows.append(pr)
            for i, cid in enumerate(idsg):
                for j, v in enumerate(names):
                    long_rows.append((case, lab, int(cid), v, S[i, j]))
    CI = pd.DataFrame(ci_rows)
    P = pd.concat(pair_rows, ignore_index=True)
    P["significant_holm05"] = P["p_holm"] < 0.05
    CI.to_csv(OUT / "I_bootstrap_ci.csv", index=False)
    P.to_csv(OUT / "I_wilcoxon_all_pairs.csv", index=False)
    pd.DataFrame(long_rows, columns=["case", "window", "cfg", "var", "share"]).to_csv(OUT / "I_shares_per_config.csv", index=False)
    # orient every pair as larger > smaller (by mean share)
    flip = P["mean_diff"] < 0
    P["larger"] = np.where(flip, P["var_b"], P["var_a"])
    P["smaller"] = np.where(flip, P["var_a"], P["var_b"])
    P["mean_larger"] = np.where(flip, P["mean_b"], P["mean_a"])
    P["mean_smaller"] = np.where(flip, P["mean_a"], P["mean_b"])
    P["mean_diff_abs"] = P["mean_diff"].abs()
    P["median_diff_larger_minus_smaller"] = np.where(flip, -P["median_diff"], P["median_diff"])
    P["frac_configs_larger_gt_smaller"] = np.where(flip, P["frac_b_gt_a"], P["frac_a_gt_b"])
    P["mean_and_median_agree"] = P["median_diff_larger_minus_smaller"] >= 0
    P["pair"] = [f"{C.PLAIN.get(a, a)} > {C.PLAIN.get(b, b)}" for a, b in zip(P["larger"], P["smaller"])]
    P.to_csv(OUT / "I_wilcoxon_all_pairs.csv", index=False)
    cols = ["case", "window", "pair", "mean_larger", "mean_smaller", "mean_diff_abs",
            "median_diff_larger_minus_smaller", "frac_configs_larger_gt_smaller", "mean_and_median_agree",
            "n", "p_raw", "p_holm"]
    sig = P[P["significant_holm05"]].copy()
    sig[cols].to_csv(OUT / "I_significant_pairs.csv", index=False)
    with open(OUT / "I_significant_pairs.md", "w") as f:
        f.write("Significant pairs: paired two-sided Wilcoxon signed-rank on per-config shares, Holm within "
                "case x window, p_holm < 0.05. `pair` = larger > smaller by mean share. Wilcoxon tests the "
                "location of the paired differences, so where `mean_and_median_agree` is False the median "
                "difference has the opposite sign to the mean.\n\n")
        sig2 = sig[cols].copy()
        for c in ("p_raw", "p_holm"):
            sig2[c] = sig2[c].map(lambda x: f"{x:.2g}")
        f.write(C.to_markdown(sig2))
        f.write("\n\nNon-significant pairs (p_holm >= 0.05):\n\n")
        ns = P[~P["significant_holm05"]][["case", "window", "pair", "mean_diff_abs", "p_holm"]].copy()
        ns["p_holm"] = ns["p_holm"].map(lambda x: f"{x:.2g}")
        f.write(C.to_markdown(ns))

    # ---------------- explicit TCWV question
    ans = {}
    for lab in ["1–10 d", "1–10 d excl. start"]:
        S, boots = boots_store[("pnw", lab)]
        names = C.VARS["pnw"]
        t = names.index("TCWV")
        res = {}
        for j, v in enumerate(names):
            if j == t:
                continue
            d = boots[:, t] - boots[:, j]
            pr = P[(P.case == "pnw") & (P.window == lab) & (((P.var_a == "TCWV") & (P.var_b == v)) | ((P.var_b == "TCWV") & (P.var_a == v)))].iloc[0]
            res[v] = dict(mean_TCWV_minus_var=float(S[:, t].mean() - S[:, j].mean()),
                          boot95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                          p_holm=float(pr.p_holm), frac_configs_TCWV_gt=float(np.mean(S[:, t] > S[:, j])))
        means = {v: float(S[:, j].mean()) for j, v in enumerate(names)}
        ans[lab] = dict(mean_shares=means, n_configs=int(len(S)),
                        TCWV_gt_every_other_wilcoxon_holm=bool(all(r["mean_TCWV_minus_var"] > 0 and r["p_holm"] < 0.05 for r in res.values())),
                        TCWV_gt_every_other_boot_ci_excludes_0=bool(all(r["boot95"][0] > 0 for r in res.values())),
                        vs=res)
    json.dump(ans, open(OUT / "I_tcwv_answer.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(ans, indent=1, ensure_ascii=False))
    print(CI.round(3).to_string())
    print(sig[cols].round(4).to_string())

    # ---------------- figures
    fig_pnw(CI, "pnw", ["1–10 d", "11–20 d", "21–30 d"], "fig6B_pnw_windows_ci",
            "PNW 2021: variable shares by backward window (mean, 95 % bootstrap CI)")
    fig_pnw(CI, "pnw", ["1–10 d", "1–10 d excl. start"], "fig6B_pnw_days1_10_start_step",
            "PNW 1–10 d: the start step (always Z500) carries 0.1 of the share")
    for case, lab, name in (("debby", "96–360 h", "fig4C_debby_vorebuilt_ci"),
                            ("debby_orig", "96–360 h", "fig4C_debby_original_ci"),
                            ("pinatubo", "4–16 d", "fig5C_pinatubo_ci")):
        fig_case_box(boots_store, CI, case, lab, name)
    for case, wins, name in (("debby", ["0–24 h", "24–96 h", "96–360 h", "120–360 h"], "debby_vorebuilt_subwindows_ci"),
                             ("debby_orig", ["0–24 h", "24–96 h", "96–360 h", "120–360 h"], "debby_original_subwindows_ci"),
                             ("pinatubo", ["1–3 d", "3–8 d", "8–16 d", "4–16 d"], "pinatubo_subwindows_ci")):
        fig_pnw(CI, case, wins, name, f"{C.CASE_TITLES[case]}: variable shares by backward window (mean, 95 % CI)")


def _wcolors(n):
    import matplotlib as mpl
    cm = mpl.colormaps["magma"]
    return [cm(x) for x in np.linspace(0.12, 0.72, n)]


def fig_pnw(CI, case, windows, name, title):
    names = C.PLOT_ORDER[case]
    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(names))
    w = 0.8 / len(windows)
    cols = _wcolors(len(windows))
    for i, lab in enumerate(windows):
        d = CI[(CI.case == case) & (CI.window == lab)].set_index("var").loc[names]
        pos = x - 0.4 + w / 2 + i * w
        ax.bar(pos, d.mean_share, width=w * 0.9, color=cols[i], alpha=0.85, label=f"{lab} (n={int(d.n_configs.iloc[0])})")
        ax.errorbar(pos, d.mean_share, yerr=[d.mean_share - d.ci95_lo, d.ci95_hi - d.mean_share], fmt="none",
                    ecolor="0.15", elinewidth=2, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([C.LABELS[v] for v in names])
    ax.set_ylabel("Mean normalized integrated density (share)")
    ax.set_ylim(0, None)
    ax.legend(title="Backward window", loc="upper right")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    C.savefig(fig, TASK, name)
    plt.close(fig)


def fig_case_box(boots_store, CI, case, lab, name):
    names_par = C.VARS[case]
    names = C.PLOT_ORDER[case]
    S, _ = boots_store[(case, lab)]
    fig, ax = plt.subplots(figsize=(16, 9))
    data = [S[:, names_par.index(v)] for v in names]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                    medianprops=dict(color="0.15", lw=1.8), flierprops=dict(markersize=4, alpha=0.5))
    for patch, v in zip(bp["boxes"], names):
        patch.set_facecolor(C.VAR_COLORS[v]); patch.set_alpha(0.45); patch.set_edgecolor(C.VAR_COLORS[v])
    d = CI[(CI.case == case) & (CI.window == lab)].set_index("var").loc[names]
    xx = np.arange(1, len(names) + 1) + 0.34
    ax.errorbar(xx, d.mean_share, yerr=[d.mean_share - d.ci95_lo, d.ci95_hi - d.mean_share], fmt="D", color="0.1",
                ms=9, elinewidth=2.5, capsize=7, label="Mean ± 95 % bootstrap CI")
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([C.LABELS[v] for v in names])
    ax.set_ylabel(f"Share of integrated density, {lab}")
    ax.set_ylim(-0.02, 1.02)
    top = d.mean_share.idxmax()
    ax.set_title(f"{C.CASE_TITLES[case]}: {C.LABELS[top]} dominates {lab} "
                 f"(n={len(S)} retained configs)", loc="left")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    C.savefig(fig, TASK, name)
    plt.close(fig)


if __name__ == "__main__":
    main()
