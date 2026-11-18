"""Night-6 T4: synthetic p-curve to p = 35 and the fixed-ratio curve (figure S10; pre-declared rule 8).

Runs (T2-EN, 50 configs x alpha {0.2, 0.6}, M 20, linear rule, composition = one cycle noise, response, moving,
delayed (beta 0.2), nonlinear continued):
  fixed window: night5/t3/p{03,05,08,12,16} (night 5, rev1-fixes-v13; reproduced byte-identically at
                rev1-fixes-v17, checked on config 0 at p = 3) + night6/t4syn/p{20,24,28,35};
  fixed ratio:  night6/t4syn/fr_p{03,...,35} (--timewindow-scale-to-p3 --t-pad 60: each config's drawn W grown to
                W_p = 1 + round((W - 1) p / 3) so that N_samples / N_features stays at its p = 3 value; the series
                has 60 days prepended (blobs at their first position), the same for every fr run, so fr_p03 vs p03
                measures the effect of the longer series alone).
N_samples = (ny - 2r)(nx - 2r)(W - 1), N_features = p (2r + 1)^2 at the event box (summarize_all_vars,
scale_timewindow_to_p3); the table T4syn_samples.csv is computed from the 50 drawn configs (no run needed).
Metrics (N5_T3.per_config): share of trajectory density per class, alpha recovery A / (A + C) and its absolute
error, completion, link-level selection per class (TST_LINKSTATS). Means over configs, 95 % bootstrap CIs
(10,000 resamples of configs, seed 0; alphas pooled unless split).

python N6_T4_synthetic.py   (PBS only) -> night6/analysis/T4syn_*.csv, figures_v6/synthetic/S10_synthetic_extension.*
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
from A4_twins import boot_ci  # noqa: E402
from N5_T3 import CLASSES, _mix_names, cls, completion_from_members, linkstats, per_config  # noqa: E402
from N5_T4_data import SYN_GRID, n_over_p, tw_steps  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
N5, N6 = R1 / "night5", R1 / "night6"
OUT = N6 / "analysis"
FIG6 = R1 / "figures_v6"
PS = [3, 5, 8, 12, 16, 20, 24, 28, 35]
CYCLE = ["noise", "response", "moving", "delayed", "nonlinear"]
MIX32 = [CYCLE[i % 5] for i in range(32)]
CURVES = {"fw": "Fixed window (W drawn)", "fr": "Fixed ratio (W grown, N/p as p = 3)"}
CURVE_STYLE = {"fw": dict(color="#000000", marker="o"), "fr": dict(color="#D55E00", marker="s")}


def run_dir(curve, p):
    if curve == "fw":
        return (N5 / "t3" if p <= 16 else N6 / "t4syn") / f"p{p:02d}"
    return N6 / "t4syn" / f"fr_p{p:02d}"


def load(d):
    for nm in ("results_alphas_final.csv", "results_alphas_snapshot.csv"):
        if (d / nm).exists():
            return pd.read_csv(d / nm)
    return None


def var_names(p):
    return ["A", "B", "C"] + _mix_names(MIX32[: p - 3])


def samples_table():
    """N_samples, N_features and their ratio for the 50 drawn configs at every p, both curves."""
    spec = importlib.util.spec_from_file_location("syn", Path(__file__).resolve().parents[1] / "Scripts_Paper"
                                                  / "3_SyntheticRuns_Multivariate.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    jobs = m.build_multivar_jobs(50, seed=11)
    grid = [(SYN_GRID[0], SYN_GRID[1], (100.0, 100.0))]
    rows = []
    for p in PS:
        for curve, jj in (("fw", jobs), ("fr", m.scale_timewindow_to_p3(jobs, p))):
            for (tid, t), (_, t0) in zip(jj, jobs):
                ns, nf, r = n_over_p(float(t["box_size"]), int(t["radius"]), tw_steps(t["timewindow"]), p, grid)
                ns3, nf3, r3 = n_over_p(float(t0["box_size"]), int(t0["radius"]), tw_steps(t0["timewindow"]), 3, grid)
                rows.append(dict(curve=curve, p=p, trial_id=tid, W_drawn=tw_steps(t0["timewindow"]),
                                 W=tw_steps(t["timewindow"]), box_size=t["box_size"], radius=t["radius"],
                                 n_samples=ns, n_features=nf, ratio=r, ratio_p3=r3, ratio_rel_p3=r / r3))
    T = pd.DataFrame(rows)
    agg = T.groupby(["curve", "p"]).agg(W_min=("W", "min"), W_max=("W", "max"),
                                        n_samples_median=("n_samples", "median"), n_samples_min=("n_samples", "min"),
                                        n_samples_max=("n_samples", "max"), n_features_median=("n_features", "median"),
                                        ratio_median=("ratio", "median"), ratio_min=("ratio", "min"),
                                        ratio_max=("ratio", "max"), ratio_rel_p3_min=("ratio_rel_p3", "min"),
                                        ratio_rel_p3_max=("ratio_rel_p3", "max")).reset_index()
    return T, agg


def summarize(curve, p, pc):
    out = []
    for a in ("pooled", 0.2, 0.6):
        g = pc if a == "pooled" else pc[np.isclose(pc["alpha"], a)]
        per_cfg = g.groupby("trial_id").mean(numeric_only=True)
        r = dict(curve=curve, p=p, alpha=a, n_cfg=len(per_cfg))
        for k in CLASSES + ["alpha_err", "completion", "elapsed_sec", "ratio"]:
            col = f"share_{k}" if k in CLASSES else k
            if col in per_cfg and per_cfg[col].notna().any():
                mm, lo, hi = boot_ci(per_cfg[col].to_numpy())
                r[col] = mm; r[col + "_lo"] = lo; r[col + "_hi"] = hi
                r[col + "_median"] = float(per_cfg[col].median())
        out.append(r)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T, agg = samples_table()
    T.to_csv(OUT / "T4syn_samples_per_config.csv", index=False); agg.to_csv(OUT / "T4syn_samples.csv", index=False)
    rows, summ, links = [], [], []
    for curve in CURVES:
        for p in PS:
            d = run_dir(curve, p)
            x = load(d)
            if x is None:
                print("missing", d); continue
            pc = per_config(p, x)
            cm = completion_from_members(d, 30)
            if cm is not None and pc["completion"].isna().all():
                pc = pc.drop(columns="completion").merge(cm.rename(columns={"alpha_mix": "alpha"}),
                                                         on=["trial_id", "alpha"], how="left")
            jf = d / "results_jobs_final.csv"
            if not jf.exists():
                jf = d / "results_jobs_snapshot.csv"
            if jf.exists():
                j = pd.read_csv(jf)
                if "elapsed_sec" in j:
                    pc = pc.merge(j[["trial_id", "elapsed_sec"]], on="trial_id", how="left")
            tt = T[(T.curve == curve) & (T.p == p)][["trial_id", "W", "n_samples", "n_features", "ratio"]]
            pc = pc.merge(tt, on="trial_id", how="left")
            w_run = x.groupby("trial_id")["timewindow"].first().map(tw_steps)
            bad = (w_run - pc.groupby("trial_id")["W"].first()).abs().max()
            if bad > 0:
                print(f"WARNING {curve} p={p}: run windows differ from the table by up to {bad} d")
            pc["curve"] = curve; pc["p"] = p
            rows.append(pc)
            summ += summarize(curve, p, pc)
            ls = linkstats(d, var_names(p))
            if ls:
                links.append(dict(curve=curve, p=p, **ls))
    if not rows:
        return
    P = pd.concat(rows); Sm = pd.DataFrame(summ); Lk = pd.DataFrame(links)
    P.to_csv(OUT / "T4syn_per_config.csv", index=False); Sm.to_csv(OUT / "T4syn_summary.csv", index=False)
    Lk.to_csv(OUT / "T4syn_linkstats.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(agg.round(3).to_string())
        cols = ["curve", "p", "alpha", "n_cfg", "share_true", "share_true_lo", "share_true_hi"] \
            + [c for c in Sm.columns if c.startswith("share_") and not c.endswith(("_lo", "_hi", "_median"))
               and c != "share_true"] + ["alpha_err", "alpha_err_lo", "alpha_err_hi", "completion", "ratio_median"]
        print(Sm[[c for c in cols if c in Sm]].round(3).to_string()); print(Lk.round(3).to_string())
    figure(Sm, Lk)


def figure(Sm, Lk):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    S.set_style()
    if hasattr(S, "set_output"):
        S.set_output(FIG6)
    else:
        S.FIG5 = FIG6
    fig = plt.figure(figsize=(S.DOUBLE, 0.80 * S.DOUBLE))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.28)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    gd = gs[1, 1].subgridspec(2, 1, hspace=0.12)
    ax_d1 = fig.add_subplot(gd[0]); ax_d2 = fig.add_subplot(gd[1], sharex=ax_d1)
    fw = Sm[(Sm["curve"] == "fw") & (Sm["alpha"] == "pooled")].sort_values("p")
    # (a) stacked shares vs p (fixed window, alphas pooled), CI on the true-parent share
    xs = np.arange(len(fw)); bottom = np.zeros(len(fw))
    for k in ["true", "target", "noise", "response", "moving", "delayed", "nonlinear"]:
        c = f"share_{k}"
        if c not in fw:
            continue
        y = fw[c].fillna(0).to_numpy()
        ax_a.bar(xs, y, 0.65, bottom=bottom, color=S.CLASS_COLORS.get(k, "#DDDDDD"), lw=0,
                 label=S.CLASS_LABELS.get(k, "Target B (own density)"))
        if k == "true":
            ax_a.errorbar(xs, y, yerr=[y - fw[c + "_lo"].to_numpy(), fw[c + "_hi"].to_numpy() - y], fmt="none",
                          color="k", lw=0.6, capsize=1.5)
        bottom += y
    ax_a.set_xticks(xs); ax_a.set_xticklabels(fw["p"].astype(int)); ax_a.set_xlabel("Number of variables p")
    ax_a.set_ylabel("Share of trajectory density"); ax_a.set_ylim(0, 1.0)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=5.2, ncol=3, columnspacing=0.8)
    # (b) alpha error vs p, per alpha (fixed window)
    for a, mk, cc in ((0.2, "o", "#0072B2"), (0.6, "s", "#CC79A7")):
        q = Sm[(Sm["curve"] == "fw") & (Sm["alpha"] == a)].sort_values("p")
        if len(q):
            ax_b.errorbar(q["p"], q["alpha_err"], yerr=[q["alpha_err"] - q["alpha_err_lo"], q["alpha_err_hi"] - q["alpha_err"]],
                          marker=mk, ms=3, color=cc, lw=1, capsize=1.5, label=f"α = {a}")
    ax_b.errorbar(fw["p"], fw["alpha_err"], yerr=[fw["alpha_err"] - fw["alpha_err_lo"], fw["alpha_err_hi"] - fw["alpha_err"]],
                  marker="D", ms=2.5, color="0.45", lw=0.8, ls="--", capsize=1.2, label="pooled")
    ax_b.set_xlabel("Number of variables p"); ax_b.set_ylabel("|α recovered − α| (mean over configs)")
    ax_b.set_xticks(PS); ax_b.set_ylim(0, None); ax_b.legend(loc="lower right", ncol=3)
    # (c) per-variable selection rate vs per-variable density share (fixed window)
    mks = dict(zip(PS, ["o", "s", "^", "D", "v", "P", "X", "*", "h"]))
    lk = Lk[Lk["curve"] == "fw"] if len(Lk) else Lk
    for _, r in lk.iterrows():
        srow = fw[fw["p"] == r["p"]]
        if not len(srow):
            continue
        srow = srow.iloc[0]
        for k in ["true", "noise", "response", "moving", "delayed", "nonlinear"]:
            if f"sel_{k}" in r and np.isfinite(r[f"sel_{k}"]) and f"share_{k}" in srow and np.isfinite(srow[f"share_{k}"]):
                nvar = 2 if k == "true" else max(1, sum(1 for t in MIX32[: int(r["p"]) - 3] if t == k))
                ax_c.scatter(r[f"sel_{k}"], srow[f"share_{k}"] / nvar, s=14, marker=mks[int(r["p"])],
                             color=S.CLASS_COLORS[k], edgecolor="k", lw=0.3)
    h = [Line2D([], [], ls="", marker="o", ms=4, color=S.CLASS_COLORS[k], label=S.CLASS_LABELS[k]) for k in
         ["true", "noise", "response", "moving", "delayed", "nonlinear"]]
    h += [Line2D([], [], ls="", marker=mk, ms=4, color="0.5", label=f"p = {p}") for p, mk in mks.items()]
    ax_c.legend(handles=h, loc="center", bbox_to_anchor=(0.5, 0.6), fontsize=5, ncol=2)
    ax_c.set_xlabel("Link-level selection rate (per variable)"); ax_c.set_ylabel("Density share per variable")
    ax_c.set_xlim(-0.02, 1.02); ax_c.set_ylim(0, None)
    # (d) fixed ratio vs fixed window: true-parent share (top) and alpha error (bottom), alphas pooled
    for curve, lab in CURVES.items():
        q = Sm[(Sm["curve"] == curve) & (Sm["alpha"] == "pooled")].sort_values("p")
        if not len(q):
            continue
        st = CURVE_STYLE[curve]
        for ax, k in ((ax_d1, "share_true"), (ax_d2, "alpha_err")):
            ax.errorbar(q["p"], q[k], yerr=[q[k] - q[k + "_lo"], q[k + "_hi"] - q[k]], ms=3, lw=1, capsize=1.5,
                        label=lab, **st)
    ax_d1.set_ylabel("True-parent\nshare"); ax_d1.set_ylim(0, 1.0)
    ax_d2.set_ylabel("|α rec. − α|"); ax_d2.set_ylim(0, None); ax_d2.set_xlabel("Number of variables p")
    ax_d2.set_xticks(PS); plt.setp(ax_d1.get_xticklabels(), visible=False)
    ax_d1.legend(loc="lower left", fontsize=5.2)
    S.panel_labels([ax_a, ax_b, ax_c], x=-0.14)
    S.panel_labels([ax_d1], x=-0.14, start=3)
    print("fig", S.save(fig, "synthetic", "S10_synthetic_extension"))


if __name__ == "__main__":
    main()
