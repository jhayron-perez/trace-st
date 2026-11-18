"""Night-6 T4: synthetic p-curve to p = 35 and the fixed-ratio curve (figure S10; pre-declared rule 8).

Runs (T2-EN, 50 configs x alpha {0.2, 0.6}, linear rule, composition = one cycle noise, response, moving,
delayed (beta 0.2), nonlinear continued):
  fixed window (M 20): night5/t3/p{03,05,08,12,16} (night 5, rev1-fixes-v13; reproduced byte-identically at
                rev1-fixes-v17 on config 0) + night6/t4syn/p{20,24,28,35}_a0{2,6} (one dir per alpha);
  fixed ratio (M 4, budget): night6/t4syn/fr_p{03,...,16} and fr_p{20,...,35}_a0{2,6} (--timewindow-scale-to-p3
                --t-pad 60: each config's drawn W grown to W_p = 1 + round((W - 1) p / 3) so that
                N_samples / N_features stays at its p = 3 value; 60 days prepended (blobs at their first position), the
                same for every fixed-ratio run, so fr_p03 vs p03 measures the longer series alone).
  The fixed-window curve is also recomputed from the first 4 members of each bundle (curve fw_M4; members are seeded
  in order, so this equals an M = 4 run), because the absolute alpha error grows with Monte Carlo noise.
N_samples = (ny - 2r)(nx - 2r)(W - 1), N_features = p (2r + 1)^2 at the event box (summarize_all_vars,
scale_timewindow_to_p3); the table T4syn_samples.csv is computed from the 50 drawn configs (no run needed).
Metrics (N5_T3.per_config): share of trajectory density per class, alpha recovery A / (A + C) and its absolute
error, completion, link-level selection per class (TST_LINKSTATS). Means over configs, 95 % bootstrap CIs
(10,000 resamples of configs, seed 0; alphas pooled unless split).

python N6_T4_synthetic.py [--complete-only]   (PBS only) -> night6/analysis/T4syn_*.csv, figures_v6/synthetic/S10_synthetic_extension.*
"""
from __future__ import annotations

import importlib.util
import json
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
M_FR = 4   # ensemble size of the fixed-ratio runs (budget); the fixed-window curve is also shown at its first 4 members
CYCLE = ["noise", "response", "moving", "delayed", "nonlinear"]
MIX32 = [CYCLE[i % 5] for i in range(32)]
CURVES = {"fw": "Fixed window (W drawn)", "fr": "Fixed ratio (W grown, N/p as p = 3)"}
CURVE_STYLE = {"fw": dict(color="#000000", marker="o"), "fr": dict(color="#D55E00", marker="s")}


def run_dirs(curve, p):
    """Output dirs of one (curve, p); p >= 20 runs are split by alpha (<name>_a02, <name>_a06; one trial <= 8 h)."""
    base = (N5 / "t3" if p <= 16 else N6 / "t4syn") / f"p{p:02d}" if curve == "fw" else N6 / "t4syn" / f"fr_p{p:02d}"
    split = [base.with_name(base.name + s) for s in ("_a02", "_a06")]
    return [d for d in [base] + split if d.exists()]


def load(d):
    for nm in ("results_alphas_final.csv", "results_alphas_snapshot.csv"):
        if (d / nm).exists():
            return pd.read_csv(d / nm)
    return None


def _jobs_elapsed(d):
    for nm in ("results_jobs_final.csv", "results_jobs_snapshot.csv"):
        if (d / nm).exists():
            j = pd.read_csv(d / nm)
            if "elapsed_sec" in j:
                return j[["trial_id", "elapsed_sec"]]
    return None


def load_curve(curve, p):
    """per_config rows (N5_T3) of every dir of (curve, p), with completion and runtime per (config, alpha)."""
    out = []
    for d in run_dirs(curve, p):
        x = load(d)
        if x is None:
            continue
        pc = per_config(p, x)
        pc["timewindow"] = x["timewindow"].map(tw_steps).to_numpy()
        cm = completion_from_members(d, 30)
        if cm is not None and pc["completion"].isna().all():
            pc = pc.drop(columns="completion").merge(cm.rename(columns={"alpha_mix": "alpha"}),
                                                     on=["trial_id", "alpha"], how="left")
        j = _jobs_elapsed(d)
        if j is not None:
            pc = pc.merge(j, on="trial_id", how="left")
        out.append(pc)
    return pd.concat(out, ignore_index=True) if out else None


def linkstats_dirs(dirs, names):
    """N5_T3.linkstats pooled over dirs (selection rates weighted by the number of regressions)."""
    parts = [r for r in (linkstats(d, names) for d in dirs) if r]
    if not parts:
        return {}
    n = sum(r["n_regressions"] for r in parts)
    out = {"n_regressions": n}
    for k in {k for r in parts for k in r if k.startswith("sel_")}:
        out[k] = float(sum(r[k] * r["n_regressions"] for r in parts if k in r) / n)
    return out


def bundle_metrics(dirs, p, m_sub):
    """Class shares and alpha error from the first m_sub members of every bundle (members are seeded in order, so the
    first m_sub of an M = 20 ensemble are exactly an M = m_sub run). Density per variable as summarize_all_vars:
    sum over the member's centres with that parent of the number of grid points in the box around the centre."""
    ys, xs = SYN_GRID
    names = var_names(p)
    rows = []
    for d in dirs:
        for f in sorted((d / "bundles").glob("trial_*.npz")):
            meta = json.load(open(f.with_suffix(".json")))
            z = np.load(f)
            half = float(meta["params"]["box_size"]) / 2.0
            order = np.argsort(z["member_id"])[:m_sub]
            dens = np.zeros(len(names))
            for i in order:
                if not z["ok"][i] or z["lengths"][i] == 0:
                    continue
                c = z["centers"][i, : z["lengths"][i]].astype(float); pa = z["parents"][i, : z["lengths"][i]]
                w = (np.abs(xs[None, :] - c[:, 1:2]) <= half).sum(1) * (np.abs(ys[None, :] - c[:, 0:1]) <= half).sum(1)
                np.add.at(dens, pa - 1, w)
            tot = dens.sum()
            r = dict(trial_id=int(meta["trial_id"]), alpha=float(meta["alpha_mix"]))
            for k in CLASSES:
                idx = [i for i, v in enumerate(names) if cls(v) == k]
                r[f"share_{k}"] = float(dens[idx].sum() / tot) if idx and tot > 0 else np.nan
            a, c_ = dens[0], dens[2]
            r["alpha_err"] = float(abs(a / (a + c_) - r["alpha"])) if a + c_ > 0 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


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
    rows, summ, links, msub, store = [], [], [], [], {}
    for curve in CURVES:
        for p in PS:
            pc = load_curve(curve, p)
            if pc is None:
                print("missing", curve, p); continue
            tt = T[(T.curve == curve) & (T.p == p)][["trial_id", "W", "n_samples", "n_features", "ratio"]]
            pc = pc.merge(tt, on="trial_id", how="left")
            bad = (pc["timewindow"] - pc["W"]).abs().max()
            if bad > 0:
                print(f"WARNING {curve} p={p}: run windows differ from the table by up to {bad} d")
            pc["curve"] = curve; pc["p"] = p
            rows.append(pc)
            summ += summarize(curve, p, pc); store[(curve, p)] = pc
            ls = linkstats_dirs(run_dirs(curve, p), var_names(p))
            if ls:
                links.append(dict(curve=curve, p=p, **ls))
            # matched ensemble size: fixed window recomputed from its first M_FR members; fixed ratio from all its
            # members (check against the run's own summary)
            bm = bundle_metrics(run_dirs(curve, p), p, M_FR)
            if len(bm):
                dev = bm.merge(pc[["trial_id", "alpha", "share_true", "alpha_err"]], on=["trial_id", "alpha"],
                               suffixes=("", "_run"))
                if curve == "fr":
                    print(f"bundle check fr p={p}: max |share_true diff| "
                          f"{(dev.share_true - dev.share_true_run).abs().max():.2e}")
                msub += summarize(f"{curve}_M{M_FR}", p, bm); store[(f"{curve}_M{M_FR}", p)] = bm
    if "--complete-only" in sys.argv[1:]:   # draft: only (curve, p) runs with all 50 configs x both alphas
        keep = {k for k, g in store.items() if g.groupby("trial_id")["alpha"].nunique().eq(2).sum() == 50}
        store = {k: g for k, g in store.items() if k in keep}
        summ = [r for r in summ if (r["curve"], r["p"]) in keep]; msub = [r for r in msub if (r["curve"], r["p"]) in keep]
        links = [r for r in links if (r["curve"], r["p"]) in keep]
        rows = [g for g in rows if (g["curve"].iloc[0], g["p"].iloc[0]) in keep]
        print("complete-only:", sorted(keep))
    if not rows:
        return
    # partial runs drop the slow configs, so compare on matched configs (both alphas done):
    # common_fw = configs done at every p of the fixed window (panels a-c); paired = per p, configs done in both the
    # fixed-ratio run and the fixed-window run (panel d, matched M)
    def done(g):
        return set(g.groupby("trial_id")["alpha"].nunique().loc[lambda s: s == 2].index)
    fwk = [k for k in store if k[0] in ("fw", f"fw_M{M_FR}")]
    cfw = sorted(set.intersection(*[done(store[k]) for k in fwk])) if fwk else []
    comm = [r for k in fwk for r in summarize(k[0], k[1], store[k][store[k]["trial_id"].isin(cfw)])]
    pair = []
    for p in PS:
        ka, kb = ("fr", p), (f"fw_M{M_FR}", p)
        if ka in store and kb in store:
            cp = done(store[ka]) & done(store[kb])
            pair += [dict(r, n_pair=len(cp)) for k in (ka, kb) for r in summarize(k[0], p, store[k][store[k]["trial_id"].isin(cp)])]
    Sm = pd.concat([pd.DataFrame(summ + msub).assign(subset="all"), pd.DataFrame(comm).assign(subset="common_fw"),
                    pd.DataFrame(pair).assign(subset="paired")], ignore_index=True)
    P = pd.concat(rows); Lk = pd.DataFrame(links)
    print(f"fixed-window configs done at every p: {len(cfw)}")
    P.to_csv(OUT / "T4syn_per_config.csv", index=False); Sm.to_csv(OUT / "T4syn_summary.csv", index=False)
    Lk.to_csv(OUT / "T4syn_linkstats.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(agg.round(3).to_string())
        cols = ["curve", "p", "alpha", "n_cfg", "share_true", "share_true_lo", "share_true_hi"] \
            + [c for c in Sm.columns if c.startswith("share_") and not c.endswith(("_lo", "_hi", "_median"))
               and c != "share_true"] + ["alpha_err", "alpha_err_lo", "alpha_err_hi", "completion", "ratio_median"]
        print(Sm[[c for c in cols if c in Sm]].round(3).to_string()); print(Lk.round(3).to_string())
    a = Sm[(Sm.subset == "all") & (Sm.alpha == "pooled") & Sm.curve.isin(["fw", f"fw_M{M_FR}"])]
    sub = "all" if a.n_cfg.min() == 50 or len(cfw) < 20 else "common_fw"   # < 20 matched: all done configs, n shown
    print("figure subset (a-c):", sub)
    Sab = Sm[(Sm.subset == sub) & Sm.curve.isin(["fw", f"fw_M{M_FR}"])]
    figure(pd.concat([Sab, Sm[Sm.subset == "paired"]]), Lk, len(cfw) if sub == "common_fw" else None)


def figure(Sm, Lk, n_common=None):
    """Sm: fixed-window rows (panels a-c, fw in d) + subset 'paired' rows (fr and fw_M4 in panel d)."""
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
    curves_d = {"fw": CURVES["fw"] + ", M 20", f"fw_M{M_FR}": CURVES["fw"] + f", first {M_FR} members",
                "fr": CURVES["fr"] + f", M {M_FR}"}
    style_d = dict(CURVE_STYLE, **{f"fw_M{M_FR}": dict(color="0.55", marker="o", mfc="white", ls="--")})
    for curve, lab in curves_d.items():
        q = Sm[(Sm["curve"] == curve) & (Sm["alpha"] == "pooled") & ((Sm["subset"] == "paired") == (curve != "fw"))].sort_values("p")
        if not len(q):
            continue
        st = style_d[curve]
        for ax, k in ((ax_d1, "share_true"), (ax_d2, "alpha_err")):
            ax.errorbar(q["p"], q[k], yerr=[q[k] - q[k + "_lo"], q[k + "_hi"] - q[k]], ms=3, lw=1, capsize=1.5,
                        label=lab, **st)
    ax_d1.set_ylabel("True-parent\nshare"); ax_d1.set_ylim(0, 1.0)
    ax_d2.set_ylabel("|α rec. − α|"); ax_d2.set_ylim(0, None); ax_d2.set_xlabel("Number of variables p")
    ax_d2.set_xticks(PS); plt.setp(ax_d1.get_xticklabels(), visible=False)
    ax_d1.legend(loc="lower left", fontsize=5.2)
    if n_common is not None:   # some fixed-window runs incomplete: panels a-c on the configs done at every p
        ax_a.set_title(f"{n_common} of 50 configs (done at every p)", fontsize=6)
    nfw = Sm[(Sm["curve"] == "fw") & (Sm["alpha"] == "pooled")].sort_values("p")
    if n_common is None and nfw["n_cfg"].min() < 50:   # partial runs, all done configs: n per p above the bars
        for i, n in enumerate(nfw["n_cfg"]):
            ax_a.text(i, 1.01, f"n={int(n)}", ha="center", va="bottom", fontsize=4.8, color="0.35", transform=ax_a.get_xaxis_transform())
    qn = Sm[(Sm["subset"] == "paired") & (Sm["curve"] == "fr") & (Sm["alpha"] == "pooled")]
    if len(qn) and qn["n_pair"].min() < 50:   # panel d: configs done in both runs at that p
        for _, r in qn.iterrows():
            ax_d2.text(r["p"], 0.005, f"{int(r['n_pair'])}", ha="center", va="bottom", fontsize=4.8, color="0.35")
        ax_d2.text(1.0, 1.0, "numbers: paired configs", transform=ax_d2.transAxes, ha="right", va="top", fontsize=4.8, color="0.35")
    S.panel_labels([ax_a, ax_b, ax_c], x=-0.14)
    S.panel_labels([ax_d1], x=-0.14, start=3)
    print("fig", S.save(fig, "synthetic", "S10_synthetic_extension"))


if __name__ == "__main__":
    main()
