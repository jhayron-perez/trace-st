"""Night-5 T2: extended real cases vs the original sets, debugged baseline method (linear rule forced).

Runs: revision1/night5/t2/{debby_orig/part*, debby_ext/part*, pin_orig, pin_ext, pnw_orig, pnw_ext}.
Retention: trace_st.retention.harmonized_case (rule 3), applied to each run separately; the side-by-side
statistics use the configs retained in both sets of the case ("paired"), and the per-set statistics are also
reported on each set's own retained configs ("own").
Shares: per config, share of trajectory density per variable in each window (step 0 excluded; PNW also
"w0-10" = 1-10 d plus step 0) and per lead day; mean over configs with a 95 % percentile bootstrap CI
(10,000 resamples, seed 0); fraction of configs where each variable is rank 1.
Debby path accuracy: mean_of_member_mean_track_dist_km (distance to the IBTrACS track), original vs extended.
Rule-1 readings are evaluated as pre-declared in the night-5 report.

python N5_T2.py [--nproc 32] [--partial]   (PBS only; --partial uses snapshots of runs in progress)
Outputs: night5/analysis/T2_*.csv, T2_readings.json; figures figures_v5/real_cases/.
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
import rev1_common as C  # noqa: E402
from A4_twins import boot_ci  # noqa: E402
from N3_bed_summary import densities  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = C.R1
N5 = R1 / "night5"
T2 = N5 / "t2"
OUT = N5 / "analysis"
WINDOWS = dict(debby=[(1, 4), (5, 15)], pinatubo=[(1, 3), (4, 16)], pnw=[(1, 10), (11, 20), (21, 30)])
HEADLINE = dict(debby="w5-15", pinatubo="w4-16", pnw="w1-10")
DEB3 = ["Tb", "vo", "precip"]
PIN4 = ["TMSO201", "BURDENSO401", "TMH2SO401", "AEROD_v"]
PNW6 = ["Z500", "Z10", "MTNLWRF", "TCWV", "MSLHF", "MSSHF"]
CASES = {
    "debby": dict(orig=("debby_orig", DEB3), ext=("debby_ext", DEB3 + ["CAPE", "DIV200", "WS10", "MSLP"]),
                  roles=dict(vo="parent", Tb="intermediate", precip="intermediate", CAPE="competitor",
                             DIV200="competitor", WS10="child", MSLP="child"), title="TS Debby 2006"),
    "pinatubo": dict(orig=("pin_orig", PIN4), ext=("pin_ext", PIN4 + ["TMSO203", "BURDENSO403", "FLUT", "T050", "FSDS"]),
                     roles=dict(TMSO201="parent", BURDENSO401="intermediate", AEROD_v="intermediate",
                                TMH2SO401="intermediate", TMSO203="competitor", BURDENSO403="competitor",
                                FLUT="child", T050="child", FSDS="child"), title="Pinatubo 1991"),
    "pnw": dict(orig=("pnw_orig", PNW6), ext=("pnw_ext", PNW6 + ["T2M", "BLH"]),
                roles=dict(Z500="parent", TCWV="parent", MTNLWRF="candidate", MSLHF="candidate", MSSHF="candidate",
                           Z10="competitor", T2M="child", BLH="child"), title="PNW heat wave 2021"),
}
ROLE_ORDER = ["parent", "intermediate", "candidate", "competitor", "child"]


def run_dirs(rel):
    d = T2 / rel
    parts = sorted(d.glob("part*"))
    return parts if parts else [d]


def load_table(case, rel, partial):
    rows = []
    for d in run_dirs(rel):
        if case == "pnw":
            f = d / "hyperparam_search_summary.csv"
            if f.exists():
                rows.append(pd.read_csv(f).assign(_dir=str(d)))
            elif partial:   # PNW writes one summary.json per run
                js = [json.load(open(p)) for p in sorted(d.glob("run_*/summary.json"))]
                if js:
                    rows.append(pd.DataFrame(js).assign(_dir=str(d)))
        else:
            for nm in (["results_jobs_final.csv"] + (["results_jobs_snapshot.csv"] if partial else [])):
                if (d / nm).exists():
                    rows.append(pd.read_csv(d / nm).assign(_dir=str(d))); break
    return pd.concat(rows, ignore_index=True) if rows else None


def shares_from_D(D, case):
    out = {}
    for (w0, w1) in WINDOWS[case]:
        Dw = D[w0: w1 + 1].sum(0)
        if Dw.sum() > 0:
            out[f"w{w0}-{w1}"] = Dw / Dw.sum()
    if case == "pnw":
        Dw = D[0: 11].sum(0)
        if Dw.sum() > 0:
            out["w0-10"] = Dw / Dw.sum()
    return out


def lead_shares(D):
    tot = D.sum(1)
    return {L: D[L] / tot[L] for L in range(1, D.shape[0]) if tot[L] > 0}


def case_densities(case, rel, vars_, tab, nproc):
    idc = "run_id" if case == "pnw" else "trial_id"
    keep = tab.loc[retention.harmonized_case(case, tab), [idc, "_dir"]]
    Dall = {}
    for d, g in keep.groupby("_dir"):
        Dd, _ = densities(case, Path(d), len(vars_), sorted(g[idc].astype(int)), nproc, 0)
        Dall.update(Dd)
    return Dall


def stats_rows(case, set_, vars_, Dd, cfgs, roles, tag):
    W, L, K = [], [], []
    sh = {c: shares_from_D(Dd[c], case) for c in cfgs}
    for w in sorted({w for s in sh.values() for w in s}, key=lambda s: (int(s[1:].split("-")[0]), s)):
        X = np.array([sh[c][w] for c in cfgs if w in sh[c]])
        if len(X) < 3:
            continue
        rk1 = np.bincount(X.argmax(1), minlength=len(vars_)) / len(X)
        for i, v in enumerate(vars_):
            m, lo, hi = boot_ci(X[:, i])
            W.append(dict(case=case, set=set_, cfgset=tag, window=w, var=v, role=roles[v], n_cfg=len(X), share=m,
                          lo=lo, hi=hi, frac_rank1=rk1[i]))
        for r in ROLE_ORDER:
            idx = [i for i, v in enumerate(vars_) if roles[v] == r]
            if idx:
                m, lo, hi = boot_ci(X[:, idx].sum(1))
                K.append(dict(case=case, set=set_, cfgset=tag, window=w, role=r, n_cfg=len(X), share=m, lo=lo, hi=hi,
                              vars="+".join(vars_[i] for i in idx)))
    nl = max(D.shape[0] for D in Dd.values())
    for Lx in range(1, nl):
        X = [lead_shares(Dd[c]).get(Lx) for c in cfgs]
        X = np.array([x for x in X if x is not None])
        if len(X) < 3:
            continue
        for i, v in enumerate(vars_):
            m, lo, hi = boot_ci(X[:, i])
            L.append(dict(case=case, set=set_, cfgset=tag, lead=Lx, var=v, role=roles[v], n_cfg=len(X), share=m,
                          lo=lo, hi=hi))
        for r in ROLE_ORDER:
            idx = [i for i, v in enumerate(vars_) if roles[v] == r]
            if idx:
                m, lo, hi = boot_ci(X[:, idx].sum(1))
                L.append(dict(case=case, set=set_, cfgset=tag, lead=Lx, var=f"role:{r}", role=r, n_cfg=len(X),
                              share=m, lo=lo, hi=hi))
    return W, L, K


def readings(W):
    """Rule 1 (pre-declared), evaluated on the extended set, paired configs."""
    w = W[(W["set"] == "ext") & (W["cfgset"] == "paired")]

    def g(case, win, var, col="share"):
        x = w[(w["case"] == case) & (w["window"] == win) & (w["var"] == var)]
        return float(x[col].iloc[0]) if len(x) else np.nan

    def gs(case, win, role):
        x = w[(w["case"] == case) & (w["window"] == win) & (w["role"] == role)]
        return float(x["share"].sum()) if len(x) else np.nan

    def top(case, win):
        x = w[(w["case"] == case) & (w["window"] == win)]
        return x.sort_values("share", ascending=False)["var"].iloc[0] if len(x) else None
    R = {}
    if (w["case"] == "debby").any():
        R["debby"] = dict(
            children_1_4=gs("debby", "w1-4", "child"), children_5_15=gs("debby", "w5-15", "child"),
            children_fade=gs("debby", "w1-4", "child") > gs("debby", "w5-15", "child"),
            top_5_15=top("debby", "w5-15"), vo_not_displaced_5_15=top("debby", "w5-15") == "vo",
            top_1_4=top("debby", "w1-4"), vo_rank1_1_4=top("debby", "w1-4") == "vo",
            vo_frac_rank1_5_15=g("debby", "w5-15", "vo", "frac_rank1"))
    if (w["case"] == "pinatubo").any():
        ch13 = gs("pinatubo", "w1-3", "child")
        R["pinatubo"] = dict(
            children_1_3=ch13, children_4_16=gs("pinatubo", "w4-16", "child"),
            children_late_and_weak=bool(ch13 <= 0.10 and all(
                gs("pinatubo", wl, "child") < g("pinatubo", wl, "TMSO201") for wl in ("w1-3", "w4-16"))),
            top_4_16=top("pinatubo", "w4-16"), so2_top_4_16=top("pinatubo", "w4-16") == "TMSO201",
            so2_frac_rank1_4_16=g("pinatubo", "w4-16", "TMSO201", "frac_rank1"),
            FSDS_1_3=g("pinatubo", "w1-3", "FSDS"), FSDS_4_16=g("pinatubo", "w4-16", "FSDS"))
    if (w["case"] == "pnw").any():
        mx_child = max(g("pnw", "w1-10", "T2M"), g("pnw", "w1-10", "BLH"))
        R["pnw"] = dict(Z500_1_10=g("pnw", "w1-10", "Z500"), TCWV_1_10=g("pnw", "w1-10", "TCWV"),
                        T2M_1_10=g("pnw", "w1-10", "T2M"), BLH_1_10=g("pnw", "w1-10", "BLH"),
                        children_do_not_displace=bool(mx_child < min(g("pnw", "w1-10", "Z500"), g("pnw", "w1-10", "TCWV"))),
                        top_1_10=top("pnw", "w1-10"))
    return R


def frac_displaced(case, vars_, Dd, cfgs, main, others, win):
    """Fraction of configs in which any of `others` has a larger window share than `main`."""
    x = []
    for c in cfgs:
        s = shares_from_D(Dd[c], case).get(win)
        if s is not None:
            x.append(max(s[vars_.index(o)] for o in others) > s[vars_.index(main)])
    return float(np.mean(x)) if x else np.nan


# ------------------------------------------------------------------ figures
def fig_case(case, L, W, cfg):
    import matplotlib.pyplot as plt
    S.set_style()
    fig = plt.figure(figsize=(S.DOUBLE, 0.62 * S.DOUBLE))
    gsp = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.55, wspace=0.42)
    axa, axb = fig.add_subplot(gsp[0, 0:1]), fig.add_subplot(gsp[0, 1:2])
    axleg = fig.add_subplot(gsp[0, 2]); axleg.axis("off")
    axc, axd = fig.add_subplot(gsp[1, 0:2]), fig.add_subplot(gsp[1, 2])
    roles = cfg["roles"]
    ext_vars = cfg["ext"][1]
    for ax, set_, ttl in ((axa, "orig", f"Original set, {len(cfg['orig'][1])} variables"),
                          (axb, "ext", f"Extended set, {len(ext_vars)} variables")):
        x = L[(L["case"] == case) & (L["set"] == set_) & (L["cfgset"] == "paired") & ~L["var"].str.startswith("role:")]
        for v in cfg[set_][1]:
            g = x[x["var"] == v].sort_values("lead")
            if not len(g):
                continue
            ls = "-" if v in cfg["orig"][1] else "--"
            ax.plot(g["lead"], g["share"], ls, color=S.col(v), lw=1.1, label=S.lab(v))
            ax.fill_between(g["lead"], g["lo"], g["hi"], color=S.col(v), alpha=0.18, lw=0)
        for (w0, w1), c in zip(WINDOWS[case], S.WINDOW_COLORS):
            ax.axvspan(w0 - 0.5, w1 + 0.5, color=c, alpha=0.06, lw=0)
        n = int(x["n_cfg"].max()) if len(x) else 0
        ax.set_title(f"{ttl} (n = {n})", loc="center", fontsize=6.5)
        ax.set_xlabel("Lead before the event (days)"); ax.set_ylabel("Share of trajectory density")
        ax.set_ylim(0, None); ax.set_xlim(0.5, x["lead"].max() + 0.5 if len(x) else 1)
    hs, ls_ = axb.get_legend_handles_labels()
    role_of = {S.lab(v): roles[v] for v in ext_vars}
    order = sorted(range(len(ls_)), key=lambda i: (ROLE_ORDER.index(role_of[ls_[i]]), i))
    axleg.legend([hs[i] for i in order], [f"{ls_[i]}  ({S.ROLE_LABELS[role_of[ls_[i]]].lower()})" for i in order],
                 loc="center left", title="Variable (role); dashed = added", title_fontsize=6.5)
    # c: rank-1 fraction per window, orig (hatched) vs ext (solid)
    w = W[(W["case"] == case) & (W["cfgset"] == "paired") & (W["window"].isin([f"w{a}-{b}" for a, b in WINDOWS[case]]))]
    wins = [f"w{a}-{b}" for a, b in WINDOWS[case]]
    nv = len(ext_vars); bw = 0.8 / nv
    for k, wl in enumerate(wins):
        for i, v in enumerate(ext_vars):
            for set_, hatch, off in (("orig", "////", -bw / 4), ("ext", None, bw / 4)):
                r = w[(w["window"] == wl) & (w["var"] == v) & (w["set"] == set_)]
                if len(r):
                    axc.bar(k + (i - nv / 2 + 0.5) * bw + off, float(r["frac_rank1"].iloc[0]), bw / 2,
                            color=S.col(v), edgecolor="white" if hatch else S.col(v), hatch=hatch, lw=0.3)
    axc.set_xticks(range(len(wins))); axc.set_xticklabels([f"{wl[1:].replace('-', '–')} d" for wl in wins])
    axc.set_ylabel("Fraction of configs\nwith rank 1"); axc.set_ylim(0, 1.05)
    axc.set_title("Bar colours = variables as in a, b; hatched = original set", fontsize=6)
    from matplotlib.patches import Patch
    axc.legend([Patch(facecolor="0.6", hatch="////", edgecolor="white"), Patch(facecolor="0.6")],
               ["original set", "extended set"], loc="upper right", ncol=2)
    # d: role-summed shares by window (extended set) with CI
    K = _K[(_K["case"] == case) & (_K["set"] == "ext") & (_K["cfgset"] == "paired") & _K["window"].isin(wins)]
    rl = [r for r in ROLE_ORDER if (K["role"] == r).any()]
    for k, wl in enumerate(wins):
        for j, r in enumerate(rl):
            q = K[(K["window"] == wl) & (K["role"] == r)]
            if len(q):
                xq = k + (j - len(rl) / 2 + 0.5) * (0.8 / len(rl))
                axd.bar(xq, float(q["share"].iloc[0]), 0.8 / len(rl), color=S.ROLE_COLORS[r],
                        label=S.ROLE_LABELS[r] if k == 0 else None)
                axd.errorbar(xq, float(q["share"].iloc[0]), yerr=[[float(q["share"].iloc[0] - q["lo"].iloc[0])],
                             [float(q["hi"].iloc[0] - q["share"].iloc[0])]], color="k", lw=0.6, capsize=1)
    axd.set_xticks(range(len(wins))); axd.set_xticklabels([f"{wl[1:].replace('-', '–')} d" for wl in wins])
    axd.set_ylabel("Summed share by role\n(extended set)"); axd.set_ylim(0, 1.15)
    axd.legend(loc="upper right", fontsize=5.2, ncol=2, columnspacing=0.6, handlelength=1.0)
    S.panel_labels([axa, axb, axc, axd], x=-0.2, y=1.08)
    fig.suptitle(cfg["title"], x=0.02, ha="left", fontsize=8, fontweight="bold")
    return S.save(fig, "real_cases", f"{case}_original_vs_extended")


def fig_summary(K):
    import matplotlib.pyplot as plt
    S.set_style()
    fig, axs = plt.subplots(1, 3, figsize=(S.DOUBLE, 0.34 * S.DOUBLE))
    for ax, case in zip(axs, ("debby", "pinatubo", "pnw")):
        k = K[(K["case"] == case) & (K["cfgset"] == "paired") & (K["window"] == HEADLINE[case])]
        rl = [r for r in ROLE_ORDER if (k["role"] == r).any()]
        for j, r in enumerate(rl):
            for set_, dx, hatch in (("orig", -0.2, "////"), ("ext", 0.2, None)):
                q = k[(k["role"] == r) & (k["set"] == set_)]
                if len(q):
                    y = float(q["share"].iloc[0])
                    ax.bar(j + dx, y, 0.38, color=S.ROLE_COLORS[r], hatch=hatch, edgecolor="white" if hatch else S.ROLE_COLORS[r], lw=0.3)
                    ax.errorbar(j + dx, y, yerr=[[y - float(q["lo"].iloc[0])], [float(q["hi"].iloc[0]) - y]], color="k", lw=0.6, capsize=1)
        ax.set_xticks(range(len(rl))); ax.set_xticklabels([S.ROLE_LABELS[r].replace(" / ", "/\n").replace(" ", "\n") for r in rl])
        ax.set_ylim(0, 1); ax.set_title(f"{CASES[case]['title']}, {HEADLINE[case][1:].replace('-', '–')} d", loc="left")
    axs[0].set_ylabel("Summed share of trajectory density")
    from matplotlib.patches import Patch
    axs[2].legend([Patch(facecolor="0.6", hatch="////", edgecolor="white"), Patch(facecolor="0.6")],
                  ["original set", "extended set"], loc="upper right")
    S.panel_labels(axs, x=-0.2)
    return S.save(fig, "real_cases", "summary_3cases_roles")


_K = None


def main():
    global _K
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    Wl, Ll, Kl, counts, path, disp = [], [], [], [], [], []
    for case in a.cases:
        cfg = CASES[case]
        tabs, Ds = {}, {}
        for set_ in ("orig", "ext"):
            rel, vars_ = cfg[set_]
            t = load_table(case, rel, a.partial)
            if t is None:
                print("no output yet", case, set_, flush=True); continue
            tabs[set_] = t
            Ds[set_] = case_densities(case, rel, vars_, t, a.nproc)
            old = (retention.pnw_retained(t) if case == "pnw" else
                   retention.debby_retained(t) if case == "debby" else retention.pinatubo_retained(t))
            counts.append(dict(case=case, set=set_, n_done=len(t), n_harmonized=int(retention.harmonized_case(case, t).sum()),
                               n_old_rule=int(old.sum())))
        if not Ds:
            continue
        paired = sorted(set.intersection(*[set(D) for D in Ds.values()])) if len(Ds) == 2 else None
        for set_, Dd in Ds.items():
            vars_ = cfg[set_][1]
            for tag, cf in (("own", sorted(Dd)), ("paired", paired)):
                if not cf:
                    continue
                w, l, k = stats_rows(case, set_, vars_, Dd, cf, cfg["roles"], tag)
                Wl += w; Ll += l; Kl += k
        if "ext" in Ds and paired:
            ev = cfg["ext"][1]
            mains = dict(debby=["vo"], pinatubo=["TMSO201"], pnw=["Z500", "TCWV"])[case]
            for m in mains:
                for grp in ("child", "competitor"):
                    oth = [v for v in ev if cfg["roles"][v] == grp]
                    for (w0, w1) in WINDOWS[case]:
                        disp.append(dict(case=case, main=m, group=grp, window=f"w{w0}-{w1}", n_cfg=len(paired),
                                         frac_configs_displaced=frac_displaced(case, ev, Ds["ext"], paired, m, oth, f"w{w0}-{w1}")))
        if case == "debby" and len(tabs) == 2:
            col = "mean_of_member_mean_track_dist_km"
            o = tabs["orig"].set_index("trial_id")[col]; e = tabs["ext"].set_index("trial_id")[col]
            for tag, cf in (("paired", paired or []),):
                d = (e.reindex(cf) - o.reindex(cf)).dropna().to_numpy()
                m, lo, hi = boot_ci(d)
                path.append(dict(cfgset=tag, n_cfg=len(d), orig_km=float(o.reindex(cf).mean()), ext_km=float(e.reindex(cf).mean()),
                                 diff_ext_minus_orig_km=m, lo=lo, hi=hi, frac_ext_farther=float(np.mean(d > 0)) if len(d) else np.nan))
    W, L, K = pd.DataFrame(Wl), pd.DataFrame(Ll), pd.DataFrame(Kl)
    _K = K
    for nm, X in (("T2_window_shares", W), ("T2_lead_shares", L), ("T2_role_shares", K), ("T2_counts", pd.DataFrame(counts)),
                  ("T2_debby_path", pd.DataFrame(path)), ("T2_displaced", pd.DataFrame(disp))):
        X.to_csv(OUT / f"{nm}.csv", index=False)
    R = readings(W) if len(W) else {}
    json.dump(R, open(OUT / "T2_readings.json", "w"), indent=1, default=float)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(pd.DataFrame(counts).to_string())
        if len(W):
            print(W[W["cfgset"] == "paired"].pivot_table(index=["case", "window", "var"], columns="set",
                                                          values=["share", "frac_rank1"]).round(3).to_string())
        print(pd.DataFrame(path).round(1).to_string()); print(pd.DataFrame(disp).round(3).to_string())
    print(json.dumps(R, indent=1, default=float))
    for case in a.cases:
        if len(W) and ((W["case"] == case) & (W["cfgset"] == "paired")).any():
            print("fig", fig_case(case, L, W, CASES[case]), flush=True)
    if len(K) and set(K[K["cfgset"] == "paired"]["case"]) >= {"debby", "pinatubo", "pnw"}:
        print("fig", fig_summary(K))


if __name__ == "__main__":
    main()
