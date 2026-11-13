"""Night-4 summaries (A real-case decoys, B1 ko0_10 on evaluation years, B2/B3 selection years + guardrails,
C synthetic decoy taxonomy, E PNW 11-30 d under ko0_10). Partial outputs allowed.

Shares: same definitions as A4_twins.py / N3_bed_summary.py (box-cell density per parent variable, step 0
excluded, Debby hourly steps summed to lead days; windows PNW 1-10/11-20/21-30 d, Debby 1-4/5-15 d, Pinatubo
1-3/4-16 d). CIs: percentile bootstrap over configs (boot_ci, 10,000 resamples, seed 0). Configs: the
baseline-retained sets (PNW T6r 233, Debby rebuilt 75, Pinatubo T5r 999) restricted to those present.

python N4_summary.py --part {syn,sel,b1,a,c,all} [--nproc 32]   (PBS only)
Outputs: revision1/night4/analysis/*.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from A4_twins import boot_ci, horizon, retained_ids, PNW6, PNW_STEM, WINDOWS  # noqa: E402
from N3_bed_summary import densities, window_shares, twin_rows, summarize_twins, syn_metrics, load_first  # noqa: E402

R1 = C.R1
N3 = R1 / "night3"
N4 = R1 / "night4"
OUT = N4 / "analysis"
P4 = C.VARS["pinatubo"]
DEB3 = C.VARS["debby"]


def per_lead_shares(D):
    tot = D.sum(1)
    return {L: D[L] / tot[L] for L in range(1, D.shape[0]) if tot[L] > 0}


# ------------------------------------------------------------------ rule 1 (night 4): baseline |alpha err| CI
def t2_detail(d):
    x = load_first(d, ["results_alphas_final.csv", "results_alphas_snapshot.csv"])
    if x is None or not len(x):
        return {}
    ok = x["estimated_alpha_density"].notna()
    r = dict(T2_n=int(x["trial_id"].nunique()), T2_alphas=",".join(map(str, sorted(x["alpha_mix"].unique()))),
             T2_r=float(np.corrcoef(x.loc[ok, "alpha_mix"], x.loc[ok, "estimated_alpha_density"])[0, 1]))
    rng = np.random.default_rng(0)
    for a in sorted(x["alpha_mix"].unique()):
        g = x[(x["alpha_mix"] - a).abs() < 1e-9]
        e = (g["estimated_alpha_density"] - g["alpha_mix"]).abs().dropna().to_numpy()
        if not len(e):
            continue
        b = np.median(e[rng.integers(0, len(e), size=(2000, len(e)))], axis=1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        r[f"err_a{a}"] = float(np.median(e)); r[f"err_hw_a{a}"] = float((hi - lo) / 2)
        if "is_valid_physical" in g:
            r[f"valid_a{a}"] = float(g["is_valid_physical"].mean())
    return r


def part_syn(a):
    rows = []
    roots = [("night3_bed", N3 / "bed"), ("night3_bedM10", N3 / "bedM10"), ("night4_b1", N4 / "b1"),
             ("night4_b2", N4 / "b2")]
    for tag, root in roots:
        if not root.exists():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            r = syn_metrics(d, d.name); r["root"] = tag
            for t2 in ("T2en", "T2en_all"):
                det = t2_detail(d / t2)
                rows.append(dict(r, t2_task=t2, **{f"{k}": v for k, v in det.items()})) if det else None
            if not any((d / t).exists() for t in ("T2en", "T2en_all")):
                rows.append(r)
            t1 = load_first(d / "T1_1000", ["results_tracks_final.csv", "results_tracks_snapshot.csv"])
            if t1 is not None:
                from trace_st import retention
                t1 = t1[t1["method"] == "granger"]
                per = t1.groupby("trial_id")["mean_distance"].mean()
                rows.append(dict(variant=d.name, root=tag, t2_task="T1_1000", T1_n=int(per.size),
                                 T1_completion=float(retention.t1_retained(t1).mean()),
                                 T1_track_err_median=float(per.median())))
    S = pd.DataFrame(rows); S.to_csv(OUT / "syn_guardrails.csv", index=False)
    # T1 1000 baseline (paper default, night 1) for comparison
    t1b = load_first(R1 / "T1r_univariate", ["results_tracks_final.csv", "results_tracks_snapshot.csv"])
    if t1b is not None:
        from trace_st import retention
        t1b = t1b[t1b["method"] == "granger"]
        per = t1b.groupby("trial_id")["mean_distance"].mean()
        pd.DataFrame([dict(variant="base", root="T1r_univariate", T1_n=int(per.size),
                           T1_completion=float(retention.t1_retained(t1b).mean()),
                           T1_track_err_median=float(per.median()))]).to_csv(OUT / "syn_T1_base1000.csv", index=False)
    # T7 decoy sweep (proxy decoys), all methods with the cells present
    dec = []
    for tag, root in (("night3_n4", N3 / "n4"), ("night3_bed", N3 / "bed"), ("night4_b1", N4 / "b1"),
                      ("night4_c", N4 / "c"), ("night3_d4", N3 / "d4")):
        for d in glob.glob(str(root / "*" / "*")):
            d = Path(d); x = load_first(d, ["results_alphas_final.csv", "results_alphas_snapshot.csv"])
            if x is None or "t7_share_density_decoy" not in x:
                continue
            dec.append(dict(root=tag, variant=d.parent.name, run=d.name, n=len(x),
                            **{c: float(x[c].median()) for c in ("t7_share_density_v1v3", "t7_share_density_decoy",
                                                                 "t7_share_density_noise", "t7_share_density_moving",
                                                                 "t7_alpha_abs_err_density_13") if c in x}))
    pd.DataFrame(dec).to_csv(OUT / "syn_t7_cells.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(S.round(3).to_string(), "\n", pd.DataFrame(dec).round(3).to_string(), flush=True)


# ------------------------------------------------------------------ B2/B3 selection years (M 10, half configs)
def part_sel(a):
    ids = {k: retained_ids(k) for k in ("pnw", "debby", "pinatubo")}
    rows, rows_b = [], []
    for v in ("kocond", "koderand"):
        for kind in ("pnw", "debby", "pinatubo"):
            rows += twin_rows(N4 / "b2", v, kind, ids[kind], a.nproc, 10, "M10")
    long = pd.DataFrame(rows)
    if long.empty:
        print("no selection-year outputs yet"); return
    long.to_csv(OUT / "sel_twins_long.csv.gz", index=False)
    TW = summarize_twins(long)
    for v in long["variant"].unique():
        for kind in long.loc[long["variant"] == v, "case"].unique():
            cfgs = sorted(long.loc[(long["variant"] == v) & (long["case"] == kind), "cfg"].unique())
            root = N3 / ("bedM10" if kind == "pinatubo" else "bed")
            rows_b += [dict(r, variant=f"base_for_{v}") for r in twin_rows(root, "base", kind, cfgs, a.nproc, 10, "M10")]
            rows_b += [dict(r, variant=f"ko0_10_for_{v}") for r in
                       twin_rows(N3 / "bed", "ko0_10", kind, cfgs, a.nproc, 10, "M10")] if kind != "pinatubo" else []
    TB = summarize_twins(pd.DataFrame(rows_b))
    S = pd.concat([TW, TB]); S.to_csv(OUT / "sel_twins.csv", index=False)
    prim = S[((S.case == "pnw") & S["var"].isin(["Z500", "TCWV"]) & (S.window == "w1-10")) |
             ((S.case == "debby") & (S.window == "w1-4")) | ((S.case == "pinatubo") & (S.window == "w1-3"))]
    print(prim[["variant", "case", "var", "window", "n_cfg", "real_mean", "twin_mean", "diff_mean", "lo", "hi"]]
          .round(3).to_string(), flush=True)


# ------------------------------------------------------------------ B1 evaluation years (+ E long lead)
def eval_runs(kind):
    if kind == "pnw":
        return [(f"pnw_{s}_{y}", f"{s}_{y}", PNW6 + [f"{PNW_STEM[s]}_P{y}"], (PNW_STEM[s], f"{PNW_STEM[s]}_P{y}"), y)
                for s in ("z500", "tcwv", "olr", "slhf") for y in (2006, 2012, 2020)]
    return [(f"debby_vo_{y}", f"vo_{y}", DEB3 + [f"VO_P{y}"], ("vo", f"VO_P{y}"), y) for y in (2003, 2005, 2008)]


def part_b1(a):
    ids = {k: retained_ids(k) for k in ("pnw", "debby", "pinatubo")}
    base_root = dict(pnw=R1 / "A1_pnw_twin", debby=R1 / "A2_debby_twin")
    rows, lead_rows = [], []
    for kind in ("pnw", "debby"):
        for task, btask, vars_, (real, twin), y in eval_runs(kind):
            Dk, sk = densities(kind, N4 / "b1" / "ko0_10" / task, len(vars_), ids[kind], a.nproc, 0)
            if not Dk:
                continue
            Db, sb = densities(kind, base_root[kind] / btask, len(vars_), sorted(Dk), a.nproc, 0)
            for v, Dd, sc in (("ko0_10", Dk, sk), ("base", Db, sb)):
                for cfg, D in Dd.items():
                    if cfg not in Dk or cfg not in Db:
                        continue
                    for w, sh in window_shares(D, kind).items():
                        rows.append(dict(variant=v, tag="M30", case=kind, task=task, year=y, cfg=cfg, window=w, var=real,
                                         share_real=sh[vars_.index(real)], share_twin=sh[vars_.index(twin)],
                                         success=sc.get(cfg, np.nan)))
                    for L, sh in per_lead_shares(D).items():
                        lead_rows.append(dict(variant=v, case=kind, year=y, cfg=cfg, lead=L, var=real,
                                              diff=sh[vars_.index(real)] - sh[vars_.index(twin)]))
    # Pinatubo SO2 twin at full scale: A3 combined set (base, night 2) vs ko0_10 (night-4 A3)
    pv = P4 + ["TMSO203", "BURDENSO403", "FLUT"]
    Dk, sk = densities("pinatubo", N4 / "a" / "ko0_10" / "combined_TMSO203_BURDENSO403_FLUT", len(pv), ids["pinatubo"], a.nproc, 0)
    if Dk:
        Db, sb = densities("pinatubo", R1 / "A3_pinatubo_twin" / "combined_TMSO203_BURDENSO403_FLUT", len(pv), sorted(Dk), a.nproc, 0)
        for v, Dd in (("ko0_10", Dk), ("base", Db)):
            for cfg, D in Dd.items():
                if cfg in Dk and cfg in Db:
                    for w, sh in window_shares(D, "pinatubo").items():
                        for real, twin in (("TMSO201", "TMSO203"), ("BURDENSO401", "BURDENSO403")):
                            rows.append(dict(variant=v, tag="M30", case="pinatubo", task="combined", year="combined",
                                             cfg=cfg, window=w, var=real, share_real=sh[pv.index(real)],
                                             share_twin=sh[pv.index(twin)], success=np.nan))
    long = pd.DataFrame(rows)
    if long.empty:
        print("no B1 outputs yet"); return
    long.to_csv(OUT / "b1_twins_long.csv.gz", index=False)
    S = summarize_twins(long); S.to_csv(OUT / "b1_twins.csv", index=False)
    print(S[["variant", "case", "var", "window", "n_cfg", "n_years", "real_mean", "twin_mean", "diff_mean", "lo", "hi"]]
          .round(3).to_string(), flush=True)
    # paired difference ko0_10 - base of the twin diff, per config (same configs)
    long["diff"] = long["share_real"] - long["share_twin"]
    P = []
    for (case, var, w), g in long.groupby(["case", "var", "window"]):
        pv_ = g.pivot_table(index="cfg", columns="variant", values="diff", aggfunc="mean").dropna()
        if {"ko0_10", "base"} <= set(pv_.columns) and len(pv_) > 2:
            m, lo, hi = boot_ci((pv_["ko0_10"] - pv_["base"]).values)
            P.append(dict(case=case, var=var, window=w, n_cfg=len(pv_), ko_minus_base=m, lo=lo, hi=hi))
    pd.DataFrame(P).to_csv(OUT / "b1_paired.csv", index=False); print(pd.DataFrame(P).round(3).to_string())
    # E: per-lead diff and horizon (PNW 11-30 d under ko0_10 vs base)
    LR = pd.DataFrame(lead_rows)
    if len(LR):
        H, Ls = [], []
        for (v, case, var), g in LR.groupby(["variant", "case", "var"]):
            for L, gl in g.groupby("lead"):
                pc = gl.groupby("cfg")["diff"].mean(); m, lo, hi = boot_ci(pc.values)
                Ls.append(dict(variant=v, case=case, var=var, lead=int(L), diff_mean=m, lo=lo, hi=hi))
            df = pd.DataFrame([x for x in Ls if x["variant"] == v and x["case"] == case and x["var"] == var]).sort_values("lead")
            h1, h2 = horizon(df)
            H.append(dict(variant=v, case=case, var=var, horizon_contiguous=h1, last_lead_ci_above_0=h2))
        pd.DataFrame(Ls).to_csv(OUT / "b1_lead.csv", index=False); pd.DataFrame(H).to_csv(OUT / "b1_horizon.csv", index=False)
        print(pd.DataFrame(H).to_string(), flush=True)


# ------------------------------------------------------------------ A real-case decoys
def a_cases():
    """(case, kind, [(method, run dir)], vars, main drivers, decoys, baseline run dir, baseline vars)"""
    pv = P4 + ["TMSO203", "BURDENSO403", "FLUT"]
    return [
        ("debby", "debby", [(m, N4 / "a" / m / "debby_cape_div200") for m in ("base", "ko0_10")],
         DEB3 + ["CAPE", "DIV200"], ["vo"], ["CAPE", "DIV200"], R1 / "T4r_debby_vorebuilt", DEB3),
        ("pnw", "pnw", [(m, N4 / "a" / m / "pnw_t2m_tcc") for m in ("base", "ko0_10")],
         PNW6 + ["T2M", "TCC"], ["Z500", "TCWV"], ["T2M", "TCC"], R1 / "T6r_pnw", PNW6),
        ("pinatubo", "pinatubo", [("base", R1 / "A3_pinatubo_twin" / "combined_TMSO203_BURDENSO403_FLUT"),
                                  ("ko0_10", N4 / "a" / "ko0_10" / "combined_TMSO203_BURDENSO403_FLUT")],
         pv, ["TMSO201", "BURDENSO401"], ["FLUT"], R1 / "T5r_pinatubo", P4),
    ]


def part_a(a):
    ids = {k: retained_ids(k) for k in ("pnw", "debby", "pinatubo")}
    W, L, T, K = [], [], [], []
    for case, kind, runs, vars_, mains, decoys, bdir, bvars in a_cases():
        Db, _ = densities(kind, bdir, len(bvars), ids[kind], a.nproc, 0)
        for meth, d in runs:
            Dd, succ = densities(kind, d, len(vars_), ids[kind], a.nproc, 0)
            if not Dd:
                continue
            cf = sorted(set(Dd) & set(Db))
            # window shares per variable (decoy run) and baseline shares of the paper variables (same configs)
            for nm, DD, vv in ((meth, Dd, vars_), (f"paper_baseline_for_{meth}", Db, bvars)):
                for w0w1 in WINDOWS[kind]:
                    wl = f"w{w0w1[0]}-{w0w1[1]}"
                    shs = {c: window_shares(DD[c], kind).get(wl) for c in cf}
                    shs = {c: s for c, s in shs.items() if s is not None}
                    for i, v in enumerate(vv):
                        x = np.array([s[i] for s in shs.values()])
                        m, lo, hi = boot_ci(x)
                        W.append(dict(case=case, method=nm, window=wl, var=v, role=("main" if v in mains else
                                      "decoy" if v in decoys else "other"), n_cfg=len(x), share=m, lo=lo, hi=hi))
                    if nm == meth:
                        # rank of each main driver among all variables, and "decoy takes over" (max decoy > main)
                        for mv in mains:
                            ranks = [int((s > s[vv.index(mv)]).sum()) + 1 for s in shs.values()]
                            take = [float(max(s[vv.index(dv)] for dv in decoys) > s[vv.index(mv)]) for s in shs.values()]
                            K.append(dict(case=case, method=meth, window=wl, main=mv, n_cfg=len(ranks),
                                          rank_median=float(np.median(ranks)), frac_rank1=float(np.mean(np.array(ranks) == 1)),
                                          frac_decoy_takes_over=float(np.mean(take))))
            # per lead: share of each variable and decoy-takes-over fraction
            for Lx in range(1, max(D.shape[0] for D in Dd.values())):
                shs = [per_lead_shares(Dd[c]).get(Lx) for c in cf]
                shs = [s for s in shs if s is not None]
                if len(shs) < 3:
                    continue
                Sx = np.array(shs)
                for i, v in enumerate(vars_):
                    m, lo, hi = boot_ci(Sx[:, i])
                    L.append(dict(case=case, method=meth, lead=Lx, var=v, n_cfg=len(Sx), share=m, lo=lo, hi=hi))
                for mv in mains:
                    dmax = Sx[:, [vars_.index(dv) for dv in decoys]].max(1)
                    T.append(dict(case=case, method=meth, lead=Lx, main=mv, n_cfg=len(Sx),
                                  frac_decoy_takes_over=float((dmax > Sx[:, vars_.index(mv)]).mean())))
    for nm, X in (("a_window_shares", W), ("a_lead_shares", L), ("a_takeover_lead", T), ("a_rank_takeover", K)):
        pd.DataFrame(X).to_csv(OUT / f"{nm}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(pd.DataFrame(W).round(3).to_string(), "\n", pd.DataFrame(K).round(3).to_string(), flush=True)


# ------------------------------------------------------------------ C synthetic decoy taxonomy
def linkstats(d):
    """Pooled over regressions of the child B (block 1): parent blocks 0 = A, 2 = C, 3 = the extra variable."""
    n = 0; sel = np.zeros(4)
    for f in glob.glob(str(d / "linkstats" / "links_*.txt")):
        for line in open(f):
            p = line.split()
            if len(p) < 3 or p[0] != "1":
                continue
            n += 1
            if p[2] != "-":
                for b in map(int, p[2].split(",")):
                    if b < 4:
                        sel[b] += 1
    if n == 0:
        return {}
    other = sel[0] + sel[2] + sel[3]
    return dict(n_regressions=n, sel_rate_A=sel[0] / n, sel_rate_C=sel[2] / n, sel_rate_extra=sel[3] / n,
                fdr_var=float(sel[3] / max(other, 1.0)), power_var=float((sel[0] + sel[2]) / (2.0 * n)))


def part_c(a):
    rows = []
    for d in sorted(glob.glob(str(N4 / "c" / "*" / "c_*"))):
        d = Path(d); x = load_first(d, ["results_alphas_final.csv", "results_alphas_snapshot.csv"])
        if x is None or not len(x):
            continue
        r = dict(method=d.parent.name, run=d.name, n_cfg=len(x))
        tot = x[[c for c in x.columns if c.startswith("t7_density_")]].sum(1)
        extra = [c for c in x.columns if c.startswith("t7_density_") and c not in ("t7_density_A", "t7_density_B", "t7_density_C")]
        if len(extra) == 1:
            sh = x[extra[0]] / tot
            r.update(extra_var=extra[0].replace("t7_density_", ""), share_extra_median=float(sh.median()),
                     share_extra_mean=float(sh.mean()))
        for p_ in ("A", "C"):
            if f"t7_density_{p_}" in x:
                r[f"share_{p_}_median"] = float((x[f"t7_density_{p_}"] / tot).median())
        r["alpha_err_median"] = float(x["t7_alpha_abs_err_density_13"].median()) if "t7_alpha_abs_err_density_13" in x else np.nan
        r.update(linkstats(d))
        if "share_extra_median" in r:
            weak = min(r.get("share_A_median", np.nan), r.get("share_C_median", np.nan))
            r["discarded_rule8"] = bool(r["share_extra_median"] <= 0.10 and r["share_extra_median"] < weak)
        rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(OUT / "c_taxonomy.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(S.round(3).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["syn", "sel", "b1", "a", "c", "all"])
    ap.add_argument("--nproc", type=int, default=32)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for p, f in (("syn", part_syn), ("sel", part_sel), ("b1", part_b1), ("a", part_a), ("c", part_c)):
        if a.part in (p, "all"):
            print(f"===== {p}", flush=True)
            f(a)


if __name__ == "__main__":
    main()
