"""Night-6 T3 / T4: surrogate experiments on the final variable sets (baseline method, linear rule, M 30 unless stated).

Experiment 1 (T3, paired one-at-a-time significance; pre-declared rule 5):
  runs night6/t3/<case>/K00 (final set, the 8 representative configs) and night6/t3/<case>/<v>_d<dd>
  (final set + one surrogate S(v) of variable v, draw dd; S(v) is the last variable of the run).
  Paired excess e = share(v) - share(S(v)) in the same run, per window and per lead day; "above the null" if the
  95 % bootstrap CI (10,000 resamples of (config, draw) pairs, seed 0) of mean(e) has lower bound > 0.
  Pinatubo SO2/SO4 also against null (i), the tag-03 field of the K00 runs (e = SO2 - TMSO203, SO4 - BURDENSO403,
  bootstrap over configs).
  Null band for panel C of Figs 4-6: 5/50/95 % of share(S(v)) over (config, draw) per lead day ->
  night6/analysis/T3_nullband_<case>.csv (case, variable, lead_day, p05, p50, p95).
Experiment 2 (T4, performance vs number of variables; rule 7):
  runs night6/t4/<case>/K<kk>_d<dd> (final set + surrogate slots 1..K of draw dd's ordered sequence); K = 0 is the
  experiment-1 K00 run. Per K and window: share of each real variable (raw and renormalized among the real
  variables), total surrogate share, fraction of (config, draw) where the parent is rank 1 among all variables.
Retention: harmonized (trace_st.retention.harmonized_case) per run; dropped (config, run) pairs are counted.
Shares exclude step 0 (as N5_T2/N5_T6).

python N6_T3T4.py --case pnw [--exp 1 2] [--partial]   (PBS only)
  -> night6/analysis/T3_<case>_*.csv, T4_<case>_*.csv, figures_v6/surrogates/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
from A4_twins import boot_ci  # noqa: E402
from N3_bed_summary import densities  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
N6 = R1 / "night6"
OUT = N6 / "analysis"
FIG = R1 / "figures_v6" / "surrogates"
WINDOWS = dict(debby=[(1, 4), (5, 15)], pinatubo=[(1, 3), (4, 16)], pnw=[(1, 10), (11, 20), (21, 30), (11, 30)])
# final sets in loader order (base variables, then the --extra order); names as in the run outputs
FINAL = dict(debby=["Tb", "vo", "precip", "CAPE", "DIV200", "MSLP"],
             pinatubo=["TMSO201", "BURDENSO401", "TMH2SO401", "AEROD_v", "TMSO203", "BURDENSO403", "T050"],
             pnw=["Z500", "Z10", "MTNLWRF", "TCWV", "MSLHF", "MSSHF", "T2M"])
CASE_TITLE = dict(debby="TS Debby 2006", pinatubo="Pinatubo 1991", pnw="PNW heat wave 2021")
PARENT = dict(debby=(["vo"], "w5-15"), pinatubo=(["TMSO201"], "w4-16"), pnw=(["Z500", "TCWV"], "w1-10"))
NULL_I = dict(TMSO201="TMSO203", BURDENSO401="BURDENSO403")


def table_and_ids(case, d, partial):
    """Run table of one run directory, retained ids (harmonized) and all ids."""
    t = _load(case, d, partial)
    if t is None or not len(t):
        return None, [], []
    idc = "run_id" if case == "pnw" else "trial_id"
    keep = retention.harmonized_case(case, t)
    return t, sorted(t.loc[keep, idc].astype(int)), sorted(t[idc].astype(int))


def _load(case, d, partial):
    if case == "pnw":
        f = d / "hyperparam_search_summary.csv"
        if f.exists():
            return pd.read_csv(f)
        if partial:
            import json
            js = [json.load(open(p)) for p in sorted(d.glob("run_*/summary.json"))]
            return pd.DataFrame(js) if js else None
        return None
    for nm in (["results_jobs_final.csv"] + (["results_jobs_snapshot.csv"] if partial else [])):
        if (d / nm).exists():
            return pd.read_csv(d / nm)
    return None


def run_densities(case, d, n_vars, partial, nproc):
    """{cfg: D (lead, var)} for the retained configs of run d, and (n_all, n_retained)."""
    t, ids, all_ids = table_and_ids(case, d, partial)
    if t is None:
        return {}, (0, 0)
    Dd, _ = densities(case, d, n_vars, ids, nproc, 0)
    return Dd, (len(all_ids), len(ids))


def wshares(D, case):
    out = {}
    for (w0, w1) in WINDOWS[case]:
        Dw = D[w0: w1 + 1].sum(0)
        if Dw.sum() > 0:
            out[f"w{w0}-{w1}"] = Dw / Dw.sum()
    return out


def lead_shares(D):
    out = {}
    for L in range(1, D.shape[0]):
        if D[L].sum() > 0:
            out[L] = D[L] / D[L].sum()
    return out


def boot_pairs(x, n_boot=10000, seed=0):
    return boot_ci(x, n_boot=n_boot, seed=seed)


# -----------------------------------------------------------------------------
def exp1(case, partial, nproc):
    real = FINAL[case]
    base = N6 / "t3" / case
    rows, lrows, cnt = [], [], []
    for d in sorted(base.glob("*_d[0-9][0-9]")):
        v, draw = d.name.rsplit("_d", 1)
        if v not in real:
            continue
        Dd, (na, nr) = run_densities(case, d, len(real) + 1, partial, nproc)
        cnt.append(dict(case=case, var=v, draw=int(draw), n_configs=na, n_retained=nr))
        iv, js = real.index(v), len(real)
        for cfg, D in Dd.items():
            for w, s in wshares(D, case).items():
                rows.append(dict(case=case, var=v, draw=int(draw), cfg=cfg, window=w, share_v=s[iv], share_sur=s[js],
                                 excess=s[iv] - s[js]))
            for L, s in lead_shares(D).items():
                lrows.append(dict(case=case, var=v, draw=int(draw), cfg=cfg, lead_day=L, share_v=s[iv],
                                  share_sur=s[js], excess=s[iv] - s[js]))
    if not rows and case != "pinatubo":
        print(f"[{case}] exp1: no runs"); return None
    X = pd.DataFrame(rows, columns=["case", "var", "draw", "cfg", "window", "share_v", "share_sur", "excess"])
    XL = pd.DataFrame(lrows, columns=["case", "var", "draw", "cfg", "lead_day", "share_v", "share_sur", "excess"])
    C = pd.DataFrame(cnt, columns=["case", "var", "draw", "n_configs", "n_retained"])
    X.to_csv(OUT / f"T3_{case}_long.csv.gz", index=False); XL.to_csv(OUT / f"T3_{case}_lead_long.csv.gz", index=False)
    C.to_csv(OUT / f"T3_{case}_counts.csv", index=False)
    res = []
    for (v, w), g in X.groupby(["var", "window"], sort=False):
        m, lo, hi = boot_pairs(g["excess"].to_numpy())
        p = np.percentile(g["share_sur"], [5, 50, 95])
        res.append(dict(case=case, var=v, window=w, null="surrogate", n_pairs=len(g), n_draws=g["draw"].nunique(),
                        n_cfg=g["cfg"].nunique(), share_v=float(g["share_v"].mean()), share_sur=float(g["share_sur"].mean()),
                        sur_p05=p[0], sur_p50=p[1], sur_p95=p[2], excess=m, lo=lo, hi=hi, above_null=bool(lo > 0)))
    # Pinatubo null (i): tag-03 fields in the K00 runs (final set, same configs)
    if case == "pinatubo":
        Dd, _ = run_densities(case, base / "K00", len(real), partial, nproc)
        for v, v0 in NULL_I.items():
            for w in [f"w{a}-{b}" for a, b in WINDOWS[case]]:
                e, sv, s0 = [], [], []
                for cfg, D in Dd.items():
                    s = wshares(D, case).get(w)
                    if s is not None:
                        e.append(s[real.index(v)] - s[real.index(v0)]); sv.append(s[real.index(v)]); s0.append(s[real.index(v0)])
                if e:
                    m, lo, hi = boot_pairs(np.array(e))
                    res.append(dict(case=case, var=v, window=w, null=f"tag-03 ({v0})", n_pairs=len(e), n_draws=0,
                                    n_cfg=len(e), share_v=float(np.mean(sv)), share_sur=float(np.mean(s0)),
                                    excess=m, lo=lo, hi=hi, above_null=bool(lo > 0)))
    R = pd.DataFrame(res); R.to_csv(OUT / f"T3_{case}_excess.csv", index=False)
    # per lead day: excess with CI and the null band
    band, lres = [], []
    for (v, L), g in XL.groupby(["var", "lead_day"]):
        p = np.percentile(g["share_sur"], [5, 50, 95])
        band.append(dict(case=case, variable=v, lead_day=int(L), p05=p[0], p50=p[1], p95=p[2], n=len(g)))
        m, lo, hi = boot_pairs(g["excess"].to_numpy())
        lres.append(dict(case=case, var=v, lead_day=int(L), excess=m, lo=lo, hi=hi, above_null=bool(lo > 0), n_pairs=len(g)))
    if band:   # no surrogate runs (Pinatubo while its generator is stopped): no null band file
        pd.DataFrame(band).to_csv(OUT / f"T3_nullband_{case}.csv", index=False)
    pd.DataFrame(lres).to_csv(OUT / f"T3_{case}_excess_by_lead.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(R.reindex(columns=["var", "window", "null", "n_pairs", "share_v", "share_sur", "sur_p95", "excess", "lo", "hi", "above_null"]).round(4).to_string())
        print(C.groupby("var")[["n_configs", "n_retained"]].sum().to_string())
    return R


def exp2(case, partial, nproc):
    real = FINAL[case]
    rows, cnt = [], []
    runs = [(0, -1, N6 / "t3" / case / "K00")] + [(int(d.name[1:3]), int(d.name.split("_d")[1]), d)
                                                 for d in sorted((N6 / "t4" / case).glob("K[0-9][0-9]_d[0-9][0-9]"))]
    for K, draw, d in runs:
        Dd, (na, nr) = run_densities(case, d, len(real) + K, partial, nproc)
        cnt.append(dict(case=case, K=K, draw=draw, n_configs=na, n_retained=nr))
        for cfg, D in Dd.items():
            for w, s in wshares(D, case).items():
                r = dict(case=case, K=K, draw=draw, cfg=cfg, window=w, sur_total=float(s[len(real):].sum()),
                         top=(real + [f"S{j + 1:02d}" for j in range(K)])[int(np.argmax(s))])
                for i, v in enumerate(real):
                    r[v] = s[i]
                rows.append(r)
    if not rows:
        print(f"[{case}] exp2: no runs"); return None
    X, C = pd.DataFrame(rows), pd.DataFrame(cnt)
    X.to_csv(OUT / f"T4_{case}_long.csv.gz", index=False); C.to_csv(OUT / f"T4_{case}_counts.csv", index=False)
    par, pw = PARENT[case]
    deg = []
    for (K, w), g in X.groupby(["K", "window"]):
        R = g[real].to_numpy(); Rn = R / R.sum(1, keepdims=True)
        for i, v in enumerate(real):
            m, lo, hi = boot_ci(R[:, i]); mn, lon, hin = boot_ci(Rn[:, i])
            deg.append(dict(case=case, K=K, window=w, var=v, n=len(g), share=m, lo=lo, hi=hi, share_renorm=mn,
                            lo_renorm=lon, hi_renorm=hin, frac_rank1=float((g["top"] == v).mean())))
        m, lo, hi = boot_ci(g["sur_total"].to_numpy())
        deg.append(dict(case=case, K=K, window=w, var="surrogates_total", n=len(g), share=m, lo=lo, hi=hi,
                        frac_rank1=float(g["top"].str.startswith("S").mean())))
        deg.append(dict(case=case, K=K, window=w, var="parent_rank1", n=len(g), share=np.nan,
                        frac_rank1=float(g["top"].isin(par).mean())))
    DEG = pd.DataFrame(deg); DEG.to_csv(OUT / f"T4_{case}_degradation.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(DEG.pivot_table(index=["window", "var"], columns="K", values="share_renorm").round(3).to_string())
        q = DEG[DEG["var"].isin(["surrogates_total", "parent_rank1"])]
        print(q.pivot_table(index=["window", "var"], columns="K", values=["share", "frac_rank1"]).round(3).to_string())
        print(C.groupby("K")[["n_configs", "n_retained"]].sum().to_string())
    return DEG


# -----------------------------------------------------------------------------
def fig_s11(cases):
    """S11: real-case curves vs K (renormalized real shares, surrogate total, parent rank-1), one column per case."""
    import matplotlib.pyplot as plt
    S.set_style()
    cases = [c for c in cases if (OUT / f"T4_{c}_degradation.csv").exists()]
    if not cases:
        return
    fig, ax = plt.subplots(2, len(cases), figsize=(S.DOUBLE, 2.3 * 2), squeeze=False)
    for j, case in enumerate(cases):
        DEG = pd.read_csv(OUT / f"T4_{case}_degradation.csv")
        par, pw = PARENT[case]
        g = DEG[DEG["window"] == pw]
        a = ax[0, j]
        for v in FINAL[case]:
            q = g[g["var"] == v].sort_values("K")
            a.errorbar(q["K"], q["share_renorm"], yerr=[q["share_renorm"] - q["lo_renorm"], q["hi_renorm"] - q["share_renorm"]],
                       color=S.col(v), lw=1.0 if v not in par else 1.6, marker="o", ms=2.5, capsize=1.5,
                       label=S.lab(v))
        a.set_title(f"{CASE_TITLE[case]}, {pw[1:].replace('-', '–')} d", fontsize=7)
        a.set_ylabel("share among real variables" if j == 0 else ""); a.set_xticks([0, 4, 8, 16, 24, 32])
        a.legend(fontsize=5, ncol=2, frameon=False, loc="upper right")
        b = ax[1, j]
        q = g[g["var"] == "surrogates_total"].sort_values("K")
        b.errorbar(q["K"], q["share"], yerr=[q["share"] - q["lo"], q["hi"] - q["share"]], color="0.35", marker="s", ms=2.5,
                   capsize=1.5, label="surrogates, total share")
        q = g[g["var"] == "parent_rank1"].sort_values("K")
        b.plot(q["K"], q["frac_rank1"], color="k", marker="o", ms=2.5, label="parent rank 1 (fraction of runs)")
        b.set_ylim(0, 1.02); b.set_xlabel("number of surrogate fields K"); b.set_xticks([0, 4, 8, 16, 24, 32])
        b.legend(fontsize=5, frameon=False, loc="center right")
    S.panel_labels(ax.ravel())
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "S11_realcase_vs_K.pdf", bbox_inches="tight"); fig.savefig(FIG / "S11_realcase_vs_K.png", dpi=300, bbox_inches="tight")


def above_table(cases):
    """Supplement table: above the null (single rule) for every variable x window."""
    lines = ["| Case | Variable | Window | Null | Share(v) | Share(S) | Excess [95 % CI] | n pairs | Above |", "|---|---|---|---|---|---|---|---|---|"]
    for case in cases:
        f = OUT / f"T3_{case}_excess.csv"
        if not f.exists():
            continue
        for _, r in pd.read_csv(f).iterrows():
            lines.append(f"| {case} | {r['var']} | {r['window'][1:]} d | {r['null']} | {r['share_v']:.3f} | {r['share_sur']:.3f} | "
                         f"{r['excess']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {int(r['n_pairs'])} | {'yes' if r['above_null'] else 'no'} |")
    (OUT / "T3_above_null_table.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs="+", default=["pnw", "pinatubo", "debby"])
    ap.add_argument("--exp", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for case in a.case:
        if 1 in a.exp:
            exp1(case, a.partial, a.nproc)
        if 2 in a.exp:
            exp2(case, a.partial, a.nproc)
    above_table(["pnw", "pinatubo", "debby"])
    fig_s11(["pnw", "pinatubo", "debby"])


if __name__ == "__main__":
    main()
