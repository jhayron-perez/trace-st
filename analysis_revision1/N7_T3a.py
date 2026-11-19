"""Night-7 T3a: performance vs number of added linkless fields K in the real cases (final set + K independent linkless fields).

Runs night7/t3/<case>/K<kk>_d<dd> (K in 2, 4, 8, 16; 3 configs x 5 draws; the first K fields of draw dd's ordered
sequence of 16); K = 0 = the night-6 final-set runs night6/t3/<case>/K00 at the same 3 configs (first three of the
representative list). Shares exclude step 0; retention harmonized per run (N6_T3T4.run_densities).
Metrics per case, window and K (pre-declared):
 (i)   share of each real variable renormalized among the real variables;
 (ii)  fraction of runs in which the parent is rank 1 among all variables (Debby vo 1-4 and 5-15 d; Pinatubo SO2 4-16 d;
       PNW Z500 or TCWV 1-10 d);
 (iii) Kendall tau of the real-variable ranking against K = 0 (same config);
 (iv)  fraction of runs in which an added field outranks the parent (the larger parent for PNW);
 (v)   mean share per added field against 1/p (p = real + K variables), mean of the largest single-field share, total
       share of the added fields.
Outputs night7/analysis/N7_T3a_<case>_{long.csv.gz,table.csv}, N7_T3a_table.md; figure figures_v7/S11_realcase_vs_K/s11_realcase_vs_K.{pdf,png}.

python N7_T3a.py [--cases ...] [--nproc 16] [--partial] [--test-n6]   (PBS only)
  --test-n6: development; the night-6 exp-2 runs night6/t4/<case>/K<kk>_d<dd> in place of the night-7 runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
import N6_T3T4 as T  # noqa: E402
from A4_twins import boot_ci  # noqa: E402

R1 = T.R1
N6, N7 = R1 / "night6", R1 / "night7"
OUT = N7 / "analysis"
FIG = R1 / "figures_v7" / "S11_realcase_vs_K"
FINAL, WINDOWS, CASE_TITLE = T.FINAL, T.WINDOWS, T.CASE_TITLE
KS = [0, 2, 4, 8, 16]
CONFIGS3 = dict(debby=[246, 11, 45], pinatubo=[929, 787, 533], pnw=[205, 67, 71])
PARENTS = dict(debby=["vo"], pinatubo=["TMSO201"], pnw=["Z500", "TCWV"])
PARENT_WIN = dict(debby="w5-15", pinatubo="w4-16", pnw="w1-10")   # the headline window of the rank-1 fraction
KENDALL_MIN = 3


def kendall(a, b):
    from scipy.stats import kendalltau
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(kendalltau(a, b)[0])


def collect(case, partial, nproc, test):
    real = FINAL[case]
    base = (N6 / "t4" if test else N7 / "t3") / case
    runs = [(0, -1, N6 / "t3" / case / "K00")] + [
        (int(d.name[1:3]), int(d.name.split("_d")[1]), d) for d in sorted(base.glob("K[0-9][0-9]_d[0-9][0-9]"))
        if int(d.name[1:3]) in KS]
    rows, cnt = [], []
    for K, draw, d in runs:
        Dd, (na, nr) = T.run_densities(case, d, len(real) + K, partial, nproc)
        cnt.append(dict(case=case, K=K, draw=draw, n_configs=na, n_retained=nr))
        for cfg, D in Dd.items():
            if int(cfg) not in CONFIGS3[case] and not test:
                continue
            for w, s in T.wshares(D, case).items():
                r = dict(case=case, K=K, draw=draw, cfg=int(cfg), window=w)
                r.update({v: float(s[i]) for i, v in enumerate(real)})
                r.update({f"S{j + 1:02d}": float(s[len(real) + j]) for j in range(K)})
                rows.append(r)
    X, C = pd.DataFrame(rows), pd.DataFrame(cnt)
    X.to_csv(OUT / f"N7_T3a_{case}_long.csv.gz", index=False); C.to_csv(OUT / f"N7_T3a_{case}_counts.csv", index=False)
    return X, C


def metrics(case, X):
    real, par = FINAL[case], PARENTS[case]
    out = []
    ref = X[X["K"] == 0].set_index(["cfg", "window"])[real]
    for (K, w), g in X.groupby(["K", "window"]):
        R = g[real].to_numpy(); Rn = R / R.sum(1, keepdims=True)
        add = [c for c in g.columns if c.startswith("S") and c[1:].isdigit() and g[c].notna().any()]
        A = g[add].to_numpy() if add else np.zeros((len(g), 0))
        allsh = np.hstack([R, A])
        pi = [real.index(p) for p in par]
        parent_max = R[:, pi].max(1)
        top = allsh.argmax(1)
        taus = []
        for r_, (cfg, ww) in zip(Rn, zip(g["cfg"], g["window"])):
            if (cfg, ww) in ref.index and K > 0:
                rv = ref.loc[(cfg, ww)].to_numpy()
                taus.append(kendall(rv, r_))
        rec = dict(case=case, K=K, window=w, n_runs=len(g), p=len(real) + K,
                   parent_rank1=float(np.isin(top, pi).mean()),
                   added_outranks_parent=float((A.max(1) > parent_max).mean()) if A.shape[1] else 0.0,
                   added_total=float(A.sum(1).mean()) if A.shape[1] else 0.0,
                   added_per_field=float(A.mean()) if A.shape[1] else np.nan,
                   added_largest=float(A.max(1).mean()) if A.shape[1] else np.nan,
                   one_over_p=1.0 / (len(real) + K),
                   kendall_vs_K0=float(np.nanmean(taus)) if len(taus) and np.isfinite(taus).any() else (1.0 if K == 0 else np.nan))
        for i, v in enumerate(real):
            m, lo, hi = boot_ci(Rn[:, i])
            rec.update({f"renorm_{v}": m, f"renorm_{v}_lo": lo, f"renorm_{v}_hi": hi})
        out.append(rec)
    return pd.DataFrame(out)


def fig_s11(tabs, cases):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    S.set_style()
    cases = [c for c in cases if c in tabs]
    if not cases:
        return
    fig, ax = plt.subplots(1, len(cases), figsize=(S.DOUBLE, 2.9), squeeze=False)
    for j, case in enumerate(cases):
        a = ax[0, j]
        g = tabs[case]; g = g[g["window"] == PARENT_WIN[case]].sort_values("K")
        for v in FINAL[case]:
            lw = 1.6 if v in PARENTS[case] else 0.9
            a.plot(g["K"], g[f"renorm_{v}"], color=S.col(v), lw=lw, marker="o", ms=2.2)
            a.fill_between(g["K"], g[f"renorm_{v}_lo"], g[f"renorm_{v}_hi"], color=S.col(v), alpha=0.15, lw=0)
        a.plot(g["K"], g["parent_rank1"], color="k", ls="--", marker="s", ms=2.2, lw=1.0)
        a.plot(g["K"], g["added_total"], color="0.45", ls=":", marker="^", ms=2.2, lw=1.0)
        a.set_ylim(0, 1.02); a.set_xticks(KS); a.set_xlabel("added linkless fields K")
        a.set_title(f"{CASE_TITLE[case]}, {PARENT_WIN[case][1:].replace('-', '–')} d", fontsize=7, pad=4)
        if j == 0:
            a.set_ylabel("share / fraction of runs")
        a.grid(alpha=0.2, lw=0.4)
    hl = [Line2D([0], [0], color="k", ls="--", marker="s", ms=2.5, label="parent rank 1 (fraction of runs)"),
          Line2D([0], [0], color="0.45", ls=":", marker="^", ms=2.5, label="total share of added fields"),
          Line2D([0], [0], color="0.3", lw=1.0, marker="o", ms=2.5, label="real variables, renormalized among real (colours as Figs 4–6)")]
    fig.legend(handles=hl, loc="lower center", ncol=3, fontsize=5.5, bbox_to_anchor=(0.5, -0.02), frameon=False)
    S.panel_labels(ax.ravel())
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "s11_realcase_vs_K.pdf", bbox_inches="tight"); fig.savefig(FIG / "s11_realcase_vs_K.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    global OUT, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["pnw", "pinatubo", "debby"])
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--test-n6", action="store_true")
    a = ap.parse_args()
    if a.test_n6:
        OUT, FIG = N7 / "analysis_test", R1 / "figures_v7" / "test" / "S11_realcase_vs_K"
    OUT.mkdir(parents=True, exist_ok=True)
    tabs, md = {}, ["# Night 7 T3a: variable-count curves\n"]
    for case in a.cases:
        X, C = collect(case, a.partial or a.test_n6, a.nproc, a.test_n6)
        if not len(X):
            print(f"[{case}] no runs"); continue
        M = metrics(case, X)
        M.to_csv(OUT / f"N7_T3a_{case}_table.csv", index=False)
        tabs[case] = M
        cols = ["K", "window", "n_runs", "parent_rank1", "added_outranks_parent", "kendall_vs_K0", "added_total", "added_per_field",
                "one_over_p", "added_largest"]
        md += [f"\n## {CASE_TITLE[case]}\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in M[cols].itertuples(index=False):
            md.append("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |")
        md += ["", "retained runs per K: " + ", ".join(f"K{k}: {int(n)} of {int(t)}" for k, n, t in
                                                       C.groupby("K")[["n_retained", "n_configs"]].sum().itertuples())]
    (OUT / "N7_T3a_table.md").write_text("\n".join(md) + "\n")
    fig_s11(tabs, a.cases)
    print("\n".join(md))


if __name__ == "__main__":
    main()
