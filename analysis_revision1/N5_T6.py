"""Night-5 T6: surrogate sweep (Gaussian-random-field surrogates added to the extended set), baseline method, linear rule.

Runs: night5/t6/<case>/K00 (extended set only) and K{02,04,08,16}_d{00..19}, each with the 8 rule-5 configs, M 30.
Variables per run = extended set + the first K round-robin surrogate slots (names from surrogates/<case>/run_args.txt;
surrogate S<nn><SRC> has the type of its source variable SRC).
Per (config, draw, window) shares are computed from the run bundles (step 0 excluded), as in N5_T2.
Outputs per case and window:
- degradation: mean share per real variable vs K, the same renormalized among the real variables, total surrogate
  share, and rank-1 stability = fraction of (config, draw) whose top real variable equals that config's K = 0 top;
- surrogate null per variable type: 2.5/50/97.5 (and 95th) percentiles of the individual surrogate shares of that
  type over draws x configs (x surrogates of the type), at each K and pooled over K > 0; a real variable is
  "above the null" if its mean share in the same runs exceeds the 95th percentile of its type's null (rule 7);
- PNW 11-30 d (OLR, SLHF, SSHF) as the pre-agreed statement.
python N5_T6.py --case pnw [--partial]   (PBS only) -> night5/analysis/T6_<case>_*.csv, figures_v5/surrogates/
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
from A4_twins import boot_ci  # noqa: E402
from N3_bed_summary import densities  # noqa: E402
from N5_T2 import CASES, load_table  # noqa: E402

N5 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/night5")
T6 = N5 / "t6"
SUR = N5 / "surrogates"
OUT = N5 / "analysis"
WINDOWS = dict(debby=[(1, 4), (5, 15)], pinatubo=[(1, 3), (4, 16)], pnw=[(1, 10), (11, 20), (21, 30), (11, 30)])
PNW_STEM = dict(z500="Z500", z10="Z10", olr="MTNLWRF", tcwv="TCWV", slhf="MSLHF", sshf="MSSHF", t2m="T2M", blh="BLH")
PIN_SRC = {"AERODv": "AEROD_v"}


def slot_names(case, K):
    """Variable names of the K surrogate slots, in loader order, and their source types."""
    line = [l for l in open(SUR / case / "run_args.txt") if l.startswith(f"{K}|0|")][0].split("|", 2)[2].split()
    if case == "debby":
        ex = line[line.index("--extra-inputs") + 1:]
        ex = [e for e in ex if not e.startswith("--")]
        names = [e.split("=")[0] for e in ex][4:]       # after CAPE, DIV200, WS10, MSLP
    else:
        ex = line[line.index("--extra-vars") + 1: line.index("--extra-path")]
        names = ex[2:] if case == "pnw" else ex[5:]
    types = []
    for n in names:
        src = n[3:]
        types.append(PNW_STEM[src] if case == "pnw" else PIN_SRC.get(src, src))
    return names, types


def run_dirs(case):
    out = []
    for d in sorted((T6 / case).glob("K*")):
        nm = d.name
        K = int(nm[1:3]); draw = int(nm.split("_d")[1]) if "_d" in nm else -1
        out.append((K, draw, d))
    return out


def shares(D, w0, w1):
    Dw = D[w0: w1 + 1].sum(0)
    return Dw / Dw.sum() if Dw.sum() > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list(CASES))
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--configs", type=int, nargs="+", default=None,
                    help="restrict to these configs (same set at every K); outputs get suffix _cfg<n>")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    case = a.case
    real = CASES[case]["ext"][1]
    rows = []
    for K, draw, d in run_dirs(case):
        names, types = slot_names(case, K) if K > 0 else ([], [])
        vars_ = real + names
        idc = "run_id" if case == "pnw" else "trial_id"
        f = d / ("hyperparam_search_summary.csv" if case == "pnw" else "results_jobs_final.csv")
        if not f.exists() and a.partial:
            f = d / "results_jobs_snapshot.csv"
        if case == "pnw":
            ids = sorted(int(p.name.split("_")[1]) for p in d.glob("run_*") if (p / "ensemble_results.pkl").exists())
        else:
            if not f.exists():
                continue
            ids = sorted(pd.read_csv(f)[idc].astype(int))
        Dd, _ = densities(case, d, len(vars_), ids, a.nproc, 0)
        for cfg, D in Dd.items():
            for (w0, w1) in WINDOWS[case]:
                s = shares(D, w0, w1)
                if s is None:
                    continue
                r = dict(case=case, K=K, draw=draw, cfg=cfg, window=f"w{w0}-{w1}")
                for i, v in enumerate(real):
                    r[v] = s[i]
                r["sur_total"] = float(s[len(real):].sum())
                for j, (n, ty) in enumerate(zip(names, types)):
                    r[f"sur:{ty}:{n}"] = s[len(real) + j]
                rows.append(r)
    if not rows:
        print("no runs"); return
    X = pd.DataFrame(rows)
    tag = ""
    if a.configs:
        X = X[X["cfg"].isin(a.configs)].copy(); tag = f"_cfg{len(a.configs)}"
        print("configs restricted to", a.configs, "rows", len(X))
    X.to_csv(OUT / f"T6_{case}{tag}_long.csv.gz", index=False)
    # --- degradation
    base_top = {}
    for _, r in X[X["K"] == 0].iterrows():
        base_top[(r["cfg"], r["window"])] = real[int(np.argmax([r[v] for v in real]))]
    deg = []
    for (K, w), g in X.groupby(["K", "window"]):
        R = g[real].to_numpy()
        Rn = R / R.sum(1, keepdims=True)
        tops = [real[i] for i in R.argmax(1)]
        same = [t == base_top.get((c, w)) for t, c in zip(tops, g["cfg"])]
        for i, v in enumerate(real):
            m, lo, hi = boot_ci(R[:, i]); mn, lon, hin = boot_ci(Rn[:, i])
            deg.append(dict(case=case, K=K, window=w, var=v, n=len(g), share=m, lo=lo, hi=hi, share_renorm=mn,
                            lo_renorm=lon, hi_renorm=hin, frac_rank1=float(np.mean([t == v for t in tops]))))
        m, lo, hi = boot_ci(g["sur_total"].to_numpy())
        deg.append(dict(case=case, K=K, window=w, var="surrogates_total", n=len(g), share=m, lo=lo, hi=hi,
                        frac_rank1=float(np.mean(same)) if base_top else np.nan))
    DEG = pd.DataFrame(deg); DEG.to_csv(OUT / f"T6_{case}{tag}_degradation.csv", index=False)
    # --- null per type
    nul = []
    scol = [c for c in X.columns if c.startswith("sur:")]
    for Ksel in sorted(X["K"].unique().tolist()) + ["pooled"]:
        if Ksel == 0:
            continue
        g0 = X[X["K"] > 0] if Ksel == "pooled" else X[X["K"] == Ksel]
        for w, g in g0.groupby("window"):
            for v in real:
                cols = [c for c in scol if c.split(":")[1] == v]
                vals = g[cols].to_numpy().ravel() if cols else np.array([])
                vals = vals[np.isfinite(vals)]
                if not len(vals):
                    continue
                p = np.percentile(vals, [2.5, 50, 95, 97.5])
                rs = float(g[v].mean())
                nul.append(dict(case=case, K=Ksel, window=w, var=v, n_null=len(vals), null_p025=p[0], null_p50=p[1],
                                null_p95=p[2], null_p975=p[3], real_share=rs, above_null=bool(rs > p[2]),
                                frac_cfg_draw_above_p95=float((g[v] > p[2]).mean())))
    NUL = pd.DataFrame(nul); NUL.to_csv(OUT / f"T6_{case}{tag}_null.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(DEG.pivot_table(index=["window", "var"], columns="K", values="share").round(3).to_string())
        print(DEG[DEG["var"] == "surrogates_total"][["window", "K", "share", "frac_rank1"]].round(3).to_string())
        print(NUL[NUL["K"].astype(str).isin(["16", "pooled"])].round(3).to_string())
    if case == "pnw":
        q = NUL[(NUL["K"].astype(str) == "pooled") & (NUL["window"] == "w11-30") & NUL["var"].isin(["MTNLWRF", "MSLHF", "MSSHF"])]
        stm = {r["var"]: dict(real=r["real_share"], null_p95=r["null_p95"], above=r["above_null"]) for _, r in q.iterrows()}
        json.dump(stm, open(OUT / "T6_pnw_11_30_statement.json", "w"), indent=1, default=float)
        print("PNW 11-30 d:", stm)
    figure(case, DEG, NUL, real, tag)


def figure(case, DEG, NUL, real, tag=""):
    import matplotlib.pyplot as plt
    S.set_style()
    head = dict(debby="w5-15", pinatubo="w4-16", pnw="w1-10")[case]
    wins = [f"w{a}-{b}" for a, b in WINDOWS[case]]
    fig, axs = plt.subplots(1, 4, figsize=(S.DOUBLE, 0.4 * S.DOUBLE), gridspec_kw=dict(width_ratios=[1.1, 0.8, 0.8, 1.5]))
    d = DEG[(DEG["window"] == head)]
    Ks = sorted(d["K"].unique())
    for v in real:
        g = d[d["var"] == v].sort_values("K")
        axs[0].errorbar(g["K"], g["share_renorm"], yerr=[g["share_renorm"] - g["lo_renorm"], g["hi_renorm"] - g["share_renorm"]],
                        color=S.col(v), marker="o", ms=2.5, lw=1, capsize=1, label=S.lab(v))
    axs[0].set_xlabel("Surrogates added, K"); axs[0].set_ylabel("Share among real variables")
    axs[0].set_xticks(Ks); axs[0].legend(fontsize=5, ncol=4, loc="upper left", bbox_to_anchor=(0.0, -0.28), columnspacing=0.6, handlelength=1.2)
    g = d[d["var"] == "surrogates_total"].sort_values("K")
    axs[1].errorbar(g["K"], g["share"], yerr=[g["share"] - g["lo"], g["hi"] - g["share"]], color=S.ROLE_COLORS["surrogate"],
                    marker="o", ms=2.5, lw=1, capsize=1)
    axs[1].set_xlabel("K"); axs[1].set_ylabel("Total surrogate share"); axs[1].set_xticks(Ks); axs[1].set_ylim(0, None)
    axs[2].plot(g["K"], g["frac_rank1"], "o-", color="k", ms=2.5)
    axs[2].set_xlabel("K"); axs[2].set_ylabel("Same top variable as K = 0"); axs[2].set_ylim(0, 1.02); axs[2].set_xticks(Ks)
    nw = NUL[(NUL["K"].astype(str) == "16")]
    ax = axs[3]
    xw = {"pnw": "w11-30"}.get(case, head)
    for k, wl in enumerate([head] + ([xw] if xw != head else [])):
        q = nw[nw["window"] == wl]
        for i, v in enumerate(real):
            r = q[q["var"] == v]
            if not len(r):
                continue
            x = i + 0.2 * (k - 0.5) * (1 if xw != head else 0)
            r = r.iloc[0]
            ax.vlines(x, r["null_p025"], r["null_p975"], color="0.6", lw=3 if k == 0 else 1.5)
            ax.plot(x, r["null_p95"], "_", color="0.3", ms=6)
            ax.plot(x, r["real_share"], "o" if k == 0 else "s", color=S.col(v), ms=3.5,
                    mec="k" if r["above_null"] else S.col(v), mew=0.6)
    ax.set_xticks(range(len(real))); ax.set_xticklabels([S.lab(v) for v in real], rotation=60, ha="right", fontsize=5.5)
    ax.set_ylabel("Share of density (K = 16)")
    ttl = ("markers: real (black edge = above null p95, tick)\nbars: null 2.5–97.5 %"
           + (f"\n{head[1:].replace('-', '–')} d circles, {xw[1:].replace('-', '–')} d squares" if xw != head else ""))
    ax.set_title(ttl, fontsize=5.2)
    S.panel_labels(axs, x=-0.32, y=1.06)
    fig.suptitle(f"{CASES[case]['title']}: surrogate sweep", x=0.02, ha="left", fontsize=8, fontweight="bold")
    fig.tight_layout()
    print("fig", S.save(fig, "surrogates", f"{case}_surrogate_sweep{tag}"))


if __name__ == "__main__":
    main()
