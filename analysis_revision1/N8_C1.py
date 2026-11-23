"""Night-8 C1: decoy filter on the night-6 experiment-1 runs (real cases PNW, Debby), existing outputs only.

Runs night6/t3/<case>/<v>_d<dd> (final set + one surrogate S(v), 8 configs x 20 draws x variables). Decoy = S(v) of the run
(matched to v: a stronger decoy than the typical field). Decoy share of a configuration = surrogate density over all leads
(step 0 excluded) / total density, pooled over the configuration's runs (all v, all draws; harmonized retention per run).
Rules (pre-declared): not-rank-1 (drop if decoy share > every real variable's share), equal-share (drop if decoy share > 1/p,
p = real + 1). Real shares are renormalized among the real variables.
python N8_C1.py --case pnw|debby --nproc 32  -> night8/analysis/C1_<case>_*.csv, C1_<case>.md, C1_<case>_D.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N6_T3T4 as T  # noqa: E402

OUT = T.R1 / "night8" / "analysis"


def share_tab(D, real):
    """D: (lead, n+1) pooled densities -> dict of integrated shares (all leads, step 0 excluded)."""
    tot = D[1:].sum(0)
    return tot, tot.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    case, real = a.case, T.FINAL[a.case]
    n, p = len(real), len(real) + 1
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"C1_{case}_D.npz"
    Ds = {}
    if a.reuse and f.exists():
        z = np.load(f, allow_pickle=False)
        Ds = {tuple(k.split("|")): z[k] for k in z.files}
    else:
        for d in sorted((T.N6 / "t3" / case).glob("*_d[0-9][0-9]")):
            v, draw = d.name.rsplit("_d", 1)
            if v not in real:
                continue
            Dd, _ = T.run_densities(case, d, p, False, a.nproc)
            for cfg, D in Dd.items():
                Ds[(v, draw, str(cfg))] = np.asarray(D, dtype=float)
            print(v, draw, len(Dd), flush=True)
        np.savez_compressed(f, **{"|".join(k): v for k, v in Ds.items()})
    rows = [dict(var=k[0], draw=int(k[1]), cfg=int(k[2]), D=D) for k, D in Ds.items()]
    R = pd.DataFrame(rows)

    def summ(g):
        tot = sum(D[1:].sum(0) for D in g["D"])
        s = tot / tot.sum()
        sur, rs = s[n], s[:n]
        return dict(n_runs=len(g), decoy_share=sur, top_real=real[int(np.argmax(rs))], top_real_share=rs.max(),
                    rank1=bool(sur > rs.max()), exceeds_1_over_p=bool(sur > 1 / p),
                    decoy_rank=int(1 + (rs > sur).sum()))

    # per configuration, pooled over all runs (the decision unit; documented choice)
    C = pd.DataFrame([dict(cfg=c, **summ(g)) for c, g in R.groupby("cfg")]).set_index("cfg")
    # spread over draws (pooled over v within a draw) and over runs
    sp = []
    for c, g in R.groupby("cfg"):
        per = [summ(h) for _, h in g.groupby("draw")]
        run = [summ(g.iloc[[i]]) for i in range(len(g))]
        ds = np.array([x["decoy_share"] for x in per])
        rs_ = np.array([x["decoy_share"] for x in run])
        sp.append(dict(cfg=c, n_draws=len(per), draw_mean=ds.mean(), draw_sd=ds.std(ddof=1), draw_min=ds.min(), draw_max=ds.max(),
                       frac_draws_rank1=np.mean([x["rank1"] for x in per]), frac_draws_exceed=np.mean([x["exceeds_1_over_p"] for x in per]),
                       run_sd=rs_.std(ddof=1), frac_runs_rank1=np.mean([x["rank1"] for x in run]),
                       frac_runs_exceed=np.mean([x["exceeds_1_over_p"] for x in run])))
    S = pd.DataFrame(sp).set_index("cfg")
    # per (config, matched variable), pooled over draws
    V = pd.DataFrame([dict(cfg=c, var=v, **summ(g)) for (c, v), g in R.groupby(["cfg", "var"])])
    C.to_csv(OUT / f"C1_{case}_configs.csv"); S.to_csv(OUT / f"C1_{case}_spread.csv"); V.to_csv(OUT / f"C1_{case}_cfg_var.csv", index=False)

    # real-variable shares by window, kept vs removed (pooled over each config's runs, renormalized among real)
    wins = [f"w{a_}-{b_}" for a_, b_ in T.WINDOWS[case]] + (["w11-30"] if case == "pnw" and "w11-30" not in [f"w{a_}-{b_}" for a_, b_ in T.WINDOWS[case]] else [])
    allw = list(T.WINDOWS[case]) + ([(11, 30)] if case == "pnw" and (11, 30) not in T.WINDOWS[case] else [])
    W = []
    for c, g in R.groupby("cfg"):
        Dsum = sum(D for D in g["D"])
        for (w0, w1) in allw:
            x = Dsum[w0:w1 + 1].sum(0)[:n]
            x = x / x.sum()
            W.append(dict(cfg=c, window=f"{w0}-{w1}", **{r_: x[i] for i, r_ in enumerate(real)}))
    W = pd.DataFrame(W)
    W.to_csv(OUT / f"C1_{case}_windows_by_cfg.csv", index=False)
    groups = {"all": C.index, "rank1_kept": C.index[~C.rank1], "rank1_removed": C.index[C.rank1],
              "equal_kept": C.index[~C.exceeds_1_over_p], "equal_removed": C.index[C.exceeds_1_over_p]}
    G = []
    for gname, idx in groups.items():
        for w, h in W[W.cfg.isin(idx)].groupby("window"):
            m = h[real].mean()
            G.append(dict(group=gname, n_cfg=len(idx), window=w, **m.to_dict(), top=m.idxmax(),
                          rank_order=" > ".join(m.sort_values(ascending=False).index)))
    G = pd.DataFrame(G)
    G.to_csv(OUT / f"C1_{case}_windows_groups.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        txt = [f"# C1 {case}: decoy = matched surrogate S(v) of experiment 1; p = {p} (real {n} + decoy); 1/p = {1 / p:.3f}",
               f"runs read: {len(R)} (config, v, draw) retained runs over {R.cfg.nunique()} configs",
               "Caveat: night-6 surrogate is matched to one real variable (stronger than the typical field).",
               "\n## Per configuration (pooled over all its runs: all v, all draws)\n```", C.to_string(),
               f"```\nremoved by not-rank-1: {int(C.rank1.sum())}/{len(C)}; removed by equal-share (>1/p): {int(C.exceeds_1_over_p.sum())}/{len(C)}",
               "\n## Spread over draws (pooled over v within a draw) and runs\n```", S.to_string(),
               "```\n## Per (config, matched variable), pooled over draws\n```", V.to_string(index=False),
               "```\n## Real-variable shares (renormalized among real) by window and group\n```", G.drop(columns="rank_order").to_string(index=False),
               "```\n## Order of real variables\n```", G[["group", "window", "rank_order"]].to_string(index=False), "```"]
    (OUT / f"C1_{case}.md").write_text("\n".join(txt))
    print("\n".join(txt[:8]))
    print(C.to_string())


if __name__ == "__main__":
    main()
