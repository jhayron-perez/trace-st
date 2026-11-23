"""Night-8 S10 (new synthetic variable-count experiment): analysis.
Rows = (config, alpha) of the 9 pre-declared configs (best 5 = 15 23 45 20 34, random 5 = 29 24 13 15 39; union 9) at p = 3 (night5/t3/p03,
restricted to the 9 configs), 6, 9, ..., 21 (night8/s10/pXX). Types per step: typical field (Z), response of V2 (R), nonlinear child (Q).
Metrics: true-parent share (A + C) / all, |alpha error| (A/(A+C)), share per family, link-level selection (fraction of B-regressions with any link from
the family's variables) vs density share. Subsets best, random, pooled (union, each config once). 95 % bootstrap CI over rows (10,000, seed 0).
-> night8/analysis/S10_{rows,summary,linkstats}.csv, S10_results.md, figures_v8/S10/S10_new.{png,pdf}"""
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_T3 as T3  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
OUT = R1 / "night8" / "analysis"
FIG = R1 / "figures_v8" / "S10"
BEST, RAND = [15, 23, 45, 20, 34], [29, 24, 13, 15, 39]
UNION = sorted(set(BEST) | set(RAND))
PS = [3, 6, 9, 12, 15, 18, 21]
FAM = ["true", "target", "typical", "response", "nonlinear"]
COL = dict(true="#000000", target="#7f7f7f", typical="#0072B2", response="#D55E00", nonlinear="#CC79A7")


def fam(n):
    return "true" if n in ("A", "C") else "target" if n == "B" else dict(Z="typical", R="response", Q="nonlinear")[n[0]]


def names_for(p):
    mix = ["typical", "response", "nonlinear"] * ((p - 3) // 3)
    cnt, out = {}, ["A", "B", "C"]
    for t in mix:
        k = cnt.get(t, 0); cnt[t] = k + 1
        out.append({"typical": f"Z{k + 1}", "response": f"R{k + 1}_B", "nonlinear": f"Q{k + 1}_{'A' if k % 2 == 0 else 'C'}"}[t])
    return out


def load(p):
    d = (R1 / "night5/t3/p03") if p == 3 else (R1 / f"night8/s10/p{p:02d}")
    for nm in ("results_alphas_final.csv", "results_alphas_snapshot.csv"):
        if (d / nm).exists():
            return d, pd.read_csv(d / nm)
    return d, None


def link_sel(d, names):
    n, sel = 0, np.zeros(len(names))
    for f in glob.glob(str(d / "linkstats" / "links_*.txt")):
        for line in open(f):
            q = line.split()
            if len(q) < 3 or q[0] != "1":
                continue
            n += 1
            if q[2] != "-":
                for b in map(int, q[2].split(",")):
                    if b < len(names):
                        sel[b] += 1
    return {k: float(sel[[i for i, v in enumerate(names) if fam(v) == k]].mean() / n) for k in FAM if k != "target" and any(fam(v) == k for v in names)} if n else {}


def boot(x, rng, n=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan,) * 3
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    rng = np.random.default_rng(0)
    rows, ls = [], []
    for p in PS:
        d, x = load(p)
        if x is None:
            print("p", p, "missing"); continue
        x = x[x["trial_id"].isin(UNION)]
        names = ["A", "B", "C"] if p == 3 else names_for(p)
        if p == 3:
            D = pd.DataFrame({"A": x["density_integral_var1"], "B": x["density_integral_var2"], "C": x["density_integral_var3"]})
        else:
            D = pd.DataFrame({n: x[f"t7_density_{n}"] for n in names})
        tot = D.sum(1)
        r = pd.DataFrame({"p": p, "trial_id": x["trial_id"].to_numpy(), "alpha": x["alpha_mix"].to_numpy()})
        for k in FAM:
            cols = [c for c in names if fam(c) == k]
            r[f"share_{k}"] = (D[cols].sum(1) / tot).to_numpy() if cols else np.nan
        r["alpha_err"] = ((D["A"] / (D["A"] + D["C"])) - x["alpha_mix"]).abs().to_numpy()
        r["ok_frac"] = x["ok_frac"].to_numpy() if "ok_frac" in x else np.nan
        rows.append(r)
        if p > 3:
            ls.append(dict(p=p, **link_sel(d, names)))
    R = pd.concat(rows, ignore_index=True)
    R["in_best"] = R.trial_id.isin(BEST); R["in_random"] = R.trial_id.isin(RAND)
    R.to_csv(OUT / "S10_rows.csv", index=False)
    summ = []
    for p, g in R.groupby("p"):
        for lab, sel in [("best", g.in_best), ("random", g.in_random), ("pooled", g.in_best | g.in_random)]:
            h = g[sel]
            e = dict(p=p, subset=lab, n_rows=len(h), n_configs=h.trial_id.nunique())
            for k in FAM + ["alpha_err"]:
                col = k if k == "alpha_err" else f"share_{k}"
                m, lo, hi = boot(h[col], rng)
                e[col], e[col + "_lo"], e[col + "_hi"] = m, lo, hi
            summ.append(e)
    S = pd.DataFrame(summ); S.to_csv(OUT / "S10_summary.csv", index=False)
    L = pd.DataFrame(ls); L.to_csv(OUT / "S10_linkstats.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        keep = ["p", "subset", "n_rows", "share_true", "share_true_lo", "share_true_hi", "alpha_err", "alpha_err_lo", "alpha_err_hi",
                "share_typical", "share_response", "share_nonlinear"]
        txt = ["# S10 (new): typical field + response + nonlinear child, 9 configs (best 5 + random 5, one overlap) x alpha 0.2/0.6, M 20, linear rule",
               "true-parent share = (A + C) / all variables; alpha error = |A/(A+C) - alpha|; CI = 95 % bootstrap over (config, alpha) rows", "```",
               S[keep].to_string(index=False), "```", "link-level selection (fraction of B-regressions with any link from the family; p > 3)", "```", L.to_string(index=False), "```",
               "density share per family (pooled): see table above (typical, response, nonlinear); 1/p reference: " + ", ".join(f"p{p}: {1 / p:.3f}" for p in PS)]
    (OUT / "S10_results.md").write_text("\n".join(txt)); print("\n".join(txt))
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    sty = dict(best=("-o", 1.0), random=("--s", 1.0), pooled=("-", 2.2))
    for lab in ("best", "random", "pooled"):
        h = S[S.subset == lab].sort_values("p")
        for a, col, tt in [(ax[0], "share_true", "True-parent share (A + C)"), (ax[1], "alpha_err", "|alpha error|")]:
            a.plot(h.p, h[col], sty[lab][0], color="k" if lab == "pooled" else ("#555555" if lab == "best" else "#999999"), lw=sty[lab][1], label=lab)
            if lab == "pooled":
                a.fill_between(h.p, h[col + "_lo"], h[col + "_hi"], color="k", alpha=0.15, lw=0)
            a.set_title(tt, fontsize=10); a.set_xlabel("p (variables)")
    ax[0].legend(frameon=False, fontsize=8)
    h = S[S.subset == "pooled"].sort_values("p")
    for k in FAM:
        ax[2].plot(h.p, h[f"share_{k}"], "-o", color=COL[k], label=k, ms=4)
    ax[2].set_title("Density share per family (pooled)", fontsize=10); ax[2].set_xlabel("p (variables)"); ax[2].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "S10_new.png", dpi=200); fig.savefig(FIG / "S10_new.pdf")


if __name__ == "__main__":
    main()
