"""Night-9 N5 item 1: S10 with the decoy filter applied post hoc to the night-8 S10 runs (night8/s10/p06..p21; p = 3 has no
typical field = reference, unfiltered).
Decoy = the typical fields Z1..Z(k): (a) mean share per typical field, (b) the largest single typical field. Rules per
row (config, alpha): 'not rank 1' (decoy share > every other variable's share of the total, i.e. all A, B, C, response and
nonlinear fields) and 'equal share' (decoy share > 1/p). Metrics as S10: true-parent share (A + C)/all, |alpha error|.
Subsets best / random / pooled (9 configs x alpha 0.2/0.6); 95 % bootstrap CI over rows (10,000, seed 0).
-> night9/analysis/N5_S10_filter_rows.csv, N5_S10_filter_summary.csv, N5_S10_filter.md"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N8_S10 as S  # noqa: E402

OUT = S.R1 / "night9" / "analysis"
DEFS = ["mean", "max"]
RULES = ["R1", "R2"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for p in S.PS:
        d, x = S.load(p)
        if x is None:
            print("p", p, "missing"); continue
        x = x[x["trial_id"].isin(S.UNION)].reset_index(drop=True)
        names = ["A", "B", "C"] if p == 3 else S.names_for(p)
        if p == 3:
            D = pd.DataFrame({"A": x["density_integral_var1"], "B": x["density_integral_var2"], "C": x["density_integral_var3"]})
        else:
            D = pd.DataFrame({n: x[f"t7_density_{n}"] for n in names})
        tot = D.sum(1)
        typ = [n for n in names if S.fam(n) == "typical"]
        oth = [n for n in names if n not in typ]
        r = pd.DataFrame({"p": p, "trial_id": x["trial_id"], "alpha": x["alpha_mix"],
                          "share_true": ((D["A"] + D["C"]) / tot), "alpha_err": ((D["A"] / (D["A"] + D["C"])) - x["alpha_mix"]).abs()})
        if typ:
            sh = D[typ].div(tot, axis=0)
            top_other = D[oth].div(tot, axis=0).max(1)
            for dfn in DEFS:
                ds = sh.mean(1) if dfn == "mean" else sh.max(1)
                r[f"decoy_{dfn}"] = ds
                r[f"R1_{dfn}"] = ds > top_other
                r[f"R2_{dfn}"] = ds > 1.0 / p
        rows.append(r)
    R = pd.concat(rows, ignore_index=True)
    R["in_best"] = R.trial_id.isin(S.BEST); R["in_random"] = R.trial_id.isin(S.RAND)
    R.to_csv(OUT / "N5_S10_filter_rows.csv", index=False)
    summ = []
    for p, g in R.groupby("p"):
        for lab, sel in [("best", g.in_best), ("random", g.in_random), ("pooled", g.in_best | g.in_random)]:
            h = g[sel]
            variants = [("none", "-", np.ones(len(h), bool))]
            if p > 3:
                variants += [(rule, dfn, ~h[f"{rule}_{dfn}"].to_numpy(bool)) for dfn in DEFS for rule in RULES]
            for rule, dfn, keep in variants:
                k = h[keep]
                e = dict(p=p, subset=lab, rule=rule, decoy_def=dfn, n_rows=len(h), n_kept=len(k), n_removed=len(h) - len(k),
                         decoy_share_mean=float(h[f"decoy_{dfn}"].mean()) if dfn != "-" else np.nan, one_over_p=1 / p)
                for col in ("share_true", "alpha_err"):
                    m, lo, hi = S.boot(k[col], np.random.default_rng(0))   # same stream per variant: unchanged rows give identical CIs
                    e[col], e[col + "_lo"], e[col + "_hi"] = m, lo, hi
                summ.append(e)
    Sm = pd.DataFrame(summ); Sm.to_csv(OUT / "N5_S10_filter_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format, "display.max_rows", 400):
        pool = Sm[Sm.subset == "pooled"]
        txt = ["# N5 item 1: S10 with the filter applied post hoc (pooled 9 configs x alpha 0.2/0.6 = 18 rows per p)",
               "decoy = typical fields: 'mean' = mean share per typical field, 'max' = largest single typical field; R1 not rank 1, R2 share > 1/p",
               "```", pool[["p", "rule", "decoy_def", "n_rows", "n_removed", "decoy_share_mean", "one_over_p", "share_true", "share_true_lo", "share_true_hi",
                            "alpha_err", "alpha_err_lo", "alpha_err_hi"]].to_string(index=False), "```",
               "\nBest and random subsets: N5_S10_filter_summary.csv"]
        (OUT / "N5_S10_filter.md").write_text("\n".join(txt))
        print("\n".join(txt))


if __name__ == "__main__":
    main()
