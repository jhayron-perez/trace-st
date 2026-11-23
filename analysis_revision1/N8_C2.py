"""Night-8 C2: decoy filter on the existing synthetic five-type S10 runs (fixed window: night5/t3/p05..p16, night6/t4syn/p20..p35).

Per (config, alpha) row: integrated density per variable (t7_density_*), decoy = the independent-noise fields (mean share per
noise field), real = every other variable (incl. the other types). Rules: not-rank-1 (decoy mean share > every real share),
equal-share (decoy mean share > 1/p). True-parent share = (A+C)/sum over real variables (renormalized); |alpha error| as in the
runs (A/(A+C), unchanged by the filter). Comparison: unfiltered vs filtered vs random removal of the same number of rows
(1,000 removals, seed 0). 95 % bootstrap CIs over rows (2,000 resamples, seed 0).
python N8_C2.py -> night8/analysis/C2_*.csv, C2.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N6_T4_synthetic as T4  # noqa: E402
from N5_T3 import cls  # noqa: E402

OUT = T4.R1 / "night8" / "analysis"
PS = [5, 8, 12, 16, 20, 24, 28, 35]
RULES = ["rank1", "equal"]


def rows_for(p):
    out = []
    for d in T4.run_dirs("fw", p):
        x = T4.load(d)
        if x is None:
            continue
        cols = [c for c in x.columns if c.startswith("t7_density_")]
        D = x[cols].astype(float)
        D.columns = [c.replace("t7_density_", "") for c in cols]
        out.append(pd.concat([x[["trial_id", "alpha_mix"] + (["ok_frac"] if "ok_frac" in x else [])].reset_index(drop=True), D.reset_index(drop=True)], axis=1))
    return pd.concat(out, ignore_index=True)


def metrics(df, names, noise, real):
    tot_real = df[real].sum(1)
    tp = (df["A"] + df["C"]) / tot_real
    err = ((df["A"] / (df["A"] + df["C"])) - df["alpha_mix"]).abs()
    return tp.to_numpy(), err.to_numpy()


def boot(x, rng, n=2000):
    if len(x) == 0:
        return (np.nan,) * 3
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    rng = np.random.default_rng(0)
    res, rem = [], []
    for p in PS:
        df = rows_for(p)
        n0 = len(df)
        names = [c for c in df.columns if c not in ("trial_id", "alpha_mix", "ok_frac")]
        tot = df[names].sum(1)
        ok = (tot > 0) & df[["A", "C"]].sum(1).gt(0)
        if "ok_frac" in df:
            ok &= df["ok_frac"] >= 2 / 3
        df = df[ok].reset_index(drop=True)
        noise = [c for c in names if cls(c) == "noise"]
        real = [c for c in names if cls(c) != "noise"]
        tot = df[names].sum(1)
        dshare = df[noise].sum(1) / len(noise) / tot       # mean share per noise field
        rshare = df[real].div(tot, axis=0)
        flag = {"rank1": (dshare > rshare.max(1)).to_numpy(), "equal": (dshare > 1 / p).to_numpy()}
        tp, err = metrics(df, names, noise, real)
        tp_raw = ((df["A"] + df["C"]) / tot).to_numpy()
        for alpha in ["pooled", 0.2, 0.6]:
            sel = np.ones(len(df), bool) if alpha == "pooled" else np.isclose(df["alpha_mix"].to_numpy(), alpha)
            def line(label, keep, extra=None):
                k = keep & sel
                m, lo, hi = boot(tp[k], rng); me, loe, hie = boot(err[k], rng)
                return dict(p=p, alpha=alpha, subset=label, n_rows=int(k.sum()), share_true=m, st_lo=lo, st_hi=hi,
                            alpha_err=me, ae_lo=loe, ae_hi=hie, sd_share_true=float(tp[k].std(ddof=1)) if k.sum() > 1 else np.nan,
                            sd_alpha_err=float(err[k].std(ddof=1)) if k.sum() > 1 else np.nan, share_true_raw=float(tp_raw[k].mean()) if k.any() else np.nan,
                            decoy_share_mean=float(dshare[k].mean()) if k.any() else np.nan, **(extra or {}))
            res.append(line("unfiltered", np.ones(len(df), bool)))
            for r in RULES:
                keep = ~flag[r]
                res.append(line(f"{r}_filtered", keep, dict(n_removed=int((~keep & sel).sum()))))
                # random removal of the same number of rows among the rows of this (p, alpha) selection
                idx = np.where(sel)[0]; nrem = int((~keep & sel).sum())
                if nrem == 0:
                    continue
                st, ae, sds = [], [], []
                for _ in range(1000):
                    drop = rng.choice(idx, size=nrem, replace=False)
                    k = sel.copy(); k[drop] = False
                    st.append(tp[k].mean()); ae.append(err[k].mean()); sds.append(tp[k].std(ddof=1))
                st, ae = np.array(st), np.array(ae)
                f_st, f_ae = tp[keep & sel].mean(), err[keep & sel].mean()
                rem.append(dict(p=p, alpha=alpha, rule=r, n_rows=int(sel.sum()), n_removed=nrem,
                                filt_share_true=f_st, rand_share_true=st.mean(), rand_st_lo=np.percentile(st, 2.5), rand_st_hi=np.percentile(st, 97.5),
                                frac_random_ge_filtered_st=float((st >= f_st).mean()),
                                filt_alpha_err=f_ae, rand_alpha_err=ae.mean(), rand_ae_lo=np.percentile(ae, 2.5), rand_ae_hi=np.percentile(ae, 97.5),
                                frac_random_le_filtered_ae=float((ae <= f_ae).mean()),
                                filt_sd_st=tp[keep & sel].std(ddof=1), rand_sd_st=float(np.mean(sds))))
        print(p, "rows", n0, "->", len(df), "removed rank1", int(flag["rank1"].sum()), "equal", int(flag["equal"].sum()),
              "decoy share mean", round(float(dshare.mean()), 4), "1/p", round(1 / p, 4), flush=True)
    R, Q = pd.DataFrame(res), pd.DataFrame(rem)
    R.to_csv(OUT / "C2_summary.csv", index=False); Q.to_csv(OUT / "C2_random_removal.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        txt = ["# C2 synthetic (five-type S10, fixed window, 50 configs x alpha 0.2/0.6; rows = (config, alpha))",
               "true-parent share renormalized among the real variables (decoy noise fields excluded); alpha error unchanged by the filter.",
               "\n## Pooled alphas: unfiltered vs filtered\n```", R[R.alpha == "pooled"].drop(columns=["alpha"]).to_string(index=False),
               "```\n## Filtered vs random removal of the same number of rows (1,000, seed 0), pooled alphas\n```",
               (Q[Q.alpha == "pooled"].drop(columns=["alpha"]).to_string(index=False) if len(Q) else "no configuration removed by either rule: nothing to compare with random removal"), "```"]
    (OUT / "C2.md").write_text("\n".join(txt))


if __name__ == "__main__":
    main()
