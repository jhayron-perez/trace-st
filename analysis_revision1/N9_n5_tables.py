"""Night-9 N5 items 2 and 3 (existing outputs; Debby from N1 when night9/analysis/N1_debby_configs.csv exists).

Item 2: per case, of the harmonized-retained configs of the decoy run (final set + typical-field pool, M 30, linear rule):
number and % in which the decoy is rank 1 (decoy share > every real variable's share of the total = rule 'not rank 1' drop)
and in which it exceeds 1/p (equal-share drop), median / range of the decoy share vs 1/p (p = real + 1).
Item 3: final-set headline table. Framing A = decoy runs with 'not rank 1' applied (own retained configs minus R1 drops; real
shares renormalized among the real variables; 95 % bootstrap CI over configs, 2,000 resamples, seed 0, rank-1 fraction with
CI); A0 = the same runs without the criterion; B = night-6 final-set runs (night6/analysis/T2_window_shares.csv, own retained
configs, shares and rank-1 fraction as in Figs 4-6; no CI on the rank-1 fraction there).
Input config tables (T1b_<case>_configs.csv; N1_debby_configs.csv must have the same columns: id, decoy_share, R1_drop, R2_drop,
s_<w0>-<w1>_<var>).
-> night9/analysis/N5_decoy_rank1_counts.{csv,md}, N5_headline_table.{csv,md}"""
from pathlib import Path

import numpy as np
import pandas as pd

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
A8 = R1 / "night8" / "analysis"
A9 = R1 / "night9" / "analysis"
FINAL = dict(pnw=["Z500", "Z10", "OLR", "TCWV", "SLHF", "SSHF", "T2m"],
             pinatubo=["TMSO201", "BURDENSO401", "TMH2SO401", "AEROD_v", "TMSO203", "BURDENSO403", "T050"],
             debby=["Tb", "vo", "precip", "CAPE", "DIV200", "MSLP"])
WINS = dict(pnw=["1-10", "11-20", "21-30", "11-30"], pinatubo=["1-3", "4-16"], debby=["1-4", "5-15"])
# night-6 names (T2_window_shares.csv) -> names of the decoy-run tables
B_NAMES = dict(pnw={"Z500": "Z500", "Z10": "Z10", "MTNLWRF": "OLR", "TCWV": "TCWV", "MSLHF": "SLHF", "MSSHF": "SSHF", "T2M": "T2m"},
               pinatubo={v: v for v in FINAL["pinatubo"]}, debby={v: v for v in FINAL["debby"]})
TITLE = dict(pnw="PNW heat wave 2021", pinatubo="Pinatubo 1991", debby="TS Debby 2006")


def cfg_path(case):
    return (A8 / f"T1b_{case}_configs.csv") if case != "debby" else (A9 / "N1_debby_configs.csv")


def boot(x, rng, n=2000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x):
        return (np.nan,) * 3
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def shares_rank1(C, case, w, rng):
    cols = [f"s_{w}_{v}" for v in FINAL[case]]
    S = C[cols].to_numpy(float)
    S = S[np.isfinite(S).all(1)]
    top = np.argmax(S, 1)
    out = {}
    for j, v in enumerate(FINAL[case]):
        m, lo, hi = boot(S[:, j], rng)
        r, rlo, rhi = boot((top == j).astype(float), rng)
        out[v] = dict(n=len(S), share=m, lo=lo, hi=hi, rank1=r, r1_lo=rlo, r1_hi=rhi)
    return out


def main():
    A9.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cnt, head = [], []
    T2 = pd.read_csv(R1 / "night6/analysis/T2_window_shares.csv")
    for case in ("pnw", "pinatubo", "debby"):
        f = cfg_path(case)
        if not f.exists():
            print(case, "decoy-run table missing:", f, flush=True)
            continue
        C = pd.read_csv(f)
        C = C[C["decoy_share"].notna()]
        p = len(FINAL[case]) + 1
        ds = C["decoy_share"].to_numpy()
        n = len(C)
        cnt.append(dict(case=case, n_retained=n, p=p, one_over_p=1 / p, n_decoy_rank1=int(C.R1_drop.sum()), pct_decoy_rank1=100 * C.R1_drop.mean(),
                        n_decoy_above_1_over_p=int(C.R2_drop.sum()), pct_decoy_above_1_over_p=100 * C.R2_drop.mean(),
                        decoy_share_median=float(np.median(ds)), decoy_share_p05=float(np.percentile(ds, 5)),
                        decoy_share_p95=float(np.percentile(ds, 95)), decoy_share_max=float(ds.max()),
                        median_over_1_over_p=float(np.median(ds) * p)))
        K = C[~C.R1_drop.astype(bool)]
        for w in WINS[case]:
            for fr, D in (("A (not rank 1 applied)", K), ("A0 (decoy runs, no criterion)", C)):
                for v, r in shares_rank1(D, case, w, rng).items():
                    head.append(dict(case=case, window=w, framing=fr, var=v, n_cfg=r["n"], share=r["share"], lo=r["lo"], hi=r["hi"],
                                     rank1=r["rank1"], r1_lo=r["r1_lo"], r1_hi=r["r1_hi"]))
            t = T2[(T2.case == case) & (T2.set == "final") & (T2.cfgset == "own") & (T2.window == f"w{w}")]
            for _, r in t.iterrows():
                v = B_NAMES[case].get(r["var"], r["var"])
                head.append(dict(case=case, window=w, framing="B (night-6 final-set runs)", var=v, n_cfg=int(r["n_cfg"]), share=r["share"], lo=r["lo"],
                                 hi=r["hi"], rank1=r["frac_rank1"], r1_lo=np.nan, r1_hi=np.nan))
    Cn, H = pd.DataFrame(cnt), pd.DataFrame(head)
    Cn.to_csv(A9 / "N5_decoy_rank1_counts.csv", index=False); H.to_csv(A9 / "N5_headline_table.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format, "display.max_rows", 500):
        (A9 / "N5_decoy_rank1_counts.md").write_text(
            "# N5 item 2: typical-field decoy in the harmonized-retained configs of the decoy runs\n"
            "rank 1 = decoy share (all leads, step 0 excluded, pooled over members) > every real variable's share of the total; "
            "above 1/p = equal-share rule.\n```\n" + Cn.to_string(index=False) + "\n```\n")
        parts = ["# N5 item 3: final-set headline numbers (share [95 % CI over configs]; rank-1 fraction), framings A, A0 and B",
                 "A: decoy runs, 'not rank 1' applied; A0: same runs, no criterion; B: night-6 final-set runs (rank-1 without CI)."]
        for case in ("pnw", "pinatubo", "debby"):
            h = H[H.case == case]
            if h.empty:
                parts.append(f"\n## {TITLE[case]}: waits for N1"); continue
            for w in WINS[case]:
                parts.append(f"\n## {TITLE[case]}, {w} d")
                rows = []
                for v in FINAL[case]:
                    row = {"var": v}
                    for fr in ("A (not rank 1 applied)", "A0 (decoy runs, no criterion)", "B (night-6 final-set runs)"):
                        x = h[(h.window == w) & (h.framing == fr) & (h["var"] == v)]
                        if len(x):
                            x = x.iloc[0]
                            r1 = f"{x.rank1:.2f}" + (f" [{x.r1_lo:.2f},{x.r1_hi:.2f}]" if np.isfinite(x.r1_lo) else "")
                            row[fr[:2].strip()] = f"{x.share:.3f} [{x.lo:.3f},{x.hi:.3f}] / {r1} (n={int(x.n_cfg)})"
                    rows.append(row)
                parts.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```")
        (A9 / "N5_headline_table.md").write_text("\n".join(parts))
    print(Cn.to_string(index=False))


if __name__ == "__main__":
    main()
