"""Night-7 T4: above-null tables and the criteria table (approach A, one rule for the three cases).

Rule (pre-declared): a real variable is above the null in a window if the lower bound of the 95 % bootstrap CI (over
the configs, 10,000 resamples, seed 0) of the mean over the window's lead days of (real config-mean share - u(t)) is
> 0. Real shares = the final-set runs (night-6 K00, same 8 configs; harmonized retention); u(t) from N7_bands.py
(night7/analysis/N7_bandfull_<case>_<variant>.csv, series "pooled" for single variables, one series per group for the
group reading: real = sum of the members' shares). Windows as night 6 (Debby 1-4, 5-15 (5-14 in the daily
aggregation); Pinatubo 1-3, 4-16; PNW 1-10, 11-20, 21-30, 11-30 d), step 0 excluded.
Also: each variable's own null mean in the window (marker, not used for decisions), A vs indep status changes, the
criteria table (brief: validity, band quality, negative controls, known parents) and the PNW 11-30 d statement
(OLR, SLHF, SSHF, SLHF+SSHF).

python N7_above_null.py [--cases ...] [--test-n6]   (PBS only; after N7_bands.py --extract and N7_bands.py)
Outputs: night7/analysis/N7_above_null_<case>.csv, N7_above_null.md, N7_status_A_vs_indep.md, N7_criteria.md.
Optional input for criterion 3: night7/analysis/T3b_status.csv (columns var, kind = parent|decoy, above = 0/1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N7_bands as B  # noqa: E402
from A4_twins import boot_ci  # noqa: E402

FINAL, WINDOWS, N7 = B.FINAL, B.WINDOWS, B.N7
PARENT_WINDOWS = dict(debby=[("vo", "w1-4"), ("vo", "w5-15")], pinatubo=[("TMSO201", "w1-3"), ("TMSO201", "w4-16")],
                      pnw=[("Z500", "w1-10"), ("TCWV", "w1-10")])
NEG_CONTROLS = dict(pinatubo=["TMSO203", "BURDENSO403"])
SHOW = dict(MTNLWRF="OLR", MSLHF="SLHF", MSSHF="SSHF", T2M="T2m")


def disp(v):
    return "+".join(SHOW.get(p, p) for p in v.split("+"))


def real_array(case):
    X = B.load(case, "real")
    if X is None or not len(X):
        return None
    return B.to_array(X, case)   # cfgs, draws(-1), N[cfg, 1, lead, var]


def above_rows(case, variant, band):
    cfgs, _, N = real_array(case)
    R = N[:, 0]   # [cfg, lead, var]
    groups = B.read_groups(case)
    rows = []
    items = [(v, R[..., i], "pooled", v) for i, v in enumerate(FINAL[case])]
    items += [(g, R[..., ix].sum(-1), g, g) for g, ix in groups.items()]
    nm = pd.read_csv(B.OUT / f"N7_null_means_{case}_{variant}.csv") if (B.OUT / f"N7_null_means_{case}_{variant}.csv").exists() else None
    for name, S, series, _ in items:
        b = band[band["series"] == series].sort_values("lead_day")
        if not len(b):
            continue
        u = b["u"].to_numpy()
        for (w0, w1) in WINDOWS[case]:
            sl = slice(w0, min(w1, S.shape[1] - 1, len(u) - 1) + 1)
            x = np.nanmean(S[:, sl] - u[sl][None, :], axis=1)
            m, lo, hi = boot_ci(x)
            r = dict(case=case, variant=variant, var=name, is_group=series != "pooled", window=f"w{w0}-{w1}", n_cfg=int(np.isfinite(x).sum()),
                     share=float(np.nanmean(S[:, sl])), u_mean=float(u[sl].mean()), excess=m, lo=lo, hi=hi, above=bool(lo > 0))
            if nm is not None and series == "pooled":
                q = nm[(nm["var"] == name) & (nm["window"] == r["window"])]
                r["own_null_mean"] = float(q["null_mean"].iloc[0]) if len(q) else np.nan
            rows.append(r)
    return rows


def md(df, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False):
        out.append("| " + " | ".join(f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def validity(case, variant):
    f = N7 / "surrogates" / f"validation_{case}_{variant}.csv"
    if not f.exists():
        return "no file"
    d = pd.read_csv(f)
    c = [k for k in d.columns if "pass" in k.lower() or k.lower() in ("ok", "within")]
    if not c:
        return f"see {f.name}"
    ok = d[c].astype(str).apply(lambda s: s.str.lower().isin(["true", "1", "pass", "yes", "ok"])).all().all()
    return "pass" if ok else "FAIL"


def criteria(T, fits, cases):
    L = ["| Criterion | Case | Result | Detail |", "|---|---|---|---|"]
    for case in cases:
        L.append(f"| 1 generator validity (A, indep) | {case} | {validity(case, 'A')} / {validity(case, 'indep')} | night7/surrogates/validation_{case}_*.md |")
        f = fits[(fits["case"] == case) & (fits["variant"] == "A") & (fits["series"] == "pooled")]
        if len(f):
            f = f.iloc[0]
            ok = bool(f["used"] == "fit" and f["monotone"] and (f.get("half_rel_diff", np.nan) <= B.HALF_TOL))
            L.append(f"| 2 band (smooth, monotone, R2 >= 0.9, halves within 20 %) | {case} | {'pass' if ok else 'FAIL'} | "
                     f"used {f['used']}, R2 {f['r2']:.3f}, u_inf {f['u_inf']:.3f}, tau {f['tau']:.2f}, monotone {f['monotone']}, "
                     f"half-draw rel. diff {f.get('half_rel_diff', np.nan):.3f}, n pairs {int(f['n_pairs'])} |")
        a = T[(T["case"] == case) & (T["variant"] == "A") & (~T["is_group"])]
        for v in NEG_CONTROLS.get(case, []):
            q = a[a["var"] == v]
            if len(q):
                L.append(f"| 3 negative control not above in any window | {case} {v} | {'pass' if not q['above'].any() else 'FAIL'} | "
                         + ", ".join(f"{r.window[1:]} d {r.excess:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" for r in q.itertuples()) + " |")
        for v, w in PARENT_WINDOWS[case]:
            q = a[(a["var"] == v) & (a["window"] == w)]
            if len(q):
                r = q.iloc[0]
                L.append(f"| 4 known parent above | {case} {disp(v)} {w[1:]} d | {'pass' if r['above'] else 'FAIL'} | "
                         f"excess {r['excess']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] |")
    f = B.OUT / "T3b_status.csv"
    if f.exists():
        s = pd.read_csv(f)
        bad = s[((s["kind"] == "decoy") & (s["above"] == 1)) | ((s["kind"] == "parent") & (s["above"] == 0))]
        L.append(f"| 3 synthetic decoys not above / parents above (T3b) | synthetic | {'pass' if not len(bad) else 'FAIL'} | "
                 f"{len(s)} variables; failing: {', '.join(bad['var'].astype(str)) or 'none'} |")
    else:
        L.append("| 3 synthetic decoys / parents (T3b) | synthetic | pending | T3b_status.csv not yet written |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["pnw", "pinatubo", "debby"])
    ap.add_argument("--test-n6", action="store_true")
    a = ap.parse_args()
    if a.test_n6:
        B.OUT = N7 / "analysis_test"
    rows = []
    for case in a.cases:
        if real_array(case) is None:
            print(f"[{case}] no real-run cache (N7_bands.py --extract)"); continue
        for variant in B.VARIANTS:
            f = B.OUT / f"N7_bandfull_{case}_{variant}.csv"
            if f.exists():
                rows += above_rows(case, variant, pd.read_csv(f))
    if not rows:
        print("nothing to do"); return
    T = pd.DataFrame(rows)
    for case, g in T.groupby("case"):
        g.to_csv(B.OUT / f"N7_above_null_{case}.csv", index=False)
    lines = ["# Night 7: above the linkless-system null (rule: lower 95 % bound of mean over window of share - u(t) > 0)\n"]
    for (case, variant, grp), g in T.groupby(["case", "variant", "is_group"], sort=False):
        g = g.assign(var=g["var"].map(disp), window=g["window"].str[1:] + " d", ci=[f"{e:+.3f} [{l:+.3f}, {h:+.3f}]" for e, l, h in zip(g["excess"], g["lo"], g["hi"])],
                     above=np.where(g["above"], "yes", "no"))
        cols = ["var", "window", "share", "u_mean", "ci", "above"] + (["own_null_mean"] if "own_null_mean" in g and not grp else [])
        lines += [f"\n## {case}, variant {variant}, {'groups' if grp else 'single variables'}\n", md(g, cols)]
    (B.OUT / "N7_above_null.md").write_text("\n".join(lines))
    # A vs indep status changes
    P = T[~T["is_group"]].pivot_table(index=["case", "var", "window"], columns="variant", values="above", aggfunc="first")
    ch = ["# A vs indep: status changes (single variables)\n", "| case | variable | window | A | indep |", "|---|---|---|---|---|"]
    if {"A", "indep"} <= set(P.columns):
        for (c, v, w), r in P.iterrows():
            if r["A"] != r["indep"]:
                ch.append(f"| {c} | {disp(v)} | {w[1:]} d | {'above' if r['A'] else 'not above'} | {'above' if r['indep'] else 'not above'} |")
        if len(ch) == 3:
            ch.append("| (no change) | | | | |")
    (B.OUT / "N7_status_A_vs_indep.md").write_text("\n".join(ch) + "\n")
    # PNW 11-30 d statement
    pn = T[(T["case"] == "pnw") & (T["window"] == "w11-30") & (T["variant"] == "A")]
    if len(pn):
        s = ["\n# PNW 11-30 d (A)\n"] + [f"- {disp(r.var)}: excess {r.excess:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] -> {'above' if r.above else 'not above'}" for r in pn.itertuples()]
        (B.OUT / "N7_pnw_11_30.md").write_text("\n".join(s) + "\n")
    fp = B.OUT / "N7_band_fit.csv"
    if fp.exists():
        (B.OUT / "N7_criteria.md").write_text(criteria(T, pd.read_csv(fp), [c for c in a.cases if c in set(T["case"])]))
    print((B.OUT / "N7_above_null.md").read_text()[:6000])


if __name__ == "__main__":
    main()
