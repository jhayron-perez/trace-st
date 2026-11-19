"""Night-7 T3b: synthetic arbiter of the linkless-system null (approach A), p = 8 composition.

Real runs: night5/t3/p08 (T2-EN, M 20, linear rule, 20 of the 50 paper configs = trial ids 0-19, alpha {0.2, 0.6} ->
40 (trial, alpha) "configs"). Null runs: night7/t3b/A/dNN/ (same arguments plus --linkless A --linkless-draw NN, 20 draws).
Variables: A, B (child), C, N1 (noise), R1_B (response), M1 (moving), L1_A (delayed), Q1_A (nonlinear).
Share of variable v at lead step k = fraction of the successful members whose step-k parent is v (bundle `parents`,
step 0 excluded). Retention: every run with >= 1 successful member for both the null and the real runs (a linkless null
loses almost all runs under the harmonized rule, night-7 finding); the strict counts (>= 2/3 of the members successful and
median length >= 80 % of the steps) are reported too. Band rule as N7_bands.py (config-mean bootstrap, pooled p95,
smoothed fit), above-null rule as N7_above_null.py (95 % bootstrap CI over configs of the window mean of
real config-mean share - u(t) > 0). Windows 1-10 and 11-30 d (the 30 steps of the synthetic run).

python N7_T3b.py   (PBS only) -> night7/analysis/T3b_status.csv, T3b_status.md, T3b_band.csv, T3b_counts.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N7_bands as B  # noqa: E402

R1 = B.R1
N5, N7 = R1 / "night5", R1 / "night7"
OUT = N7 / "analysis"
VARS = ["A", "B", "C", "N1", "R1_B", "M1", "L1_A", "Q1_A"]
KIND = {"A": "parent", "C": "parent", "B": "child", "N1": "decoy", "R1_B": "decoy", "M1": "decoy", "L1_A": "decoy",
        "Q1_A": "decoy"}
ALPHAS = ["0p2", "0p6"]
TRIALS = range(20)
LMAX = 30
WINDOWS = {"1-10 d": (1, 10), "11-30 d": (11, 30)}


def run_shares(d, trial, a):
    """(lead shares [LMAX+1, 8] with row 0 = 0, n_ok, strict flag) of one (trial, alpha) bundle, or None."""
    stem = d / "bundles" / f"trial_{trial:05d}_alpha_{a}"
    if not stem.with_suffix(".npz").exists():
        return None
    z = np.load(stem.with_suffix(".npz"))
    meta = json.load(open(stem.with_suffix(".json")))
    ok = z["ok"].astype(bool)
    sm = meta["summary_small"]["validity_metrics"]
    n_ok = int(ok.sum())
    strict = bool(n_ok >= (2 / 3) * len(ok) and float(sm.get("median_n_steps", 0)) >= 0.8 * 31)
    P = z["parents"][ok]
    S = np.full((LMAX + 1, len(VARS)), np.nan)
    if n_ok >= 1:
        for k in range(1, LMAX + 1):
            col = P[:, k] if k < P.shape[1] else np.full(len(P), -1)
            n = int((col > 0).sum())
            if n:
                S[k] = [(col == v + 1).sum() / n for v in range(len(VARS))]
        if np.isfinite(S[1:]).any():
            S[0] = 0.0
    return S, n_ok, strict


def collect(dirs):
    """dirs {draw: path} -> array [cfg(40), draw, lead, var], counts rows."""
    draws = sorted(dirs)
    N = np.full((40, len(draws), LMAX + 1, len(VARS)), np.nan)
    cnt = []
    for j, dr in enumerate(draws):
        na = nr = ns = 0
        for ti in TRIALS:
            for ai, a in enumerate(ALPHAS):
                r = run_shares(dirs[dr], ti, a)
                if r is None:
                    continue
                na += 1
                S, n_ok, strict = r
                ns += int(strict)
                if n_ok >= 1:
                    nr += 1
                    N[ti * 2 + ai, j] = S
        cnt.append(dict(draw=dr, n_rows=na, n_any=nr, n_strict=ns))
    return N, draws, pd.DataFrame(cnt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nulldirs = {int(p.name[1:]): p for p in sorted((N7 / "t3b" / "A").glob("d[0-9][0-9]"))}
    Nn, draws, cn = collect(nulldirs)
    Nr, _, cr = collect({-1: N5 / "t3" / "p08"})
    cn.assign(kind="null_A").to_csv(OUT / "T3b_counts.csv", index=False)
    cr.assign(kind="real").to_csv(OUT / "T3b_counts.csv", mode="a", header=False, index=False)
    u_raw, fit, used, info = B.series_band(Nn, "pooled")
    h = len(draws) // 2
    half = []
    for sl in (slice(0, h), slice(h, None)):
        half.append(B.series_band(Nn[:, sl], "half")[2][-1])
    info["half_rel_diff_last"] = float(abs(half[0] - half[1]) / max(np.mean(half), 1e-12)) if h >= 2 else np.nan
    pd.DataFrame(dict(lead_day=np.arange(LMAX + 1), u_raw=u_raw, u_fit=fit, u=used)).to_csv(OUT / "T3b_band.csv", index=False)
    # null mean per variable (marker only)
    nullmean = np.nanmean(Nn[:, :, 1:, :], axis=(0, 1, 2))
    # real config-mean share per lead
    R = Nr[:, 0]                                    # [cfg, lead, var]
    rng = np.random.default_rng(0)
    rows = []
    cfg_ok = ~np.all(np.isnan(R[:, 1:, :]), axis=(1, 2))
    Rk = R[cfg_ok]
    for vi, v in enumerate(VARS):
        rec = dict(var=v, kind=KIND[v], null_mean=float(nullmean[vi]))
        flags = []
        for w, (a, b) in WINDOWS.items():
            x = Rk[:, a:b + 1, vi] - used[None, a:b + 1]           # [cfg, lead]
            wm = np.nanmean(x, axis=1)                              # window mean per config
            wm = wm[np.isfinite(wm)]
            idx = rng.integers(0, len(wm), size=(10000, len(wm)))
            bm = wm[idx].mean(axis=1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            rec[f"real_{w}"] = float(np.nanmean(Rk[:, a:b + 1, vi]))
            rec[f"excess_{w}"], rec[f"lo_{w}"], rec[f"hi_{w}"] = float(wm.mean()), float(lo), float(hi)
            rec[f"above_{w}"] = bool(lo > 0)
            flags.append(bool(lo > 0))
        rec["above"] = bool(any(flags))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "T3b_status_full.csv", index=False)
    df[["var", "kind", "above"]].to_csv(OUT / "T3b_status.csv", index=False)
    par_ok = bool(df.loc[df.kind == "parent", "above"].all())
    dec_ok = bool((~df.loc[df.kind == "decoy", "above"]).all())
    lines = ["# T3b synthetic arbiter (p = 8, linkless system A)", "",
             f"Null runs: {len(draws)} draws; rows (trial, alpha) with >= 1 successful member per draw: median "
             f"{int(cn.n_any.median())} of {int(cn.n_rows.median())} present; strict-retained median {int(cn.n_strict.median())}. "
             f"Real runs (night-5 p08, trials 0-19 x alpha): {int(cr.n_any.iloc[0])} of {int(cr.n_rows.iloc[0])} with >= 1 successful member, "
             f"strict {int(cr.n_strict.iloc[0])}.", "",
             f"Band u(t): {info['used']} (u_inf {info['u_inf']:.3f}, tau {info['tau']:.2f}, R2 {info['r2']:.3f}); "
             f"u(30 d) = {used[-1]:.3f}; half-draw relative difference at the last lead day {info['half_rel_diff_last']:.3f}.", "",
             "| var | kind | null mean | real 1-10 d | excess [95 % CI] | above | real 11-30 d | excess [95 % CI] | above |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['var']} | {r['kind']} | {r['null_mean']:.3f} | {r['real_1-10 d']:.3f} | "
                     f"{r['excess_1-10 d']:+.3f} [{r['lo_1-10 d']:+.3f}, {r['hi_1-10 d']:+.3f}] | {'yes' if r['above_1-10 d'] else 'no'} | "
                     f"{r['real_11-30 d']:.3f} | {r['excess_11-30 d']:+.3f} [{r['lo_11-30 d']:+.3f}, {r['hi_11-30 d']:+.3f}] | "
                     f"{'yes' if r['above_11-30 d'] else 'no'} |")
    lines += ["", f"Criterion 3 (synthetic): true parents A, C above the band in some window: **{'PASS' if par_ok else 'FAIL'}**; "
              f"decoys (N1, R1_B, M1, L1_A, Q1_A) not above in any window: **{'PASS' if dec_ok else 'FAIL'}**."]
    (OUT / "T3b_status.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
