"""Night-7 T4: the linkless-system null (approach A) -> one band u(t) per case.

Runs: night7/t2/<case>/<variant>/dNN/ (variant A = linkless system with the lag-0 correlation R of the final set,
indep = R identity; every candidate variable is a surrogate; 8 representative configs x 20 draws per variant);
real final-set runs = night-6 K00 runs night6/t3/<case>/K00 (same 8 configs). Run tables, retention (harmonized)
and per-lead shares as night 6 (N6_T3T4.run_densities / lead_shares; step 0 excluded).

  --extract : per case and variant, per-lead shares of every (draw, config, variable) -> night7/analysis/N7_lead_<case>_<variant>.csv.gz
              (variant "real" = K00); needs the density files, so this is the slow step.
  (default) : bands from the caches.
Pre-declared rule (night-7 brief): null distribution of the CONFIG-MEAN share = bootstrap (2,000 resamples, seed 0):
for each config pick one draw at random, average over the configs; per lead day t and variable; pool over the case's
variables; u(t) = 95th percentile of the pooled values, smoothed by u(t) = u_inf (1 - exp(-t / tau)) (u(0) = 0); if
R^2 < 0.9 the monotone envelope through the origin (running maximum of the raw u) is used and recorded. Half-draw
stability: u from draws below / above the median draw, relative difference at the last lead day (criterion 20 %).
Groups (supplementary): connected sets of variables with |r| >= 0.7 in R (night7/T1_corr_<case>.csv); real = sum of the
members' shares, null = sum of the members' null shares, same p95 rule (own series per group).
A and indep are restricted to (config, draw) pairs present in both variants (--no-common to switch off).

python N7_bands.py [--cases ...] [--extract] [--nproc 16] [--partial] [--test-n6]   (PBS only)
  --test-n6: development run, the night-6 exp-1 surrogate runs t3/<case>/<v>_d<dd> (one surrogate each) as variant A.
Outputs: night7/analysis/N7_bandfull_<case>_<variant>.csv (series, lead_day, u_raw, u_fit, u), N7_band_<case>_A.csv (pooled
only: lead_day, u; for N6_figs456 --band-csv), N7_band_fit.csv (fit, R^2, half-draw stability, counts),
N7_null_means_<case>_<variant>.csv (per-variable null mean per window; marker only).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N6_T3T4 as T  # noqa: E402

R1 = T.R1
N6, N7 = R1 / "night6", R1 / "night7"
OUT = N7 / "analysis"
FINAL, WINDOWS = T.FINAL, T.WINDOWS
VARIANTS = ["A", "indep"]
N_BOOT, SEED, PCT = 2000, 0, 95
R2_MIN, HALF_TOL = 0.9, 0.20
ALIAS = dict(olr="MTNLWRF", slhf="MSLHF", sshf="MSSHF", t2m="T2M", so2="TMSO201", so4="BURDENSO401", aod="AEROD_v",
             h2so4="TMH2SO401", so2tag03="TMSO203", so4tag03="BURDENSO403", tmso203="TMSO203", z500="Z500",
             z10="Z10", tcwv="TCWV", vorticity="vo", bt="Tb", precip="precip")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def var_index(case, name):
    n = norm(name)
    for i, v in enumerate(FINAL[case]):
        if norm(v) == n or ALIAS.get(n) == v:
            return i
    return None


# ------------------------------------------------------------------ extraction
NULL_RETENTION = "harmonized"
_ids_harmonized = T.table_and_ids


def _ids_any(case, d, partial):
    """Run table, ids with at least one successful member, all ids (null runs, --null-retention any)."""
    t = T._load(case, d, partial)
    if t is None or not len(t):
        return None, [], []
    idc = "run_id" if case == "pnw" else "trial_id"
    ok = (t["success_rate"] > 0) if case == "pnw" else (t["n_members_success"] >= 1)
    return t, sorted(t.loc[ok, idc].astype(int)), sorted(t[idc].astype(int))


def run_dirs(case, variant, test_n6):
    if variant == "real":
        return {-1: N6 / "t3" / case / "K00"}
    if test_n6:   # stand-in: night-6 exp-1 surrogate runs (one surrogate each) as "A"; the last column is dropped
        return {int(d.name.rsplit("_d", 1)[1]) + 100 * FINAL[case].index(d.name.rsplit("_d", 1)[0]): d
                for d in sorted((N6 / "t3" / case).glob("*_d[0-9][0-9]")) if d.name.rsplit("_d", 1)[0] in FINAL[case]}
    return {int(d.name[1:]): d for d in sorted((N7 / "t2" / case / variant).glob("d[0-9][0-9]"))}


def extract(case, variant, partial, nproc, test_n6):
    real = FINAL[case]
    rows, cnt = [], []
    for draw, d in run_dirs(case, variant, test_n6).items():
        nv = len(real) + (1 if test_n6 and variant != "real" else 0)
        if variant != "real" and NULL_RETENTION == "any":
            # linkless systems cannot keep trajectories alive (night-7 finding): harmonized retention (2/3 of members,
            # median length >= 80 % of the steps) removes almost every null run, so the null uses every run that has
            # at least one successful member
            T.table_and_ids = _ids_any
        else:
            T.table_and_ids = _ids_harmonized
        Dd, (na, nr) = T.run_densities(case, d, nv, partial, nproc)
        cnt.append(dict(case=case, variant=variant, draw=draw, n_configs=na, n_retained=nr))
        for cfg, D in Dd.items():
            for L, s in T.lead_shares(D).items():
                for i, v in enumerate(real):
                    rows.append((draw, int(cfg), int(L), v, float(s[i])))
    X = pd.DataFrame(rows, columns=["draw", "cfg", "lead", "var", "share"])
    OUT.mkdir(parents=True, exist_ok=True)
    X.to_csv(OUT / f"N7_lead_{case}_{variant}.csv.gz", index=False)
    pd.DataFrame(cnt, columns=["case", "variant", "draw", "n_configs", "n_retained"]).to_csv(
        OUT / f"N7_counts_{case}_{variant}.csv", index=False)
    print(f"[{case}/{variant}] extracted {len(X)} rows from {len(cnt)} run dirs", flush=True)
    return X


def load(case, variant):
    f = OUT / f"N7_lead_{case}_{variant}.csv.gz"
    return pd.read_csv(f) if f.exists() else None


def to_array(X, case, cfgs=None, draws=None):
    """{(cfg, draw)} -> array [lead (0..Lmax), var]; missing = nan. Returns cfg list, draw list, N[cfg, draw, lead, var]."""
    real = FINAL[case]
    cfgs = sorted(X["cfg"].unique()) if cfgs is None else cfgs
    draws = sorted(X["draw"].unique()) if draws is None else draws
    Lmax = int(X["lead"].max())
    N = np.full((len(cfgs), len(draws), Lmax + 1, len(real)), np.nan)
    ci, di = {c: i for i, c in enumerate(cfgs)}, {d: i for i, d in enumerate(draws)}
    vi = {v: i for i, v in enumerate(real)}
    for (c, d, L, v, s) in X[["cfg", "draw", "lead", "var", "share"]].itertuples(index=False):
        if c in ci and d in di:
            N[ci[c], di[d], L, vi[v]] = s
    has = ~np.all(np.isnan(N[:, :, 1:, :]), axis=(2, 3))   # step 0 excluded; u(0) = 0 needs lead 0 = 0 where a run exists
    N[:, :, 0, :] = np.where(has[:, :, None], 0.0, np.nan)
    return cfgs, draws, N


# ------------------------------------------------------------------ band
def bootstrap_config_mean(N, nb=N_BOOT, seed=SEED):
    """N [cfg, draw, lead, series] -> B [nb, lead, series]: per config one random draw (among those with data), mean over configs."""
    rng = np.random.default_rng(seed)
    acc, n = np.zeros((nb,) + N.shape[2:]), 0
    for i in range(N.shape[0]):
        ok = np.where(~np.all(np.isnan(N[i, :, 1:, :]), axis=(1, 2)))[0]
        if not len(ok):
            continue
        acc += N[i][ok[rng.integers(0, len(ok), size=nb)]]
        n += 1
    return acc / max(n, 1), n


def fit_band(u_raw):
    """u_raw[lead] (lead 0 = 0). Returns u_fit, u_used, dict(fit params, R2, used)."""
    from scipy.optimize import curve_fit
    t = np.arange(len(u_raw), dtype=float)
    m = (t >= 1) & np.isfinite(u_raw)
    info = dict(u_inf=np.nan, tau=np.nan, r2=np.nan, used="envelope")
    fit = np.full(len(t), np.nan)
    try:
        f = lambda x, ui, tau: ui * (1.0 - np.exp(-x / tau))
        p, _ = curve_fit(f, t[m], u_raw[m], p0=[float(np.nanmax(u_raw)), 5.0], bounds=([0, 1e-3], [np.inf, 1e3]), maxfev=20000)
        fit = f(t, *p)
        ss_res, ss_tot = np.sum((u_raw[m] - fit[m]) ** 2), np.sum((u_raw[m] - u_raw[m].mean()) ** 2)
        info.update(u_inf=float(p[0]), tau=float(p[1]), r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan)
    except Exception as e:   # noqa: BLE001
        print("fit failed:", e)
    env = np.maximum.accumulate(np.nan_to_num(u_raw, nan=0.0)); env[0] = 0.0
    if np.isfinite(info["r2"]) and info["r2"] >= R2_MIN:
        info["used"] = "fit"
        return fit, fit, info
    return fit, env, info


def series_band(N, label, nb=N_BOOT):
    """Pooled p95 of the config-mean null for the series axis of N [cfg, draw, lead, series]."""
    B, ncfg = bootstrap_config_mean(N, nb)
    u_raw = np.nanpercentile(B[:, :, :].transpose(1, 0, 2).reshape(B.shape[1], -1), PCT, axis=1)
    u_raw[0] = 0.0
    fit, used, info = fit_band(u_raw)
    info.update(series=label, n_cfg=ncfg, n_draws=N.shape[1])
    return u_raw, fit, used, info


def read_groups(case):
    """Connected components (size >= 2) of |r| >= 0.7 from night7/T1_corr_<case>.csv -> {name: [var indices]}."""
    f = N7 / f"T1_corr_{case}.csv"
    if not f.exists():
        return {}
    R = pd.read_csv(f, index_col=0)
    idx = [var_index(case, c) for c in R.columns]
    if any(i is None for i in idx):
        print(f"[{case}] T1 names not matched: {[c for c, i in zip(R.columns, idx) if i is None]}; groups skipped")
        return {}
    M = np.abs(R.to_numpy(float)) >= 0.7
    seen, groups = set(), {}
    for s in range(len(idx)):
        if s in seen:
            continue
        comp, stack = [], [s]
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a); comp.append(a)
            stack += [b for b in range(len(idx)) if M[a, b] and b not in seen]
        if len(comp) >= 2:
            groups["+".join(FINAL[case][idx[a]] for a in sorted(comp))] = [idx[a] for a in sorted(comp)]
    return groups


def window_means(N, case):
    """Per variable and window: mean null share over (cfg, draw, lead days)."""
    rows = []
    for (w0, w1) in WINDOWS[case]:
        sl = N[:, :, w0: w1 + 1, :]
        for i, v in enumerate(FINAL[case]):
            rows.append(dict(case=case, var=v, window=f"w{w0}-{w1}", null_mean=float(np.nanmean(sl[..., i]))))
    return rows


def run_case(case, test_n6=False):
    Xs = {v: load(case, v) for v in VARIANTS}
    Xs = {v: x for v, x in Xs.items() if x is not None and len(x)}
    if not Xs:
        print(f"[{case}] no cached null shares"); return
    # common (config, draw) pairs over the variants present
    pairs = None
    if len(Xs) > 1 and not ARGS.no_common:
        for x in Xs.values():
            p = set(map(tuple, x[["cfg", "draw"]].drop_duplicates().to_numpy()))
            pairs = p if pairs is None else pairs & p
    fits, groups = [], read_groups(case)
    for variant, X in Xs.items():
        if pairs is not None:
            X = X[[(c, d) in pairs for c, d in zip(X["cfg"], X["draw"])]]
        cfgs, draws, N = to_array(X, case)
        n_pairs = int((~np.all(np.isnan(N[:, :, 1:, :]), axis=(2, 3))).sum())
        bands = []
        series = [("pooled", N)] + [(g, N[..., ix].sum(-1, keepdims=True)) for g, ix in groups.items()]
        for label, Na in series:
            u_raw, fit, used, info = series_band(Na, label)
            # half-draw stability (draw index below / above the median draw)
            half = len(draws) // 2
            if label == "pooled" and half >= 1:
                ua = series_band(Na[:, :half], "a")[2][-1]; ub = series_band(Na[:, half:], "b")[2][-1]
                info.update(u_half1=float(ua), u_half2=float(ub), half_rel_diff=float(abs(ua - ub) / ub) if ub > 0 else np.nan)
            info.update(case=case, variant=variant, n_pairs=n_pairs, n_cfg_total=len(cfgs),
                        monotone=bool(np.all(np.diff(used) >= -1e-12)), u_last=float(used[-1]))
            fits.append(info)
            for t in range(len(u_raw)):
                bands.append(dict(case=case, variant=variant, series=label, lead_day=t, u_raw=u_raw[t], u_fit=fit[t], u=used[t]))
        B = pd.DataFrame(bands)
        B.to_csv(OUT / f"N7_bandfull_{case}_{variant}.csv", index=False)
        if variant == "A":
            B[B["series"] == "pooled"][["lead_day", "u"]].to_csv(OUT / f"N7_band_{case}_A.csv", index=False)
        pd.DataFrame(window_means(N, case)).to_csv(OUT / f"N7_null_means_{case}_{variant}.csv", index=False)
        print(f"[{case}/{variant}] pairs {n_pairs}, configs {len(cfgs)}, draws {len(draws)}", flush=True)
    return fits


def fig_bands(cases):
    """S20 supplementary: u(t) of A and indep per case (pooled series), plus the raw p95."""
    import matplotlib.pyplot as plt
    import N5_style as S
    S.set_style()
    fig, ax = plt.subplots(1, len(cases), figsize=(S.DOUBLE, 2.4), squeeze=False)
    for j, case in enumerate(cases):
        for variant, c in (("A", "#0072B2"), ("indep", "#D55E00")):
            f = OUT / f"N7_bandfull_{case}_{variant}.csv"
            if not f.exists():
                continue
            b = pd.read_csv(f); b = b[b["series"] == "pooled"]
            ax[0, j].plot(b["lead_day"], b["u"], color=c, lw=1.2, label=f"{variant} (used)")
            ax[0, j].plot(b["lead_day"], b["u_raw"], color=c, lw=0.6, ls=":", label=f"{variant} raw p95")
        ax[0, j].set_title(T.CASE_TITLE[case], fontsize=7); ax[0, j].set_xlabel("Lead before the event (days)")
        ax[0, j].set_ylim(0, None); ax[0, j].grid(alpha=0.2, lw=0.4)
    ax[0, 0].set_ylabel("u(t): 95th pct of config-mean null share"); ax[0, 0].legend(fontsize=5.5)
    S.panel_labels(ax.ravel()); fig.tight_layout()
    d = R1 / "figures_v7" / ("test" if OUT.name == "analysis_test" else "") / "S20_bands"; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "s20_bands_A_vs_indep.pdf", bbox_inches="tight"); fig.savefig(d / "s20_bands_A_vs_indep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


ARGS = None


def main():
    global ARGS, NULL_RETENTION
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["pnw", "pinatubo", "debby"])
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--no-common", action="store_true")
    ap.add_argument("--test-n6", action="store_true")
    ap.add_argument("--null-retention", choices=["harmonized", "any"], default="harmonized",
                    help="retention of the null runs (A, indep): harmonized (night-6 rule) or any (>= 1 successful member)")
    ap.add_argument("--variants", nargs="+", default=None, help="extract only these variants (real, A, indep)")
    ARGS = a = ap.parse_args()
    global OUT
    NULL_RETENTION = a.null_retention
    if a.test_n6:
        OUT = N7 / "analysis_test"
    OUT.mkdir(parents=True, exist_ok=True)
    allfits = []
    for case in a.cases:
        if a.extract:
            for v in (a.variants or (["real"] + (["A"] if a.test_n6 else VARIANTS))):
                extract(case, v, True if a.partial or a.test_n6 else False, a.nproc, a.test_n6)
        else:
            f = run_case(case, a.test_n6)
            allfits += f or []
    if not a.extract:
        fig_bands(a.cases)
    if allfits:
        F = pd.DataFrame(allfits)
        F.to_csv(OUT / "N7_band_fit.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(F[["case", "variant", "series", "used", "u_inf", "tau", "r2", "monotone", "u_last", "half_rel_diff", "n_pairs"]].round(3).to_string())


if __name__ == "__main__":
    main()
