"""Night-8 T1c: one common range for lambda_a / l1_ratio in every case (pre-declared in the report, 17:25 MST).

Per case a per-configuration table (night8/analysis/T1c_<case>_configs.csv): id, drawn lambda_a / l1_ratio, harmonized retention,
decoy share (all leads, step 0 excluded, pooled over members / rows), rule flags R1 (decoy share > every real variable's
share of the total) and R2 (decoy share > 1/p), headline-window shares (renormalized among the real variables).
Checks: (1) survival = fraction of the drawn configs retained AND kept by the rule; (2) kept vs removed lambda_a / l1_ratio
(quantiles, fraction inside the paper range of the case); (3) headline shares and rank-1 fractions with 95 % bootstrap CIs
(2,000 resamples over configs, seed 0) against the case-specific ensembles (PNW/Pinatubo: T1b configs; Debby: night6/t2/debby_final;
synthetic: night5/t3/p03).
python N8_T1c.py --case pnw|pinatubo|debby|synthetic --nproc 32
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N8_T1b as B  # noqa: E402
import N6_T3T4 as T  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = B.R1
T1C = R1 / "night8" / "t1c"
OUT = R1 / "night8" / "analysis"
REAL = dict(pnw=B.FINAL["pnw"], pinatubo=B.FINAL["pinatubo"], debby=["Tb", "vo", "precip", "CAPE", "DIV200", "MSLP"],
            synthetic=["A", "B", "C"])
WINDOWS = dict(pnw=B.WINDOWS["pnw"], pinatubo=B.WINDOWS["pinatubo"], debby=[(1, 4), (5, 15)], synthetic=[(0, 0)])
PAPER = dict(pnw=dict(lam=(-2.5, 0.0), l1=(-1.0, 0.0)), pinatubo=dict(lam=(-0.3, 0.0), l1=(-0.2, 0.0)),
             debby=dict(lam=(-2.5, -1.0), l1=(-1.0, 0.0)), synthetic=dict(lam=(-2.5, -1.0), l1=(-1.0, 0.0)))
DRAWN = dict(pnw=300, pinatubo=300, debby=60, synthetic=50)


def first(df, *subs):
    for c in df.columns:
        if all(s in c for s in subs):
            return c
    return None


def window_shares(Dp, nv, wins):
    out = {}
    for (w0, w1) in wins:
        _, s = B.shares(Dp, nv, (w0, w1))
        out[(w0, w1)] = s
    return out


def table_from_D(case, D, hp, ids_all, ids_ret, decoy=True):
    """D {id: (lead, nv)} pooled over members -> per-config frame (retained configs only) + all ids' hp.
    decoy=False: reference runs without a decoy column (shares of the real variables only)."""
    real = REAL[case]; nv = len(real) + 1
    rows = []
    for i in ids_all:
        row = dict(id=i, retained=bool(i in ids_ret), lam=hp.get(i, (np.nan, np.nan))[0], l1=hp.get(i, (np.nan, np.nan))[1])
        if i in ids_ret and i in D:
            Dp = D[i]
            if decoy:
                ds, _ = B.shares(Dp, nv)
                tot = Dp[1:].sum()
                ra = Dp[1:, :nv - 1].sum(0) / tot
                r1, r2 = B.flags(ds, ra, nv)
                row.update(decoy_share=ds, R1_drop=bool(r1), R2_drop=bool(r2))
                wsh = window_shares(Dp, nv, WINDOWS[case])
            else:
                wsh = {}
                for (w0, w1) in WINDOWS[case]:
                    x = Dp[w0:w1 + 1].sum(0)
                    wsh[(w0, w1)] = x / x.sum() if x.sum() > 0 else np.full(len(real), np.nan)
            for (w0, w1), s in wsh.items():
                for j, v in enumerate(real):
                    row[f"s_{w0}-{w1}_{v}"] = s[j]
        rows.append(row)
    return pd.DataFrame(rows).set_index("id")


def hp_from_table(t, idc, lamc=None, l1c=None):
    lamc = lamc or first(t, "lambda_a"); l1c = l1c or first(t, "l1_ratio")
    return {int(r[idc]): (float(r[lamc]), float(r[l1c])) for _, r in t.iterrows()}


def case_pnw(nproc):
    B.T1B = T1C
    data, ids_ret, n_all = B.load("pnw", nproc)
    hp = {int(i): (float(v[2]["lambda_a"]), float(v[2]["l1_ratio"])) for i, v in data.items()}
    nv = len(REAL["pnw"]) + 1
    D = {i: v[0].sum(0) for i, v in data.items()}
    return table_from_D("pnw", D, hp, sorted(data), ids_ret)


def case_pinatubo(nproc):
    nv = len(REAL["pinatubo"]) + 1
    jobs, tabs = [], []
    for d in sorted(T1C.glob("pin_p[0-9]")):
        f = d / "results_jobs_final.csv"
        t = pd.read_csv(f if f.exists() else d / "results_jobs_snapshot.csv"); tabs.append(t)
        j = t[t["bundle_npz"].notna()]
        for r in j.itertuples():
            jobs.append((r.bundle_npz, r.box_size, nv, r.trial_id, {}))
    t = pd.concat(tabs, ignore_index=True)
    keep = retention.harmonized_case("pinatubo", t)
    ids_ret = set(t.loc[keep.to_numpy(), "trial_id"].astype(int))
    with Pool(nproc) as pool:
        out = pool.map(B.member_dens_pin, jobs)
    D = {i: d.sum(0) for i, d, ok, hp in out}
    hp = hp_from_table(t, "trial_id")
    return table_from_D("pinatubo", D, hp, sorted(int(x) for x in t["trial_id"]), ids_ret)


def case_debby(nproc, base=None, parts=None, nv=None, drawn_only=False):
    nv = nv or len(REAL["debby"]) + 1
    D, hp, ids_all, ids_ret = {}, {}, [], set()
    for d in (parts if parts is not None else sorted(T1C.glob("deb_p[0-9]"))):
        t, ids, all_ids = T.table_and_ids("debby", d, True)
        if t is None:
            continue
        hp.update(hp_from_table(t, "trial_id"))
        Dd, _ = T.run_densities("debby", d, nv, True, nproc)
        D.update({int(k): np.asarray(v, float) for k, v in Dd.items()})
        ids_all += all_ids; ids_ret |= set(ids)
    return table_from_D("debby", D, hp, sorted(ids_all), ids_ret, decoy=(nv == len(REAL["debby"]) + 1))


def case_synthetic(d=None, extra_name=None):
    dirs = [d] if d else sorted(T1C.glob("syn_p[0-9]"))
    xs = []
    for dd in dirs:
        for nm in ("results_alphas_final.csv", "results_alphas_snapshot.csv"):
            if (dd / nm).exists():
                xs.append(pd.read_csv(dd / nm)); break
    x = pd.concat(xs, ignore_index=True)
    dens = [c for c in x.columns if c.startswith("t7_density_")]
    names = [c.replace("t7_density_", "") for c in dens]
    D = x[dens].astype(float); D.columns = names
    tot = D.sum(1)
    ok = (tot > 0) & D[["A", "C"]].sum(1).gt(0)
    if "ok_frac" in x:
        ok &= x["ok_frac"] >= 2 / 3
    dec = [n for n in names if n not in ("A", "B", "C")]
    p = len(names)
    rows = pd.DataFrame(dict(id=[f"{int(t)}_{a}" for t, a in zip(x.trial_id, x.alpha_mix)], trial_id=x.trial_id.to_numpy(),
                             alpha=x.alpha_mix.to_numpy(), retained=ok.to_numpy()))
    lam = first(x, "lambda_a"); l1 = first(x, "l1_ratio")
    rows["lam"] = x[lam].to_numpy() if lam else np.nan
    rows["l1"] = x[l1].to_numpy() if l1 else np.nan
    if dec:
        dsh = D[dec].sum(1) / tot
        real_sh = D[["A", "B", "C"]].div(tot, axis=0)
        rows["decoy_share"] = dsh.to_numpy()
        rows["R1_drop"] = (dsh > real_sh.max(1)).to_numpy()
        rows["R2_drop"] = (dsh > 1.0 / p).to_numpy()
    rows["true_parent_share"] = ((D["A"] + D["C"]) / D[["A", "B", "C"]].sum(1)).to_numpy()
    rows["alpha_err"] = ((D["A"] / (D["A"] + D["C"])) - x["alpha_mix"]).abs().to_numpy()
    return rows.set_index("id"), p, dec


def boot(x, rng, n=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan,) * 3
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarize(case, R, ref=None, extra_cols=()):
    """check 1, 2, 3 tables from a per-config frame (and reference frame with the same share columns)."""
    real = REAL[case]
    rng = np.random.default_rng(0)
    N = DRAWN[case] if case != "synthetic" else 2 * DRAWN[case]
    ret = R[R.retained]
    sets = {"retained (unfiltered)": ret}
    if "R1_drop" in R:
        sets["R1 not-rank-1 kept"] = ret[~ret.R1_drop.astype(bool)]
        sets["R2 equal-share kept"] = ret[~ret.R2_drop.astype(bool)]
    c1 = pd.DataFrame([dict(case=case, set=k, n_drawn=N, n=len(v), frac_of_drawn=len(v) / N, passes_20pct=len(v) / N >= 0.2)
                       for k, v in sets.items()])
    pp = PAPER[case]
    rows2 = []
    groups = dict(all_drawn=R, retained=ret, **{k.split()[0] + "_kept": v for k, v in sets.items() if k.startswith("R")})
    if "R1_drop" in R:
        groups["R1_removed"] = ret[ret.R1_drop.astype(bool)]
        groups["R2_removed"] = ret[ret.R2_drop.astype(bool)]
    groups["not_retained"] = R[~R.retained]
    for g, v in groups.items():
        for col, key in (("lam", "lam"), ("l1", "l1")):
            x = np.log10(v[col].dropna().astype(float))
            if len(x) == 0:
                rows2.append(dict(case=case, group=g, param=col, n=0)); continue
            lo, hi = pp[key]
            rows2.append(dict(case=case, group=g, param=col, n=len(x), q10=x.quantile(.1), q25=x.quantile(.25), median=x.median(),
                              q75=x.quantile(.75), q90=x.quantile(.9), frac_inside_paper_range=float(((x >= lo - 1e-9) & (x <= hi + 1e-9)).mean()),
                              paper_range=f"[{lo}, {hi}]"))
    c2 = pd.DataFrame(rows2)
    rows3 = []
    if case != "synthetic":
        for w0, w1 in WINDOWS[case]:
            cols = [f"s_{w0}-{w1}_{v}" for v in real]
            allsets = {"T1c " + k: v for k, v in sets.items()}
            if ref is not None:
                for k, v in ref.items():
                    allsets["case-specific " + k] = v
            for k, v in allsets.items():
                S = v[cols].to_numpy(float)
                S = S[np.isfinite(S).all(1)]
                if len(S) == 0:
                    continue
                top = np.argmax(S, 1)
                for j, name in enumerate(real):
                    m, lo, hi = boot(S[:, j], rng)
                    r1 = boot((top == j).astype(float), rng)
                    rows3.append(dict(case=case, window=f"{w0}-{w1}", set=k, n=len(S), var=name, share=m, lo=lo, hi=hi,
                                      rank1=r1[0], r1_lo=r1[1], r1_hi=r1[2]))
    else:
        for k, v in list(sets.items()) + [("case-specific " + a, b) for a, b in (ref or {}).items()]:
            for al in ("pooled", 0.2, 0.6):
                s = v if al == "pooled" else v[np.isclose(v["alpha"], al)]
                if len(s) == 0:
                    continue
                tp = boot(s.true_parent_share, rng); er = boot(s.alpha_err, rng)
                rows3.append(dict(case=case, set=k, alpha=al, n=len(s), true_parent_share=tp[0], tp_lo=tp[1], tp_hi=tp[2],
                                  alpha_err=er[0], ae_lo=er[1], ae_hi=er[2]))
    return c1, c2, pd.DataFrame(rows3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--inrange", action="store_true", help="amended range: keep only configs with lambda_a >= 10^-2.5 and l1_ratio >= 0.1 (PNW, Pinatubo); outputs get the _inrange suffix")
    a = ap.parse_args()
    case = a.case
    OUT.mkdir(parents=True, exist_ok=True)
    ref = {}
    if case == "pnw":
        R = case_pnw(a.nproc)
    elif case == "pinatubo":
        R = case_pinatubo(a.nproc)
    elif case == "debby":
        R = case_debby(a.nproc)
    else:
        R, p, dec = case_synthetic()
        R3 = None
    if case in ("pnw", "pinatubo"):
        Rb = pd.read_csv(OUT / f"T1b_{'pinatubo' if case == 'pinatubo' else 'pnw'}_configs.csv").set_index("id")
        ref = {"unfiltered": Rb, "R1 kept": Rb[~Rb.R1_drop], "R2 kept": Rb[~Rb.R2_drop]}
    elif case == "debby":
        Rr = case_debby(a.nproc, parts=sorted((R1 / "night6/t2/debby_final").glob("part[0-9][0-9]")), nv=6)
        Rr = Rr[Rr.retained]
        ref = {"final-set ensemble (no decoy)": Rr}
    else:
        try:
            Rr, _, _ = case_synthetic(d=R1 / "night5/t3/p03")
            Rr["R1_drop"] = False
            ref = {"T2-EN p03 (no extra field)": Rr[Rr.retained]}
        except Exception as e:  # noqa: BLE001
            print("synthetic reference failed:", e)
    suf = ""
    if a.inrange:
        suf = "_inrange"
        n_wide = len(R)
        R = R[(np.log10(R["lam"]) >= -2.5 - 1e-12) & (np.log10(R["l1"]) >= -1.0 - 1e-12)]
        DRAWN[case] = len(R)
        print(f"{case}: {n_wide} wide-range configs run, {len(R)} inside the amended range", flush=True)
    R.to_csv(OUT / f"T1c_{case}{suf}_configs.csv")
    c1, c2, c3 = summarize(case, R, ref)
    c1.to_csv(OUT / f"T1c_{case}{suf}_check1.csv", index=False); c2.to_csv(OUT / f"T1c_{case}{suf}_check2.csv", index=False)
    c3.to_csv(OUT / f"T1c_{case}{suf}_check3.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format, "display.max_rows", 400):
        txt = [f"# T1c {case}", "\n## Check 1: survival (fraction of drawn configs retained / kept)\n```", c1.to_string(index=False), "```",
               "\n## Check 2: log10 lambda_a and l1_ratio of kept vs removed (vs the paper range of the case)\n```", c2.to_string(index=False), "```",
               "\n## Check 3: headline shares / rank-1 fractions with 95 % CIs (T1c vs case-specific)\n```", c3.to_string(index=False), "```"]
    (OUT / f"T1c_{case}{suf}.md").write_text("\n".join(txt))
    print("\n".join(txt[:6]))


if __name__ == "__main__":
    main()
