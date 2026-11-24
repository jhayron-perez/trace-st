"""Night-9 N2 analysis (S11): performance vs number of typical fields K (K = 1 is the decoy run itself).
Runs night9/n2/<case>_d<dd>/K<kk>_c<cfg> (K = 3..15, draws 0-2, 5 configs) + K = 1 from the decoy run (PNW/Pinatubo: night8/t1b;
Debby: night9/n1). Per run: densities pooled over members (harmonized retention per run); per window: real shares renormalized among
the real variables, Kendall tau of the real shares vs the same config's K = 1 shares, leaders of the window (PNW 1-10 d Z500, TCWV;
11-30 d OLR, SLHF, SSHF; Pinatubo 4-16 d SO2; Debby 1-4 and 5-15 d vo): 'outrank_any' = largest added field's share (of the total,
added fields included) > the smallest leader's share, 'outrank_all' = > the largest leader's share; mean share per added field vs 1/p
(p = real + K), largest single added field. 95 % bootstrap CIs over runs (2,000, seed 0).
python N9_n2_an.py --case pnw|pinatubo|debby --nproc 32"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N8_T1b as B  # noqa: E402
import N8_T1c as C1  # noqa: E402
import N6_T3T4 as T  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = B.R1
OUT = R1 / "night9" / "analysis"
REAL = C1.REAL
WIN = dict(pnw=[(1, 10), (11, 30)], pinatubo=[(4, 16)], debby=[(1, 4), (5, 15)])
LEAD = dict(pnw={(1, 10): ["Z500", "TCWV"], (11, 30): ["OLR", "SLHF", "SSHF"]}, pinatubo={(4, 16): ["TMSO201"]},
            debby={(1, 4): ["vo"], (5, 15): ["vo"]})
KS = [1, 3, 6, 9, 12, 15]
ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--nproc", type=int, default=32)
a = ap.parse_args()
case = a.case
real = REAL[case]; nr = len(real)
cfgs = json.load(open(R1 / f"night9/n2_{case}_configs.json"))["configs"]


def D_of_dir(d, nv):
    """{cfg: pooled D (lead, nv)} for the retained configs of one run directory."""
    if case == "pnw":
        t = pd.read_csv(d / "hyperparam_search_summary.csv")
        keep = retention.harmonized_case("pnw", t)
        ids = set(t.loc[keep.to_numpy(), "run_id"].astype(int))
        out = {}
        for i in t["run_id"].astype(int):
            _, D, ok, _ = B.member_dens_pnw((str(d / f"run_{int(i):04d}"), nv))
            out[int(i)] = (D.sum(0), int(i) in ids)
        return out
    if case == "pinatubo":
        f = d / "results_jobs_final.csv"
        t = pd.read_csv(f if f.exists() else d / "results_jobs_snapshot.csv")
        keep = retention.harmonized_case("pinatubo", t)
        ids = set(t.loc[keep.to_numpy(), "trial_id"].astype(int))
        out = {}
        for r in t[t["bundle_npz"].notna()].itertuples():
            _, D, ok, _ = B.member_dens_pin((r.bundle_npz, r.box_size, nv, r.trial_id, {}))
            out[int(r.trial_id)] = (D.sum(0), int(r.trial_id) in ids)
        return out
    t, ids, all_ids = T.table_and_ids("debby", d, True)
    if t is None:
        return {}
    Dd, _ = T.run_densities("debby", d, nv, True, a.nproc)
    return {int(i): (np.asarray(Dd[i], float), True) for i in Dd} | {int(i): (None, False) for i in all_ids if i not in Dd}


def metrics(D, K, base_shares):
    """D (lead, nr + K); returns {window: dict}."""
    nv = nr + K; res = {}
    for (w0, w1) in WIN[case]:
        Dw = D[w0:w1 + 1].sum(0)
        tot = Dw.sum()
        if tot <= 0:
            continue
        rs = Dw[:nr] / max(Dw[:nr].sum(), 1e-30)
        added = Dw[nr:] / tot
        lead = [real.index(v) for v in LEAD[case][(w0, w1)]]
        ls = Dw[lead] / tot
        tau = kendalltau(rs, base_shares[(w0, w1)])[0] if base_shares is not None else 1.0
        res[(w0, w1)] = dict(real=rs, tau=float(tau) if np.isfinite(tau) else np.nan, outrank_any=bool(added.max() > ls.min()),
                             outrank_all=bool(added.max() > ls.max()), added_mean=float(added.mean()), added_max=float(added.max()),
                             over_1p=float(added.mean() * nv), added_sum=float(added.sum()))
    return res


rows = []
# K = 1 base runs
if case in ("pnw", "pinatubo"):
    data, ids_ret, _ = B.load(case, a.nproc)
    base = {c: (data[c][0].sum(0), c in ids_ret) for c in cfgs if c in data}
else:
    base = {}
    for p in sorted((R1 / "night9/n1").glob("deb_p[0-9]")):
        for i, v in D_of_dir(p, nr + 1).items():
            if i in cfgs:
                base[i] = v
base_sh = {}
for c, (D, ok) in base.items():
    if not ok or D is None:
        continue
    r1 = metrics(D, 1, None)
    base_sh[c] = {w: r1[w]["real"] for w in r1}
    for w, m in r1.items():
        rows.append(dict(case=case, draw=-1, K=1, cfg=c, retained=True, window=f"{w[0]}-{w[1]}", **{k: v for k, v in m.items() if k != "real"},
                         **{f"s_{x}": m["real"][j] for j, x in enumerate(real)}))
for dd in range(3):
    for K in KS[1:]:
        for c in cfgs:
            d = R1 / f"night9/n2/{case}_d{dd}/K{K:02d}_c{c:03d}"
            if not d.exists():
                continue
            try:
                got = D_of_dir(d, nr + K)
            except Exception as e:  # noqa: BLE001
                print("fail", d, e, flush=True); continue
            D, ok = got.get(c, (None, False))
            if not ok or D is None or c not in base_sh:
                rows.append(dict(case=case, draw=dd, K=K, cfg=c, retained=False))
                continue
            for w, m in metrics(D, K, base_sh[c]).items():
                rows.append(dict(case=case, draw=dd, K=K, cfg=c, retained=True, window=f"{w[0]}-{w[1]}",
                                 **{k: v for k, v in m.items() if k != "real"}, **{f"s_{x}": m["real"][j] for j, x in enumerate(real)}))
Rn = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
Rn.to_csv(OUT / f"N2_{case}_runs.csv", index=False)
rng = np.random.default_rng(0)


def boot(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan,) * 3
    b = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


summ = []
for K in KS:
    n_expected = 5 if K == 1 else 15
    sub = Rn[(Rn.K == K)]
    n_ret = int(sub.drop_duplicates(["draw", "cfg"]).retained.sum()) if len(sub) else 0
    for (w0, w1) in WIN[case]:
        g = sub[(sub.retained == True) & (sub.window == f"{w0}-{w1}")]  # noqa: E712
        if g.empty:
            summ.append(dict(case=case, K=K, window=f"{w0}-{w1}", runs_retained=n_ret, runs_expected=n_expected)); continue
        e = dict(case=case, K=K, window=f"{w0}-{w1}", runs_retained=n_ret, runs_expected=n_expected, p=nr + K)
        for x in real:
            m, lo, hi = boot(g[f"s_{x}"]); e[f"share_{x}"], e[f"lo_{x}"], e[f"hi_{x}"] = m, lo, hi
        for k in ("tau", "outrank_any", "outrank_all", "added_mean", "added_max", "over_1p"):
            m, lo, hi = boot(g[k].astype(float)); e[k], e[k + "_lo"], e[k + "_hi"] = m, lo, hi
        e["one_over_p"] = 1.0 / (nr + K)
        summ.append(e)
S = pd.DataFrame(summ)
S.to_csv(OUT / f"N2_{case}_summary.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.3f}".format):
    (OUT / f"N2_{case}.md").write_text(f"# N2 / S11 {case}: configs {cfgs}\n```\n"
                                      + S[["K", "window", "runs_retained", "runs_expected", "p", "tau", "tau_lo", "tau_hi", "outrank_any", "outrank_all",
                                           "added_mean", "one_over_p", "added_max"]].to_string(index=False) + "\n```\n")
print((OUT / f"N2_{case}.md").read_text())
