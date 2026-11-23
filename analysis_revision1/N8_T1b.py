"""Night-8 T1b: decoy filter on one run per case with the typical-field decoy pool (PNW, Pinatubo).

Runs night8/t1b/pnw_p0..p5 (50 run ids each) and night8/t1b/pin: final set + DECOY (last variable), M 30, linear rule.
Criteria pre-declared in the report (T1b pre-declaration, commit bfd9c77): per rule (R1 not rank 1, R2 decoy share > 1/p):
 1 split-half agreement of keep/drop >= 80 % (members 0-14 vs 15-29); 2 better than 95 % of 1,000 random removals (seed 0) of the
 same number of configs on rank-1 agreement (modal rank-1 real variable's fraction) or mean 95 % CI width of the real shares in
 the headline window; 3 removes <= 50 % of the harmonized-retained configs. Shares reported, not deciding.
python N8_T1b.py --case pnw|pinatubo --nproc 32 -> night8/analysis/T1b_<case>_*.csv, T1b_<case>.md, T1b_<case>_D.npz
"""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
T1B = R1 / "night8" / "t1b"
OUT = R1 / "night8" / "analysis"
FINAL = dict(pnw=["Z500", "Z10", "OLR", "TCWV", "SLHF", "SSHF", "T2m"],
             pinatubo=["TMSO201", "BURDENSO401", "TMH2SO401", "AEROD_v", "TMSO203", "BURDENSO403", "T050"])
WINDOWS = dict(pnw=[(1, 10), (11, 20), (21, 30), (11, 30)], pinatubo=[(1, 3), (4, 16)])
HEAD = dict(pnw=(1, 10), pinatubo=(4, 16))
HP = ["timewindow", "box_size", "radius", "eps_dbscan", "min_samples_dbscan", "score_mode", "alpha", "lambda_a", "l1_ratio"]


def member_dens_pnw(args):
    run_dir, nv = args
    p = json.load(open(Path(run_dir) / "params.json"))
    res = pickle.load(open(Path(run_dir) / "ensemble_results.pkl", "rb"))
    D = np.zeros((len(res), 31, nv)); ok = np.zeros(len(res), bool)
    for i, r in enumerate(res):
        if not r.get("success", True):
            continue
        c = np.asarray(r["centers"], float); pa = np.asarray(r["parents"], int)
        n = min(len(c), len(pa), 31)
        if n == 0:
            continue
        ok[i] = True
        cnt = C._box_cell_counts_wrap(c[:n], C.PNW_LONS, C.PNW_LATS, float(p["box_size"]))
        for k in range(1, nv + 1):
            m = pa[:n] == k
            if m.any():
                D[i, np.nonzero(m)[0], k - 1] += cnt[m]
    hp = dict(timewindow=p["timewindow"], box_size=p["box_size"], radius=p["radius"], eps_dbscan=p["eps_dbscan"],
              min_samples_dbscan=p["min_samples_dbscan"], score_mode=p["score_mode"], alpha=p["alpha"],
              lambda_a=p["cd_kwargs"]["lambda_a"], l1_ratio=p["cd_kwargs"]["l1_ratio"])
    return int(Path(run_dir).name.split("_")[1]), D, ok, hp


def member_dens_pin(args):
    npz, box, nv, tid, hp = args
    b = np.load(npz, allow_pickle=True)
    xs = np.asarray(b["xs"], float); ys = np.asarray(b["ys"], float)
    Cc, P, L, okk = b["centers"], b["parents"], b["lengths"], b["ok"]
    D = np.zeros((len(L), Cc.shape[1], nv)); ok = np.zeros(len(L), bool)
    for i in range(len(L)):
        n = int(L[i])
        if int(okk[i]) != 1 or n <= 0:
            continue
        ok[i] = True
        cnt = C._box_cell_counts(Cc[i, :n], xs, ys, float(box))
        par = np.asarray(P[i, :n], int)
        for q in range(1, nv + 1):
            m = par == q
            if m.any():
                D[i, np.nonzero(m)[0], q - 1] += cnt[m]
    return int(tid), D, ok, hp


def load(case, nproc):
    nv = len(FINAL[case]) + 1
    if case == "pnw":
        tabs, jobs = [], []
        for d in sorted(T1B.glob("pnw_p[0-9]")):
            t = pd.read_csv(d / "hyperparam_search_summary.csv"); tabs.append(t)
            jobs += [(str(d / f"run_{int(i):04d}"), nv) for i in t["run_id"]]
        T = pd.concat(tabs, ignore_index=True)
        keep = retention.harmonized_case("pnw", T)
        ids_ret = set(T.loc[keep.to_numpy(), "run_id"].astype(int))
        with Pool(nproc) as pool:
            out = pool.map(member_dens_pnw, jobs)
    else:
        d = T1B / "pin"
        f = d / "results_jobs_final.csv"
        T = pd.read_csv(f if f.exists() else d / "results_jobs_snapshot.csv")
        keep = retention.harmonized_case("pinatubo", T)
        ids_ret = set(T.loc[keep.to_numpy(), "trial_id"].astype(int))
        j = T[T["bundle_npz"].notna()]
        jobs = [(r.bundle_npz, r.box_size, nv, r.trial_id, dict(timewindow=r.timewindow, box_size=r.box_size, radius=r.radius,
                 eps_dbscan=r.eps_dbscan, min_samples_dbscan=r.min_samples_dbscan, score_mode=r.score_mode, alpha=r.alpha,
                 lambda_a=getattr(r, "cd_kwargs_lambda_a", np.nan), l1_ratio=getattr(r, "cd_kwargs_l1_ratio", np.nan)))
                for r in j.rename(columns={"cd_kwargs.lambda_a": "cd_kwargs_lambda_a", "cd_kwargs.l1_ratio": "cd_kwargs_l1_ratio"}).itertuples()]
        with Pool(nproc) as pool:
            out = pool.map(member_dens_pin, jobs)
    return {i: (D, ok, hp) for i, D, ok, hp in out}, ids_ret, len(T)


def shares(D, nv, w=None):
    """D (lead, nv) -> (decoy share all leads step 0 excluded, real shares renormalized in window w or all)."""
    tot = D[1:].sum()
    dshare = D[1:, nv - 1].sum() / tot if tot > 0 else np.nan
    Dw = D[1:] if w is None else D[w[0]:w[1] + 1]
    r = Dw[:, :nv - 1].sum(0)
    return dshare, (r / r.sum() if r.sum() > 0 else np.full(nv - 1, np.nan))


def flags(dshare, real_all, p):
    """R1 / R2 drop flags; real_all = real variables' shares of the total (decoy included in the denominator)."""
    return dshare > np.nanmax(real_all), dshare > 1.0 / p


def boot_width(S, rng, n=2000):
    idx = rng.integers(0, len(S), (n, len(S)))
    m = S[idx].mean(1)
    return float((np.percentile(m, 97.5, axis=0) - np.percentile(m, 2.5, axis=0)).mean())


def modal_frac(S):
    top = np.argmax(S, axis=1)
    return float(np.bincount(top, minlength=S.shape[1]).max() / len(S))


NB_RAND = 500   # bootstrap resamples for the CI width in the random-removal comparison (filtered and random alike)
_G = {}


def _rand_one(k):
    """Random removal k (stream [0, k]): (rank-1 agreement, mean CI width, mean shares by window, rank-1 fractions by window)."""
    g = _G
    rng = np.random.default_rng([0, k])
    kk = np.ones(g["Hn"], bool); kk[rng.choice(g["Hn"], g["n_rm"], replace=False)] = False
    S = g["S"]
    rs = {w: S[w][kk].mean(0) for w in S}
    rr = {w: np.bincount(np.argmax(S[w][kk], 1), minlength=g["nreal"]) / kk.sum() for w in S}
    return modal_frac(S[g["hk"]][kk]), boot_width(S[g["hk"]][kk], rng, n=NB_RAND), rs, rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--nproc", type=int, default=32)
    a = ap.parse_args()
    case = a.case
    real = FINAL[case]; nv = len(real) + 1; p = nv
    OUT.mkdir(parents=True, exist_ok=True)
    data, ids_ret, n_all = load(case, a.nproc)
    ids = sorted(i for i in data if i in ids_ret)
    print(f"{case}: {n_all} configs run, {len(data)} with outputs, {len(ids)} harmonized-retained", flush=True)
    wins = WINDOWS[case]; hw = HEAD[case]
    rows = []
    for i in ids:
        D, ok, hp = data[i]
        Dp = D.sum(0)
        ds, _ = shares(Dp, nv)
        tot = Dp[1:].sum()
        real_all = Dp[1:, :nv - 1].sum(0) / tot
        r1, r2 = flags(ds, real_all, p)
        h = {}
        for name, sel in (("h1", slice(0, 15)), ("h2", slice(15, 30))):
            Dh = D[sel][ok[sel]].sum(0)
            dsh, _ = shares(Dh, nv)
            ra = Dh[1:, :nv - 1].sum(0) / max(Dh[1:].sum(), 1e-30)
            f1, f2 = flags(dsh, ra, p)
            h[name] = (dsh, f1, f2)
        row = dict(id=i, n_ok=int(ok.sum()), decoy_share=ds, R1_drop=bool(r1), R2_drop=bool(r2),
                   decoy_h1=h["h1"][0], decoy_h2=h["h2"][0], R1_h1=bool(h["h1"][1]), R1_h2=bool(h["h2"][1]),
                   R2_h1=bool(h["h2"][2] if False else h["h1"][2]), R2_h2=bool(h["h2"][2]), **{f"hp_{k}": v for k, v in hp.items()})
        for (w0, w1) in wins:
            _, s = shares(Dp, nv, (w0, w1))
            for j, v in enumerate(real):
                row[f"s_{w0}-{w1}_{v}"] = s[j]
        rows.append(row)
    R = pd.DataFrame(rows).set_index("id")
    R.to_csv(OUT / f"T1b_{case}_configs.csv")
    Hn = len(R)
    S = {f"{w0}-{w1}": R[[f"s_{w0}-{w1}_{v}" for v in real]].to_numpy() for (w0, w1) in wins}
    hk = f"{hw[0]}-{hw[1]}"
    rng = np.random.default_rng(0)
    crit, shr, rk = [], [], []
    for rule in ("R1", "R2"):
        drop = R[f"{rule}_drop"].to_numpy()
        n_rm = int(drop.sum()); keep = ~drop
        agree = float((R[f"{rule}_h1"] == R[f"{rule}_h2"]).mean())
        Af = modal_frac(S[hk][keep]) if keep.any() else np.nan
        Bf = boot_width(S[hk][keep], np.random.default_rng(0), n=NB_RAND) if keep.sum() > 1 else np.nan
        A0, B0 = modal_frac(S[hk]), boot_width(S[hk], np.random.default_rng(0), n=NB_RAND)
        Ar, Br, Rs, Rr = [], [], {w: [] for w in S}, {w: [] for w in S}
        if n_rm > 0 and keep.sum() > 1:
            _G.update(S=S, hk=hk, n_rm=n_rm, Hn=Hn, nreal=len(real))
            with Pool(a.nproc) as pool:
                res = pool.map(_rand_one, range(1000))
            for ar, br, rs, rr_ in res:
                Ar.append(ar); Br.append(br)
                for w in S:
                    Rs[w].append(rs[w]); Rr[w].append(rr_[w])
            Ar, Br = np.array(Ar), np.array(Br)
            Bf = boot_width(S[hk][keep], np.random.default_rng(0), n=NB_RAND)   # same resample count as the random sets
        c1 = agree >= 0.8
        ca = (Af > np.percentile(Ar, 95)) if len(Ar) else False
        cb = (Bf < np.nanpercentile(Br, 5)) if len(Ar) and np.isfinite(Br).any() else False
        c2 = bool(ca or cb)
        c3 = (n_rm / Hn) <= 0.5
        crit.append(dict(case=case, rule=rule, n_retained=Hn, n_removed=n_rm, frac_removed=n_rm / Hn, split_half_agreement=agree,
                         crit1_stable=bool(c1), rank1_agree_filtered=Af, rank1_agree_unfiltered=A0,
                         rank1_agree_random_p95=float(np.percentile(Ar, 95)) if len(Ar) else np.nan,
                         rank1_agree_pctile=float((Ar < Af).mean()) if len(Ar) else np.nan,
                         ciwidth_filtered=Bf, ciwidth_unfiltered=B0,
                         ciwidth_random_p05=float(np.nanpercentile(Br, 5)) if len(Ar) and np.isfinite(Br).any() else np.nan,
                         crit2_better_than_chance=c2, crit3_not_destructive=bool(c3), adopt=bool(c1 and c2 and c3)))
        for w in S:
            for j, v in enumerate(real):
                shr.append(dict(case=case, rule=rule, window=w, var=v, share_unfiltered=S[w][:, j].mean(),
                                share_filtered=S[w][keep][:, j].mean() if keep.any() else np.nan,
                                share_random_median=float(np.median(np.array(Rs[w])[:, j])) if Rs[w] else np.nan,
                                rank1_unfiltered=float((np.argmax(S[w], 1) == j).mean()),
                                rank1_filtered=float((np.argmax(S[w][keep], 1) == j).mean()) if keep.any() else np.nan,
                                rank1_random_median=float(np.median(np.array(Rr[w])[:, j])) if Rr[w] else np.nan))
        if len(Ar):
            pd.DataFrame(dict(rank1_agree=Ar, ciwidth=Br)).to_csv(OUT / f"T1b_{case}_{rule}_random.csv", index=False)
    Cc, Sh = pd.DataFrame(crit), pd.DataFrame(shr)
    Cc.to_csv(OUT / f"T1b_{case}_criteria.csv", index=False); Sh.to_csv(OUT / f"T1b_{case}_shares.csv", index=False)
    # C3: hyperparameters of removed vs kept
    c3 = []
    hpc = [c for c in R.columns if c.startswith("hp_")]
    for rule in ("R1", "R2"):
        for grp, m in (("removed", R[f"{rule}_drop"]), ("kept", ~R[f"{rule}_drop"])):
            g = R[m]
            for c in hpc:
                if g.empty:
                    continue
                if g[c].dtype == object:
                    c3.append(dict(case=case, rule=rule, group=grp, hp=c[3:], n=len(g), summary=g[c].value_counts().to_dict()))
                else:
                    c3.append(dict(case=case, rule=rule, group=grp, hp=c[3:], n=len(g),
                                   summary=f"median {g[c].median():.4g} [q25 {g[c].quantile(.25):.4g}, q75 {g[c].quantile(.75):.4g}]"))
    pd.DataFrame(c3).to_csv(OUT / f"T1b_{case}_C3.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.float_format", "{:.3f}".format, "display.max_colwidth", 120):
        txt = [f"# T1b {case}: {n_all} configs run, {len(ids)} harmonized-retained; decoy share mean {R.decoy_share.mean():.3f}, "
               f"range {R.decoy_share.min():.3f}-{R.decoy_share.max():.3f}, 1/p = {1 / p:.3f}", "\n## Criteria\n```", Cc.T.to_string(), "```",
               "\n## Shares by window (filtered / unfiltered / random-removal median) and rank-1 fractions\n```", Sh.to_string(index=False), "```",
               "\n## C3: hyperparameters, removed vs kept (no interpretation)\n```", pd.DataFrame(c3).to_string(index=False), "```"]
    (OUT / f"T1b_{case}.md").write_text("\n".join(txt))
    print("\n".join(txt[:5]))


if __name__ == "__main__":
    main()
