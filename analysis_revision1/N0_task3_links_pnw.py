"""Night-3 N0 task 3 (H1, link level), PNW A1 twin runs.

Graph storage: PNW ensemble_results.pkl keeps run_track(return_debug=True) output: out["debug"][k] is the
candidate table of regression k (box at centers[k], child = parents[k]): every non-zero link
p -> child with causal_value (prefer_sign='both' => all non-zero coefficients, |coef| > dependence_threshold).
Debby / Pinatubo bundles store centers/parents only (no graphs).

Regression k is assigned lead k (days); k = 0 (starting box) excluded as in A4. Windows 1-10/11-20/21-30.
For each run (var stem x twin year), config, member, regression k and role in {real, twin}:
  nz = number of non-zero coefficients on that variable's stencil block, any = nz > 0,
  |beta| values of that block (log-histogram; median / p90 from the histogram).
Split by child_is_self (the regression's child is the variable itself, so its block holds its own lag).
Config-level N_samples = n_ref*(W-1), n_ref = (round(box/2)+1-2r)^2; N_features = 7*(2r+1)^2.
python N0_task3_links_pnw.py [--nproc N]   (PBS only)
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
import A4_twins as A  # noqa: E402

R1 = A.R1
OUT = R1 / "night3/N0/task3"
EDGES = np.logspace(-8, 2, 501)
WIN = [(1, 10), (11, 20), (21, 30)]


def wname(k):
    for a, b in WIN:
        if a <= k <= b:
            return f"w{a}-{b}"
    return None


def one(args):
    run_dir, real_idx, twin_idx = args
    res = pickle.load(open(Path(run_dir) / "ensemble_results.pkl", "rb"))
    acc = {}  # (win, role, self) -> [n_reg, n_any, sum_nz, hist]
    for r in res:
        if not r.get("success", False) or r.get("out") is None:
            continue
        dbg = r["out"].get("debug")
        par = np.asarray(r["parents"], int)
        if dbg is None:
            continue
        for k, d in enumerate(dbg):
            if k == 0:
                continue
            w = wname(k)
            if w is None:
                continue
            child = int(par[k])
            if d is None or len(d) == 0:
                pp = np.zeros(0, int); av = np.zeros(0)
            else:
                pp = d["parent_p"].to_numpy(int) if "parent_p" in d else \
                    d["vars"].map(lambda s: int(str(s).split("_")[0][1:])).to_numpy(int)
                av = np.abs(d["causal_value"].to_numpy(float))
                ok = av > 0
                pp, av = pp[ok], av[ok]
            for role, idx in (("real", real_idx), ("twin", twin_idx)):
                m = pp == idx
                key = (w, role, child == idx)
                a = acc.setdefault(key, [0, 0, 0, np.zeros(len(EDGES) - 1)])
                a[0] += 1; a[1] += int(m.any()); a[2] += int(m.sum())
                if m.any():
                    a[3] += np.histogram(np.clip(av[m], EDGES[0], EDGES[-1] * 0.999), EDGES)[0]
    return int(Path(run_dir).name.split("_")[1]), acc


def hq(h, q):
    c = np.cumsum(h)
    if c[-1] == 0:
        return np.nan
    i = np.searchsorted(c, q * c[-1])
    return float(np.sqrt(EDGES[i] * EDGES[i + 1]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=60); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = A.retained_ids("pnw")
    P = pd.DataFrame([json.load(open(R1 / f"T6r_pnw/run_{i:04d}/params.json")) for i in ids])
    P["cfg"] = P["run_id"].astype(int)
    W = (pd.to_timedelta(P["timewindow"]) / pd.Timedelta("1d")).round().astype(int)
    r = P["radius"].astype(int)
    nref = np.clip((P["box_size"] / 2.0).round().astype(int) + 1 - 2 * r, 0, None) ** 2
    P["N_samples"] = nref * (W - 1); P["N_features"] = 7 * (2 * r + 1) ** 2
    P["ratio"] = P["N_samples"] / P["N_features"]
    P["lambda_a"] = [d["lambda_a"] for d in P["cd_kwargs"]]
    P["ratio_bin"] = pd.qcut(P["ratio"], 4, duplicates="drop").astype(str)
    rb = P.set_index("cfg")["ratio_bin"]
    rows = []
    tasks = []
    for stem, nm in A.PNW_STEM.items():
        for y in A.A1_YEARS:
            d = R1 / "A1_pnw_twin" / f"{stem}_{y}"
            for i in ids:
                rd = d / f"run_{i:04d}"
                if (rd / "ensemble_results.pkl").exists():
                    tasks.append((str(rd), A.PNW6.index(nm) + 1, 7, nm, y))
    print("n run-configs", len(tasks), flush=True)
    with Pool(a.nproc) as pool:
        outs = pool.map(one, [t[:3] for t in tasks], chunksize=4)
    H = {}
    for t, (cfg, acc) in zip(tasks, outs):
        for (w, role, slf), (nreg, nany, snz, h) in acc.items():
            rows.append(dict(var=t[3], year=t[4], cfg=cfg, window=w, role=role, child_is_self=slf, n_reg=nreg,
                             n_any=nany, sum_nz=snz))
            for key in [(t[3], w, role, slf, "all"), (t[3], w, role, slf, rb[cfg])]:
                H[key] = H.get(key, 0) + h
    L = pd.DataFrame(rows)
    L.to_csv(OUT / "task3_pnw_links_long.csv.gz", index=False)
    st = (2 * P.set_index("cfg")["radius"].astype(int) + 1) ** 2
    L["stencil"] = L["cfg"].map(st)
    L["ratio_bin"] = L["cfg"].map(rb)
    out = []
    for binned in (False, True):
        keys = ["var", "window", "role", "child_is_self"] + (["ratio_bin"] if binned else [])
        for k, g in L.groupby(keys):
            # per config: frac of regressions with any non-zero on block; mean nz fraction of block
            pc = g.groupby("cfg").apply(lambda x: pd.Series(dict(
                frac_any=x.n_any.sum() / max(x.n_reg.sum(), 1),
                nz_frac_block=x.sum_nz.sum() / max((x.n_reg * x.stencil).sum(), 1),
                n_reg=x.n_reg.sum())))
            m, lo, hi = A.boot_ci(pc["frac_any"].values)
            kk = dict(zip(keys, k))
            h = H.get((kk["var"], kk["window"], kk["role"], kk["child_is_self"], kk["ratio_bin"] if binned else "all"))
            out.append(dict(kk, ratio_bin=kk.get("ratio_bin", "all"), n_cfg=len(pc), n_reg=int(pc.n_reg.sum()),
                            frac_any_mean=m, frac_any_lo=lo, frac_any_hi=hi, nz_frac_block=pc.nz_frac_block.mean(),
                            absbeta_median=hq(h, .5) if h is not None else np.nan,
                            absbeta_p90=hq(h, .9) if h is not None else np.nan))
    S = pd.DataFrame(out)
    S.to_csv(OUT / "task3_pnw_links_summary.csv", index=False)
    # share vs ratio bin (from A4 long), real and twin
    x = pd.read_csv(R1 / "analysis_v2/A4_twins/pnw_A1_long.csv.gz", keep_default_na=False)
    x = x[(x.twin != "") & x.lead.astype(str).str.startswith("w")].copy()
    x["share_real"] = x.share_real.astype(float); x["share_twin"] = x.share_twin.astype(float)
    pc = x.groupby(["var", "lead", "cfg"])[["share_real", "share_twin"]].mean().reset_index()
    pc["ratio_bin"] = pc.cfg.map(rb)
    sv = pc.groupby(["var", "lead", "ratio_bin"]).agg(n=("cfg", "size"), real=("share_real", "mean"),
                                                     twin=("share_twin", "mean")).reset_index()
    sv["diff"] = sv.real - sv.twin
    sv.to_csv(OUT / "task3_pnw_share_by_ratio.csv", index=False)
    pd.set_option("display.width", 250)
    print(P.groupby("ratio_bin")["ratio"].agg(["min", "max", "size"]).to_string())
    print(S[(S.ratio_bin == "all")].round(4).to_string(index=False))
    print(S[(S.ratio_bin != "all") & (~S.child_is_self)].round(4).to_string(index=False))
    print(sv.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
