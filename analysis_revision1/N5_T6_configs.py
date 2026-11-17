"""Night-5 T6, pre-declared rule 5: the 8 representative configs per case for the surrogate sweep.

= the paper's illustrative config (audit 2026-11-01: Debby 246, Pinatubo 929, PNW 205) + 7 configs chosen by
k-medoids (seed 0) on the standardized hyperparameters of the retained configs (harmonized rule) of the
extended-set run (night5/t2/<case>_ext; fallback: night-1 baseline, recorded). Numeric parameters z-scored
(log10 for lambda_a and l1_ratio); categorical parameters one-hot (score_mode). Parameters that are fixed in the
search space or irrelevant under the forced linear rule (prob_rule, beta_softmax) are not used.
k-medoids: k-means++-style seeding with default_rng(0), then alternating assignment / medoid update (PAM-like,
Euclidean) until stable; the illustrative config is excluded from the candidate pool.

python N5_T6_configs.py --case pnw [--fallback]   (PBS only) -> night5/t6/<case>_configs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from N5_T2 import load_table, CASES  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
OUT = R1 / "night5" / "t6"
ILLUSTRATIVE = dict(debby=246, pinatubo=929, pnw=205)
FALLBACK = dict(debby=R1 / "T4r_debby_vorebuilt", pinatubo=R1 / "T5r_pinatubo", pnw=R1 / "T6r_pnw")
NUM = ["timewindow_h", "box_size", "radius", "eps_dbscan", "alpha", "log_lambda_a", "log_l1_ratio"]


def params_table(case, t):
    idc = "run_id" if case == "pnw" else "trial_id"
    if case == "pnw":
        rows = []
        for i, d in zip(t[idc], t["_dir"]):
            p = json.load(open(Path(d) / f"run_{int(i):04d}" / "params.json"))
            rows.append(dict(id=int(i), timewindow=p["timewindow"], box_size=p["box_size"], radius=p["radius"],
                             eps_dbscan=p["eps_dbscan"], alpha=p["alpha"], score_mode=p["score_mode"],
                             lambda_a=p["cd_kwargs"]["lambda_a"], l1_ratio=p["cd_kwargs"]["l1_ratio"]))
        P = pd.DataFrame(rows)
    else:
        P = pd.DataFrame(dict(id=t[idc].astype(int), timewindow=t["timewindow"], box_size=t["box_size"], radius=t["radius"],
                              eps_dbscan=t["eps_dbscan"], alpha=t["alpha"], score_mode=t["score_mode"],
                              lambda_a=t["cd_kwargs.lambda_a"], l1_ratio=t["cd_kwargs.l1_ratio"]))
    P["timewindow_h"] = pd.to_timedelta(P["timewindow"]).dt.total_seconds() / 3600
    P["log_lambda_a"] = np.log10(P["lambda_a"]); P["log_l1_ratio"] = np.log10(P["l1_ratio"])
    return P


def kmedoids(X, k, seed=0, n_iter=100):
    rng = np.random.default_rng(seed)
    n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    med = [int(rng.integers(n))]
    while len(med) < k:
        d2 = D[:, med].min(1) ** 2
        med.append(int(rng.choice(n, p=d2 / d2.sum())))
    for _ in range(n_iter):
        lab = D[:, med].argmin(1)
        new = []
        for j in range(k):
            idx = np.nonzero(lab == j)[0]
            new.append(int(idx[np.argmin(D[np.ix_(idx, idx)].sum(1))]) if len(idx) else med[j])
        if new == med:
            break
        med = new
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list(CASES))
    ap.add_argument("--fallback", action="store_true", help="use the night-1 baseline instead of the extended run")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.fallback:
        d = FALLBACK[a.case]
        f = d / ("hyperparam_search_summary.csv" if a.case == "pnw" else "results_jobs_final.csv")
        t = pd.read_csv(f).assign(_dir=str(d)); src = str(d)
    else:
        t = load_table(a.case, CASES[a.case]["ext"][0], partial=False); src = CASES[a.case]["ext"][0]
    keep = retention.harmonized_case(a.case, t)
    P = params_table(a.case, t[keep])
    ill = ILLUSTRATIVE[a.case]
    pool = P[P["id"] != ill].reset_index(drop=True)
    Z = pool[NUM].astype(float)
    Z = (Z - Z.mean()) / Z.std(ddof=0).replace(0, 1)
    Z = pd.concat([Z, pd.get_dummies(pool["score_mode"], prefix="score").astype(float)], axis=1).to_numpy()
    med = kmedoids(Z, 7, seed=0)
    ids = [ill] + [int(pool.loc[m, "id"]) for m in med]
    out = dict(case=a.case, source=src, n_retained=int(keep.sum()), illustrative=ill,
               illustrative_retained=bool((P["id"] == ill).any()), kmedoids_ids=ids[1:], configs=ids,
               features=NUM + ["score_mode one-hot"],
               params=P.set_index("id").loc[[i for i in ids if i in set(P["id"])]].reset_index().to_dict(orient="records"))
    json.dump(out, open(OUT / f"{a.case}_configs.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "params"}, indent=1))
    print(pd.DataFrame(out["params"]).to_string())


if __name__ == "__main__":
    main()
