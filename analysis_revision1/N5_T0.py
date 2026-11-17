"""Night-5 T0: probability-rule split and harmonized retention on the existing (night-1) outputs.

Rule 2 (pre-declared): (i) T2-EN alpha recovery of the linear-only configs (slope of the median recovered
density alpha vs the prescribed grid, as analysis_revision1/alpha_gate.py); (ii) top variable in each real
case's headline window, linear-only subset vs full ensemble (old retention; also harmonized).
Rule 3: harmonized retention counts (old vs new), threshold sweep, headline shares under old vs new.

python N5_T0.py [--nproc 8]      (PBS only)
Outputs: revision1/night5/T0_rule_split.csv, T0_retention_counts.csv, T0_retention_sweep.csv,
         T0_headline_old_new.csv, T0_verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from A4_twins import boot_ci, retained_ids  # noqa: E402
from N3_bed_summary import densities, window_shares  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = C.R1
N5 = R1 / "night5"
ORIG = Path("/glade/work/jhayron/trace-st_data/original_submission")
HEAD = dict(debby="w5-15", pinatubo="w4-16", pnw="w1-10")
DIRS = dict(debby=R1 / "T4r_debby_vorebuilt", pinatubo=R1 / "T5r_pinatubo", pnw=R1 / "T6r_pnw")
BETA_BINS = [(-0.1, 1.0, "beta<=1"), (1.5, 8.0, "beta 2-8"), (15.0, 64.0, "beta 16-64")]


def alpha_fit(df, label):
    df = df[np.isfinite(df["alpha_mix"]) & df["estimated_alpha_density"].notna()]
    if df["trial_id"].nunique() < 3:
        return None
    med = df.groupby("alpha_mix")["estimated_alpha_density"].median()
    a = med.index.values.astype(float)
    slope, icpt = np.polyfit(a, med.values, 1)
    err = (df["estimated_alpha_density"] - df["alpha_mix"]).abs()
    return dict(section="T2_alpha", subset=label, n_configs=int(df["trial_id"].nunique()), slope=float(slope),
                intercept=float(icpt), r_median=float(np.corrcoef(a, med.values)[0, 1]),
                r_pooled=float(np.corrcoef(df["alpha_mix"], df["estimated_alpha_density"])[0, 1]),
                median_abs_err=float(err.median()),
                **{f"median_alpha_{x:g}": float(v) for x, v in med.items()})


def t2_rows():
    rows = []
    for src, d in (("paper (multivariate_v4)", ORIG / "ResultsSynthetic" / "multivariate_v4"),
                   ("night-1 re-run (T2r_multivariate_en)", R1 / "T2r_multivariate_en")):
        f = d / "results_alphas_final.csv"
        if not f.exists():
            print("missing", f); continue
        x = pd.read_csv(f)
        subsets = [("all", x), ("linear", x[x["prob_rule"] == "linear"]), ("softmax", x[x["prob_rule"] == "softmax"])]
        for lo, hi, nm in BETA_BINS:
            s = x[(x["prob_rule"] == "softmax") & (x["beta_softmax"] > lo) & (x["beta_softmax"] <= hi)]
            subsets.append((f"softmax {nm}", s))
        for nm, s in subsets:
            r = alpha_fit(s, nm)
            if r:
                r["source"] = src
                rows.append(r)
    return rows


def case_tables():
    t = {k: pd.read_csv(DIRS[k] / "results_jobs_final.csv") for k in ("debby", "pinatubo")}
    p = pd.read_csv(DIRS["pnw"] / "hyperparam_search_summary.csv")
    p["prob_rule"] = [json.load(open(DIRS["pnw"] / f"run_{int(i):04d}" / "params.json"))["prob_rule"] for i in p["run_id"]]
    p["beta_softmax"] = [json.load(open(DIRS["pnw"] / f"run_{int(i):04d}" / "params.json"))["beta_softmax"] for i in p["run_id"]]
    t["pnw"] = p
    return t


def idcol(case):
    return "run_id" if case == "pnw" else "trial_id"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    N5.mkdir(parents=True, exist_ok=True)
    rows = t2_rows()
    tabs = case_tables()
    counts, sweep, head = [], [], []
    verdict = {}
    for case, tab in tabs.items():
        vars_ = C.VARS[case]
        ids_old = set(retained_ids(case))
        new_mask = retention.harmonized_case(case, tab)
        ids_new = set(tab.loc[new_mask, idcol(case)].astype(int))
        ids_all = sorted(set(tab[idcol(case)].astype(int)))
        counts.append(dict(case=case, n_total=len(ids_all), n_old=len(ids_old), n_new=len(ids_new),
                           n_both=len(ids_old & ids_new), n_old_only=len(ids_old - ids_new), n_new_only=len(ids_new - ids_old)))
        for mf in (0.5, 0.6, 2 / 3, 0.75):
            for lf in (0.6, 0.7, 0.8, 0.9):
                m = retention.harmonized_case(case, tab, min_member_frac=mf, min_len_frac=lf)
                sweep.append(dict(case=case, min_member_frac=round(mf, 3), min_len_frac=lf, n_retained=int(m.sum()),
                                  n_total=len(tab)))
        Dd, _ = densities(case, DIRS[case], len(vars_), ids_all, a.nproc, 0)
        sh = {c: window_shares(D, case) for c, D in Dd.items()}
        rule = dict(zip(tab[idcol(case)].astype(int), tab["prob_rule"]))
        tops = {}
        for retn, ids in (("old", ids_old), ("harmonized", ids_new)):
            for subset in ("all", "linear", "softmax"):
                cf = [c for c in ids if c in sh and HEAD[case] in sh[c] and (subset == "all" or rule[c] == subset)]
                if not cf:
                    continue
                for w in sorted({w for c in cf for w in sh[c]}):
                    cfw = [c for c in cf if w in sh[c]]
                    S = np.array([sh[c][w] for c in cfw])
                    means = S.mean(0)
                    rk1 = np.bincount(S.argmax(1), minlength=len(vars_)) / len(S)
                    for i, v in enumerate(vars_):
                        m, lo, hi = boot_ci(S[:, i])
                        head.append(dict(case=case, retention=retn, subset=subset, window=w, var=v, n_cfg=len(S),
                                         share=m, lo=lo, hi=hi, frac_rank1=rk1[i]))
                    if w == HEAD[case]:
                        top = vars_[int(np.argmax(means))]
                        order = [vars_[k] for k in np.argsort(-means)]
                        tops[(retn, subset)] = top
                        rows.append(dict(section="real_headline", source=case, subset=f"{subset} ({retn} retention)",
                                         n_configs=len(S), window=w, top_var=top, top2=",".join(order[:2]),
                                         **{f"share_{v}": float(means[i]) for i, v in enumerate(vars_)}))
        verdict[case] = dict(top_full_old=tops.get(("old", "all")), top_linear_old=tops.get(("old", "linear")),
                             top_full_new=tops.get(("harmonized", "all")), top_linear_new=tops.get(("harmonized", "linear")),
                             same_old=tops.get(("old", "all")) == tops.get(("old", "linear")))
    t2 = [r for r in rows if r["section"] == "T2_alpha" and r["source"].startswith("night-1") and r["subset"] == "linear"]
    vi = dict(slope=t2[0]["slope"], r=t2[0]["r_median"], n=t2[0]["n_configs"]) if t2 else {}
    vi["pass"] = bool(t2) and abs(vi["slope"] - 0.84) <= 0.05 and vi["r"] >= 0.99
    vii = all(v["same_old"] for v in verdict.values())
    out = dict(rule2_i=vi, rule2_ii=verdict, rule2_ii_pass=bool(vii), rule2_pass=bool(vi["pass"] and vii))
    pd.DataFrame(rows).to_csv(N5 / "T0_rule_split.csv", index=False)
    pd.DataFrame(counts).to_csv(N5 / "T0_retention_counts.csv", index=False)
    pd.DataFrame(sweep).to_csv(N5 / "T0_retention_sweep.csv", index=False)
    pd.DataFrame(head).to_csv(N5 / "T0_headline_old_new.csv", index=False)
    json.dump(out, open(N5 / "T0_verdict.json", "w"), indent=1)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200):
        print(pd.DataFrame(rows).round(3).to_string())
        print(pd.DataFrame(counts).to_string())
        print(pd.DataFrame(sweep).pivot_table(index=["case", "min_member_frac"], columns="min_len_frac",
                                              values="n_retained").to_string())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
