"""Night-3 N0 task 2: parameter dependence of diff = share(real) - share(twin), descriptive only.

Per config: diff = mean over twin runs (years) of window share(real) - share(twin), from the A4 long tables
(revision1/analysis_v2/A4_twins/<set>_long.csv.gz; same definition as A4_twins.py).
Sets: pnw_A1 (6 vars x 5 years; windows 1-10/11-20/21-30), debby_A2 (vo; 1-4/5-15),
pinatubo_A3 (TMSO201, BURDENSO401; 1-3/4-16; averaged over the single-twin and the combined run, as in A4).
Config parameters: PNW run_XXXX/params.json of T6r_pnw; Debby/Pinatubo results_jobs_final.csv of the baselines.
Every parameter that varies across the analysed configs is binned: quartiles if > 6 distinct values,
otherwise its levels. Derived: W (steps), stencil size (2r+1)^2, n_ref cells = (round(box/res)+1-2r)^2,
N_samples = n_ref*(W-1), N_features = n_vars*(2r+1)^2, ratio N_samples/N_features.
Per bin: n configs, mean diff, 95 % bootstrap CI (10,000, seed 0), flag CI>0 / CI<0.
python N0_task2_param_dependence.py   (PBS only)
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import A4_twins as A  # noqa: E402

R1 = A.R1
OUT = R1 / "night3/N0/task2"
NV = dict(pnw_A1=7, debby_A2=4, pinatubo_A3=5)


def cfg_params(setname):
    if setname == "pnw_A1":
        ids = A.retained_ids("pnw")
        P = pd.DataFrame([json.load(open(R1 / f"T6r_pnw/run_{i:04d}/params.json")) for i in ids])
        P["cfg"] = P["run_id"].astype(int)
        ck = pd.json_normalize(P["cd_kwargs"]).add_prefix("cd_kwargs.")
        P = pd.concat([P.drop(columns=["cd_kwargs"]), ck], axis=1)
    else:
        base = "T4r_debby_vorebuilt" if setname == "debby_A2" else "T5r_pinatubo"
        P = pd.read_csv(R1 / base / "results_jobs_final.csv")
        P["cfg"] = P["trial_id"].astype(int)
    return P


def derive(P, nvars):
    tw = pd.to_timedelta(P["timewindow"]); tr = pd.to_timedelta(P["timeres"])
    P["W_steps"] = (tw / tr).round().astype(int)
    r = P["radius"].astype(int)
    P["stencil"] = (2 * r + 1) ** 2
    nside = (P["box_size"].astype(float) / P["spaceres"].astype(float)).round().astype(int) + 1
    P["n_ref"] = np.clip(nside - 2 * r, 0, None) ** 2
    P["N_samples"] = P["n_ref"] * (P["W_steps"] - 1)
    P["N_features"] = nvars * P["stencil"]
    P["ratio_Ns_Nf"] = P["N_samples"] / P["N_features"]
    return P


SKIP = {"cfg", "run_id", "trial_id", "case_name", "verbose", "status", "ok", "error", "bundle_npz", "bundle_json",
        "timewindow"}  # timewindow == W_steps


def param_columns(P):
    cols = []
    for c in P.columns:
        if c in SKIP or c.startswith(("n_members", "success", "mean_", "median_", "std_", "min_", "max_", "pct_",
                                       "density", "most_", "elapsed", "final_", "north_", "south_", "branch_",
                                       "n_steps", "fraction")):
            continue
        if c in ("starting_lat", "starting_lon", "child_of_interest", "timeres", "spaceres"):
            continue
        if P[c].astype(str).nunique() > 1:
            cols.append(c)
    return cols


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allrows, percfg = [], []
    for setname in ["pnw_A1", "debby_A2", "pinatubo_A3"]:
        x = pd.read_csv(R1 / f"analysis_v2/A4_twins/{setname}_long.csv.gz", keep_default_na=False)
        x = x[(x["twin"] != "") & x["lead"].astype(str).str.startswith("w")].copy()
        x["diff"] = x["share_real"].astype(float) - x["share_twin"].astype(float)
        pc = x.groupby(["var", "lead", "cfg"])["diff"].mean().reset_index()
        P = derive(cfg_params(setname), NV[setname])
        cols = param_columns(P)
        print(setname, "varying params:", cols, flush=True)
        M = pc.merge(P[["cfg"] + cols], on="cfg", how="left")
        M.insert(0, "set", setname)
        percfg.append(M)
        for (var, lead), g in M.groupby(["var", "lead"]):
            for c in cols:
                v = g[c]
                num = pd.api.types.is_numeric_dtype(v) and v.nunique() > 6
                if num:
                    b = pd.qcut(v.astype(float), 4, duplicates="drop")
                    labels = b.astype(str)
                else:
                    labels = v.astype(str)
                for lev, gg in g.groupby(labels):
                    m, lo, hi = A.boot_ci(gg["diff"].values)
                    allrows.append(dict(set=setname, var=var, window=lead, param=c,
                                        binning="quartile" if num else "level", bin=lev, n=len(gg), mean_diff=m,
                                        lo=lo, hi=hi, ci_gt0=bool(lo > 0), ci_lt0=bool(hi < 0)))
    S = pd.DataFrame(allrows)
    S.to_csv(OUT / "task2_param_bins.csv", index=False)
    pd.concat(percfg).to_csv(OUT / "task2_per_config.csv.gz", index=False)
    # compact: per set/var/window/param -> range of bin means (max-min) and bins with CI>0
    comp = []
    for k, g in S.groupby(["set", "var", "window", "param"]):
        comp.append(dict(zip(["set", "var", "window", "param"], k), n_bins=len(g),
                         min_bin_mean=g.mean_diff.min(), max_bin_mean=g.mean_diff.max(),
                         spread=g.mean_diff.max() - g.mean_diff.min(),
                         bins_ci_gt0=";".join(g.loc[g.ci_gt0, "bin"].astype(str)),
                         n_bins_ci_gt0=int(g.ci_gt0.sum()), n_bins_ci_lt0=int(g.ci_lt0.sum())))
    Cp = pd.DataFrame(comp)
    Cp.to_csv(OUT / "task2_param_compact.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(Cp.sort_values(["set", "var", "window", "spread"], ascending=[True, True, True, False]).round(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
