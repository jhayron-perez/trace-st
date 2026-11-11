"""Night-2 A4: one-at-a-time placebo twins, the same statistics for every case.

For each twin run, each config and each lead: the share of the trajectory density on the real
variable X and on its twin (same run, so both compete in the same system).
- Leads start at 1. Step 0 is the starting box, whose parent is always the child; it is excluded.
- Debby: hourly steps, summed in 24 h blocks to lead days.
- Statistics per variable and lead:
  - median and 5-95 % of the real and twin shares over configs x years;
  - bootstrap CI (10,000, seed 0) over configs of mean(X - twin), year-averaged per config;
  - fraction of configs with X > twin;
  - causal horizon = last lead of the initial run of leads whose CI lower bound is > 0
    (also: last lead with CI lower bound > 0 anywhere).
- Windows: PNW 1-10 / 11-20 / 21-30 d (Fig 6B); Debby 1-4 / 5-15 d; Pinatubo 1-3 / 4-16 d.

python A4_twins.py SETNAME [--nproc N]
  SETNAME in: pnw_8131602, pnw_A1, debby_A2, debby_T8, pinatubo_A3
Outputs: revision1/analysis_v2/A4_twins/<SETNAME>_{long.csv.gz,lead_summary.csv,window_summary.csv,horizon.csv}
Figures: revision1/figures_v2/A4_twins/<SETNAME>_*.png/pdf
Run through PBS only (reads the ensemble pickles / bundles).
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = C.R1
PNW6 = ["Z500", "Z10", "MTNLWRF", "TCWV", "MSLHF", "MSSHF"]
PNW_STEM = dict(z500="Z500", z10="Z10", olr="MTNLWRF", tcwv="TCWV", slhf="MSLHF", sshf="MSSHF")
A1_YEARS = [2006, 2011, 2012, 2018, 2020]
A2_YEARS = [2003, 2004, 2005, 2007, 2008]
NICE = dict(pnw_8131602="PNW 2021, Z500 and TCWV twins (2010)", pnw_A1="PNW 2021, one twin per variable (5 years)",
            debby_A2="Debby, vorticity twin (5 years)", debby_T8="Debby + CAPE, DIV200, vo twin 2005 (T8)", pinatubo_A3="Pinatubo, tag-03 twins")
WINDOWS = dict(pnw=[(1, 10), (11, 20), (21, 30)], debby=[(1, 4), (5, 15)], pinatubo=[(1, 3), (4, 16)])


# ------------------------------------------------------------------ per-config per-lead densities
def _pnw_one(args):
    run_dir, n_vars = args
    p = json.load(open(Path(run_dir) / "params.json"))
    D = C.step_density_pnw(run_dir, float(p["box_size"]), n_vars=n_vars, n_steps=31)
    return int(Path(run_dir).name.split("_")[1]), D          # leads = steps (1 d)


def _bundle_one(args):
    npz, box, n_vars, hourly = args
    D = C.step_density_bundle(npz, float(box), n_vars)
    if hourly:  # step s = s hours back; lead day k = steps 24(k-1)+1 .. 24k
        nday = (D.shape[0] - 1) // 24
        Dd = np.zeros((nday + 1, n_vars))
        for k in range(1, nday + 1):
            Dd[k] = D[24 * (k - 1) + 1: 24 * k + 1].sum(0)
        D = Dd
    return D


def densities_pnw(d, n_vars, ids, nproc):
    dirs = [d / f"run_{i:04d}" for i in ids if (d / f"run_{i:04d}" / "ensemble_results.pkl").exists()]
    with Pool(nproc) as pool:
        out = pool.map(_pnw_one, [(str(x), n_vars) for x in dirs])
    return {i: D for i, D in out}


def densities_bundles(d, n_vars, ids, nproc, hourly):
    j = pd.read_csv(d / "results_jobs_final.csv")
    j = j[j["trial_id"].isin(ids) & j["bundle_npz"].notna()]
    with Pool(nproc) as pool:
        out = pool.map(_bundle_one, [(r.bundle_npz, r.box_size, n_vars, hourly) for r in j.itertuples()])
    return {int(t): D for t, D in zip(j["trial_id"], out)}


# ------------------------------------------------------------------ set definitions
def set_defs(name):
    """List of runs: dict(dir, vars, pairs=[(real, twin)], year, kind)."""
    runs = []
    if name == "pnw_8131602":
        runs.append(dict(dir=R1 / "T8_pnw", vars=PNW6 + ["Z500_P2010", "TCWV_P2010"],
                         pairs=[("Z500", "Z500_P2010"), ("TCWV", "TCWV_P2010")], year=2010, kind="pnw"))
    elif name == "pnw_A1":
        for stem, nm in PNW_STEM.items():
            for y in A1_YEARS:
                runs.append(dict(dir=R1 / "A1_pnw_twin" / f"{stem}_{y}", vars=PNW6 + [f"{nm}_P{y}"],
                                 pairs=[(nm, f"{nm}_P{y}")], year=y, kind="pnw"))
    elif name == "debby_A2":
        for y in A2_YEARS:
            runs.append(dict(dir=R1 / "A2_debby_twin" / f"vo_{y}", vars=["Tb", "vo", "precip", f"VO_P{y}"],
                             pairs=[("vo", f"VO_P{y}")], year=y, kind="debby"))
    elif name == "debby_T8":   # night-1 T8 Debby: original vo + CAPE, DIV200 and the 2005 vo twin; original-vo retention
        runs.append(dict(dir=R1 / "T8_debby", vars=["Tb", "vo", "precip", "CAPE", "DIV200", "VO_P2005"],
                         pairs=[("vo", "VO_P2005")], year=2005, kind="debby", ids_from="T4r_debby"))
    elif name == "pinatubo_A3":
        P4 = C.VARS["pinatubo"]
        runs.append(dict(dir=R1 / "A3_pinatubo_twin" / "so2twin_TMSO203", vars=P4 + ["TMSO203"],
                         pairs=[("TMSO201", "TMSO203")], year="so2twin", kind="pinatubo"))
        runs.append(dict(dir=R1 / "A3_pinatubo_twin" / "so4twin_BURDENSO403", vars=P4 + ["BURDENSO403"],
                         pairs=[("BURDENSO401", "BURDENSO403")], year="so4twin", kind="pinatubo"))
        runs.append(dict(dir=R1 / "A3_pinatubo_twin" / "combined_TMSO203_BURDENSO403_FLUT",
                         vars=P4 + ["TMSO203", "BURDENSO403", "FLUT"],
                         pairs=[("TMSO201", "TMSO203"), ("BURDENSO401", "BURDENSO403")], year="combined",
                         kind="pinatubo"))
    else:
        raise SystemExit(f"unknown set {name}")
    return runs


def retained_ids(kind, ids_from=None):
    if ids_from is not None:   # Debby retention computed on another baseline run
        j = pd.read_csv(R1 / ids_from / "results_jobs_final.csv")
        return sorted(j.loc[retention.debby_retained(j), "trial_id"].astype(int))
    if kind == "pnw":
        return [int(x) for x in open(R1 / "T6r_pnw" / "retained_run_ids.txt").read().split()]
    if kind == "debby":   # primary = rebuilt-vorticity baseline (PI decision 1c)
        j = pd.read_csv(R1 / "T4r_debby_vorebuilt" / "results_jobs_final.csv")
        return sorted(j.loc[retention.debby_retained(j), "trial_id"].astype(int))
    j = pd.read_csv(R1 / "T5r_pinatubo" / "results_jobs_final.csv")
    return sorted(j.loc[retention.pinatubo_retained(j), "trial_id"].astype(int))


# ------------------------------------------------------------------ statistics
def boot_ci(x, n_boot=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def horizon(df):
    """df sorted by lead with column lo. Returns (contiguous horizon from lead 1, last lead with lo>0)."""
    ok = (df["lo"] > 0).to_numpy()
    leads = df["lead"].to_numpy()
    h1 = 0
    for L, o in zip(leads, ok):
        if o:
            h1 = int(L)
        else:
            break
    h2 = int(leads[ok].max()) if ok.any() else 0
    return h1, h2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--nproc", type=int, default=32)
    a = ap.parse_args()
    C.set_style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = set_defs(a.set)
    kind = runs[0]["kind"]
    ids = retained_ids(kind, runs[0].get("ids_from"))
    out = C.outdir("A4_twins")
    rows, avail = [], []
    for r in runs:
        if not r["dir"].exists():
            avail.append(dict(dir=str(r["dir"]), n_cfg=0)); continue
        nv = len(r["vars"])
        if kind == "pnw":
            Dd = densities_pnw(r["dir"], nv, ids, a.nproc)
        else:
            Dd = densities_bundles(r["dir"], nv, ids, a.nproc, hourly=(kind == "debby"))
        avail.append(dict(dir=str(r["dir"]), n_cfg=len(Dd)))
        for cfg, D in Dd.items():
            tot = D.sum(1)
            for L in range(1, D.shape[0]):
                if tot[L] <= 0:
                    continue
                sh = D[L] / tot[L]
                base = dict(set=a.set, year=r["year"], cfg=cfg, lead=L)
                for real, twin in r["pairs"]:
                    rows.append(dict(base, var=real, twin=twin, share_real=sh[r["vars"].index(real)],
                                     share_twin=sh[r["vars"].index(twin)], density_total=tot[L]))
                for v in r["vars"]:  # all shares (for FLUT and context)
                    rows.append(dict(base, var=v, twin="", share_real=sh[r["vars"].index(v)], share_twin=np.nan,
                                     density_total=tot[L]))
            # windows (shares of the window-integrated density)
            for (w0, w1) in WINDOWS[kind]:
                Dw = D[w0: w1 + 1].sum(0)
                if Dw.sum() <= 0:
                    continue
                sh = Dw / Dw.sum()
                for real, twin in r["pairs"]:
                    rows.append(dict(set=a.set, year=r["year"], cfg=cfg, lead=f"w{w0}-{w1}", var=real, twin=twin,
                                     share_real=sh[r["vars"].index(real)], share_twin=sh[r["vars"].index(twin)],
                                     density_total=Dw.sum()))
                for v in r["vars"]:
                    rows.append(dict(set=a.set, year=r["year"], cfg=cfg, lead=f"w{w0}-{w1}", var=v, twin="",
                                     share_real=sh[r["vars"].index(v)], share_twin=np.nan, density_total=Dw.sum()))
    long = pd.DataFrame(rows)
    long["lead"] = long["lead"].astype(str)
    long.to_csv(out / f"{a.set}_long.csv.gz", index=False)
    pd.DataFrame(avail).to_csv(out / f"{a.set}_availability.csv", index=False)
    print(pd.DataFrame(avail).to_string(), flush=True)

    pairs = long[long["twin"] != ""].copy()
    pairs["diff"] = pairs["share_real"] - pairs["share_twin"]
    summ = []
    for (var, lead), g in pairs.groupby(["var", "lead"]):
        pc = g.groupby("cfg")["diff"].mean()           # year-averaged per config
        m, lo, hi = boot_ci(pc.values)
        summ.append(dict(var=var, lead=lead, n_cfg=int(pc.size), n_years=int(g["year"].nunique()),
                         real_median=float(g["share_real"].median()), real_p05=float(g["share_real"].quantile(.05)),
                         real_p95=float(g["share_real"].quantile(.95)), twin_median=float(g["share_twin"].median()),
                         twin_p05=float(g["share_twin"].quantile(.05)), twin_p95=float(g["share_twin"].quantile(.95)),
                         real_mean=float(g["share_real"].mean()), twin_mean=float(g["share_twin"].mean()),
                         diff_mean=m, lo=lo, hi=hi, frac_real_gt_twin=float((pc > 0).mean())))
    S = pd.DataFrame(summ)
    win = S[S["lead"].str.startswith("w")].copy()
    lead = S[~S["lead"].str.startswith("w")].copy()
    lead["lead"] = lead["lead"].astype(int)
    lead = lead.sort_values(["var", "lead"])
    lead.to_csv(out / f"{a.set}_lead_summary.csv", index=False)
    win.to_csv(out / f"{a.set}_window_summary.csv", index=False)
    hz = []
    for var, g in lead.groupby("var"):
        h1, h2 = horizon(g)
        hz.append(dict(var=var, horizon_contiguous=h1, last_lead_ci_above_0=h2, n_cfg=int(g["n_cfg"].max())))
    HZ = pd.DataFrame(hz)
    HZ.to_csv(out / f"{a.set}_horizon.csv", index=False)
    print(HZ.to_string(), flush=True)
    print(win.round(3).to_string(), flush=True)

    unit = "h-days" if kind == "debby" else "days"
    lab = lambda v: C.LABELS.get(v, v)
    col = lambda v: C.VAR_COLORS.get(v, "#000000")
    # (1) lead-dependent significance panel
    fig, ax = plt.subplots(figsize=(14, 7.875))
    for var, g in lead.groupby("var"):
        ax.plot(g["lead"], g["diff_mean"], "-o", color=col(var), ms=4, lw=2, label=lab(var))
        ax.fill_between(g["lead"], g["lo"], g["hi"], color=col(var), alpha=0.2, lw=0)
    ax.axhline(0, color="#999999", lw=1)
    ax.set_xlabel("Lead before the event (days)")
    ax.set_ylabel("Share(X) − share(twin of X)\n(mean over configs, 95 % CI)")
    ax.legend(ncol=3, frameon=False)
    ax.set_title(f"{NICE[a.set]}: real minus twin")
    C.savefig(fig, "A4_twins", f"{a.set}_lead_significance"); plt.close(fig)
    # (2) real vs twin share profiles per variable
    vs = sorted(lead["var"].unique())
    nc = min(3, len(vs)); nr = int(np.ceil(len(vs) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(16, 4.5 * nr + 0.5), squeeze=False, sharey=True)
    for k, var in enumerate(vs):
        axx = axs[k // nc][k % nc]; g = lead[lead["var"] == var]
        axx.plot(g["lead"], g["real_median"], "-", color=col(var), lw=2.5, label="real (median)")
        axx.fill_between(g["lead"], g["real_p05"], g["real_p95"], color=col(var), alpha=0.15, lw=0)
        axx.plot(g["lead"], g["twin_median"], "--", color="#555555", lw=2.5, label="twin (median)")
        axx.fill_between(g["lead"], g["twin_p05"], g["twin_p95"], color="#999999", alpha=0.15, lw=0)
        axx.set_title(lab(var)); axx.set_xlabel("Lead (days)")
        if k % nc == 0:
            axx.set_ylabel("Share of density (5–95 %)")
        if k == 0:
            axx.legend(frameon=False)
    for k in range(len(vs), nr * nc):
        axs[k // nc][k % nc].axis("off")
    fig.tight_layout()
    C.savefig(fig, "A4_twins", f"{a.set}_lag_profiles"); plt.close(fig)
    # (3) window bars real vs twin (Fig 6B style)
    ws = sorted(win["lead"].unique(), key=lambda s: int(s[1:].split("-")[0]))
    fig, axs = plt.subplots(1, len(ws), figsize=(16, 6), sharey=True, squeeze=False)
    for k, w in enumerate(ws):
        axx = axs[0][k]; g = win[win["lead"] == w].sort_values("var")
        x = np.arange(len(g))
        axx.bar(x - 0.2, g["real_mean"], 0.4, color=[col(v) for v in g["var"]], label="real")
        axx.bar(x + 0.2, g["twin_mean"], 0.4, color="#BBBBBB", edgecolor="#555555", hatch="//", label="twin")
        for xi, (_, rr) in zip(x, g.iterrows()):
            if rr["lo"] > 0:
                axx.text(xi, max(rr["real_mean"], rr["twin_mean"]) + 0.01, "*", ha="center", fontsize=20)
        axx.set_xticks(x); axx.set_xticklabels([lab(v) for v in g["var"]], rotation=30)
        axx.set_title(f"{w[1:]} d")
        if k == 0:
            axx.set_ylabel("Mean share of window density"); axx.legend(frameon=False)
    fig.suptitle(f"{NICE[a.set]}: real vs twin (* = 95 % CI of real − twin > 0)")
    fig.tight_layout()
    C.savefig(fig, "A4_twins", f"{a.set}_window_bars"); plt.close(fig)


if __name__ == "__main__":
    main()
