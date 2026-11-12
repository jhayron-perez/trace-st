"""Night-3 N1/N6 test bed + N2 nulls: one table per variant with the pre-declared metrics.

Reads revision1/night3/{bed,bedM10,n2}/<variant>/<task>/ (partial outputs allowed) and writes
revision1/night3/analysis/:
  bed_syn.csv      synthetic guardrails per variant (T1, T2-EN, T7 decoy, T7 p8 mixed)
  bed_twins.csv    twin diffs per variant x case x variable x window (selection years; Pinatubo SO2 twin)
  bed_twins_base_matched.csv  baseline restricted to the same configs and members as each expensive variant
  n2_nulls.csv     N2 null statistics (PNW, Debby; baseline method)
  N5_table.md      compact synthesis

Shares: same definitions as A4_twins.py (box-cell density per parent variable, step 0 excluded,
Debby hourly steps summed to lead days; windows PNW 1-10/11-20/21-30 d, Debby 1-4/5-15 d,
Pinatubo 1-3/4-16). diff = share(real) - share(twin), year-averaged per config, bootstrap CI over configs.
Configs: baseline-retained sets (PNW T6r 233, Debby rebuilt 75, Pinatubo T5r) restricted to those present.

python N3_bed_summary.py [--nproc 32]   (PBS only)
"""
from __future__ import annotations

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
from A4_twins import boot_ci, retained_ids, PNW6, PNW_STEM, WINDOWS  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = C.R1
N3 = R1 / "night3"
OUT = N3 / "analysis"
P4 = C.VARS["pinatubo"]


# ------------------------------------------------------------------ densities with a member cap
def dens_pnw(args):
    run_dir, n_vars, mmax = args
    p = json.load(open(Path(run_dir) / "params.json"))
    res = pickle.load(open(Path(run_dir) / "ensemble_results.pkl", "rb"))
    if mmax:
        res = res[:mmax]
    D = np.zeros((31, n_vars)); ok = 0
    for r in res:
        if not r.get("success", True):
            continue
        c = np.asarray(r["centers"], float); pa = np.asarray(r["parents"], int)
        n = min(len(c), len(pa), 31)
        if n == 0:
            continue
        ok += 1
        cnt = C._box_cell_counts_wrap(c[:n], C.PNW_LONS, C.PNW_LATS, float(p["box_size"]))
        for k in range(1, n_vars + 1):
            m = pa[:n] == k
            if m.any():
                D[np.nonzero(m)[0], k - 1] += cnt[m]
    return int(Path(run_dir).name.split("_")[1]), D, ok / max(len(res), 1)


def dens_bundle(args):
    npz, box, n_vars, hourly, mmax = args
    b = np.load(npz, allow_pickle=True)
    xs = np.asarray(b["xs"], float); ys = np.asarray(b["ys"], float)
    Cc, P, L, ok = b["centers"], b["parents"], b["lengths"], b["ok"]
    nm = len(L) if not mmax else min(len(L), mmax)
    D = np.zeros((Cc.shape[1], n_vars)); nok = 0
    for i in range(nm):
        n = int(L[i])
        if int(ok[i]) != 1 or n <= 0:
            continue
        nok += 1
        cnt = C._box_cell_counts(Cc[i, :n], xs, ys, float(box))
        par = np.asarray(P[i, :n], int)
        for q in range(1, n_vars + 1):
            m = par == q
            if m.any():
                D[np.nonzero(m)[0], q - 1] += cnt[m]
    if hourly:
        nday = (D.shape[0] - 1) // 24
        Dd = np.zeros((nday + 1, n_vars))
        for k in range(1, nday + 1):
            Dd[k] = D[24 * (k - 1) + 1: 24 * k + 1].sum(0)
        D = Dd
    return D, nok / max(nm, 1)


def densities(kind, d, n_vars, ids, nproc, mmax):
    if not d.exists():
        return {}, {}
    if kind == "pnw":
        dirs = [d / f"run_{i:04d}" for i in ids if (d / f"run_{i:04d}" / "ensemble_results.pkl").exists()]
        with Pool(nproc) as pool:
            out = pool.map(dens_pnw, [(str(x), n_vars, mmax) for x in dirs])
        return {i: D for i, D, _ in out}, {i: s for i, _, s in out}
    f = d / "results_jobs_final.csv"
    if not f.exists():
        f = d / "results_jobs_snapshot.csv"
    if not f.exists():
        return {}, {}
    j = pd.read_csv(f)
    j = j[j["trial_id"].isin(ids) & j["bundle_npz"].notna()]
    with Pool(nproc) as pool:
        out = pool.map(dens_bundle, [(r.bundle_npz, r.box_size, n_vars, kind == "debby", mmax) for r in j.itertuples()])
    return ({int(t): D for t, (D, _) in zip(j["trial_id"], out)},
            {int(t): s for t, (_, s) in zip(j["trial_id"], out)})


def window_shares(D, kind):
    """{window label: share vector} from per-lead densities (step 0 excluded)."""
    out = {}
    for (w0, w1) in WINDOWS[kind]:
        Dw = D[w0: w1 + 1].sum(0)
        if Dw.sum() > 0:
            out[f"w{w0}-{w1}"] = Dw / Dw.sum()
    return out


# ------------------------------------------------------------------ case definitions
def bed_runs(kind):
    """[(task, vars, (real, twin), year)] for the test bed (selection years)."""
    if kind == "pnw":
        return [(f"pnw_{s}_{y}", PNW6 + [f"{PNW_STEM[s]}_P{y}"], (PNW_STEM[s], f"{PNW_STEM[s]}_P{y}"), y)
                for s in ("z500", "tcwv", "olr", "slhf") for y in (2011, 2018)]
    if kind == "debby":
        return [(f"debby_vo_{y}", ["Tb", "vo", "precip", f"VO_P{y}"], ("vo", f"VO_P{y}"), y) for y in (2004, 2007)]
    return [("Pin", P4 + ["TMSO203"], ("TMSO201", "TMSO203"), "so2twin")]


def twin_rows(root, variant, kind, ids, nproc, mmax, tag):
    rows = []
    for task, vars_, (real, twin), year in bed_runs(kind):
        Dd, succ = densities(kind, root / variant / task, len(vars_), ids, nproc, mmax)
        for cfg, D in Dd.items():
            for w, sh in window_shares(D, kind).items():
                rows.append(dict(variant=variant, tag=tag, case=kind, task=task, year=year, cfg=cfg, window=w,
                                 var=real, share_real=sh[vars_.index(real)], share_twin=sh[vars_.index(twin)],
                                 success=succ.get(cfg, np.nan)))
    return rows


def summarize_twins(long):
    if long.empty:
        return pd.DataFrame()
    long = long.copy(); long["diff"] = long["share_real"] - long["share_twin"]
    S = []
    for (v, tag, case, var, w), g in long.groupby(["variant", "tag", "case", "var", "window"]):
        pc = g.groupby("cfg")["diff"].mean()
        m, lo, hi = boot_ci(pc.values)
        S.append(dict(variant=v, tag=tag, case=case, var=var, window=w, n_cfg=int(pc.size),
                      n_years=int(g["year"].nunique()), real_mean=float(g["share_real"].mean()),
                      twin_mean=float(g["share_twin"].mean()), diff_mean=m, lo=lo, hi=hi,
                      frac_real_gt_twin=float((pc > 0).mean()), success_mean=float(g["success"].mean())))
    return pd.DataFrame(S)


# ------------------------------------------------------------------ synthetic guardrails
def load_first(d, names):
    for n in names:
        if (d / n).exists():
            return pd.read_csv(d / n)
    return None


def syn_metrics(d, variant):
    r = dict(variant=variant, root=d.parent.name)
    t = load_first(d / "T1", ["results_tracks_final.csv", "results_tracks_snapshot.csv"])
    if t is not None:
        t = t[t["method"] == "granger"]
        comp = retention.t1_retained(t)
        per = t.groupby("trial_id")["mean_distance"].mean()
        r.update(T1_n=int(per.size), T1_completion=float(comp.mean()), T1_track_err_median=float(per.median()))
    x = load_first(d / "T2en", ["results_alphas_final.csv", "results_alphas_snapshot.csv"])
    if x is not None and len(x):
        r["T2_n"] = int(x["trial_id"].nunique()) if "trial_id" in x else len(x)
        ok = x["estimated_alpha_density"].notna()
        r["T2_r"] = float(np.corrcoef(x.loc[ok, "alpha_mix"], x.loc[ok, "estimated_alpha_density"])[0, 1]) if ok.sum() > 2 else np.nan
        for a in (0.2, 0.6):
            g = x[(x["alpha_mix"] - a).abs() < 1e-9]
            if len(g):
                r[f"T2_abs_err_a{a}"] = float((g["estimated_alpha_density"] - g["alpha_mix"]).abs().median())
                if "is_valid_physical" in g:
                    r[f"T2_valid_a{a}"] = float(g["is_valid_physical"].mean())
    for task, pre in (("T7dec", "dec"), ("T7p8", "p8")):
        x = load_first(d / task, ["results_alphas_final.csv", "results_alphas_snapshot.csv"])
        if x is None or not len(x):
            continue
        r[f"{pre}_n"] = len(x)
        for c in ("t7_share_density_v1v3", "t7_share_density_decoy", "t7_share_density_noise",
                  "t7_share_density_moving", "t7_alpha_abs_err_density_13", "t7_completion_rate"):
            if c in x:
                r[f"{pre}_{c.replace('t7_', '').replace('share_density_', 'sh_')}"] = float(x[c].median())
    return r


def mcap(v, kind):
    """Members used: expensive variants ran M 10 (Debby hybrids and stab: M 5, night-3 rule 8 update)."""
    if v in ("hybv", "hybl", "stab") and kind == "debby":
        return 5
    return 10 if v in ("hybv", "hybl", "cv", "stab") else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--bed", default="bed", help="night3 subdir with the variant runs (dry = smoke test)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); bed = N3 / a.bed
    ids = {k: retained_ids(k) for k in ("pnw", "debby", "pinatubo")}
    EXP = ("hybv", "hybl", "cv", "stab")
    # --- synthetic
    syn = []
    for root in (bed, N3 / "bedM10"):
        if root.exists():
            for d in sorted(p for p in root.iterdir() if p.is_dir()):
                syn.append(syn_metrics(d, d.name))
    SYN = pd.DataFrame(syn); SYN.to_csv(out / "bed_syn.csv", index=False)
    print(SYN.round(3).to_string(), flush=True)
    # --- twins
    rows, rows_b = [], []
    variants = sorted(p.name for p in bed.iterdir() if p.is_dir())
    for v in variants:
        pass
        for kind in ("pnw", "debby", "pinatubo"):
            mm = mcap(v, kind); rows += twin_rows(bed, v, kind, ids[kind], a.nproc, mm, f"M{mm}" if mm else "M20")
    long = pd.DataFrame(rows)
    long.to_csv(out / "bed_twins_long.csv.gz", index=False)
    TW = summarize_twins(long); TW.to_csv(out / "bed_twins.csv", index=False)
    # baseline matched to each expensive variant (same configs present, members 0-9)
    if not long.empty and "base" in variants:
        for v in EXP:
            sub = long[long["variant"] == v]
            if sub.empty:
                continue
            for kind in sub["case"].unique():
                cfgs = sorted(sub.loc[sub["case"] == kind, "cfg"].unique())
                rows_b += [dict(r, variant=f"base_for_{v}") for r in
                           twin_rows(bed, "base", kind, cfgs, a.nproc, mcap(v, kind), f"M{mcap(v, kind)}")]
    TB = summarize_twins(pd.DataFrame(rows_b)); TB.to_csv(out / "bed_twins_base_matched.csv", index=False)
    show = pd.concat([TW, TB]) if len(TB) else TW
    if len(show):
        print(show[["variant", "tag", "case", "var", "window", "n_cfg", "real_mean", "twin_mean", "diff_mean", "lo",
                    "hi", "success_mean"]].round(3).to_string(), flush=True)
    # --- compact markdown
    md = ["# Night-3 test bed (auto-generated)\n", "## Synthetic\n", C.to_markdown(SYN) if len(SYN) else "(none)\n",
          "\n## Twins (selection years; Pinatubo SO2 twin)\n", C.to_markdown(show) if len(show) else "(none)\n"]
    (out / "N5_table.md").write_text("".join(md))


if __name__ == "__main__":
    main()
