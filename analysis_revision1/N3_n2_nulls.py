"""Night-3 N2: fairer nulls for "this field at this place and time drives the event" (baseline method).

Add design (like A1/A2): the original variables + one null copy of X (time-shifted within the event
year, or longitude-shifted). Statistic per config and window: share(X) - share(null), averaged over the
nulls of one family (time shifts, space shifts), bootstrap CI over configs, frac X > null.
Replace design: X replaced by its twin (p unchanged). Statistic: share(X) in the baseline run
(T6r_pnw / T4r_debby_vorebuilt, same configs) minus share(twin in its X slot) in the swap run,
year-averaged per config; plus a rank p-value per lead window: rank of the baseline share(X) among
the swap-run shares of all nulls for the same config (p = (1 + #null >= X) / (1 + n_null)), averaged.
Lead-dependent version (per lead) and causal horizon as in A4_twins.

python N3_n2_nulls.py [--nproc 32]   (PBS only). Outputs: revision1/night3/analysis/n2_*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from A4_twins import boot_ci, horizon, retained_ids, PNW6, PNW_STEM  # noqa: E402
from N3_bed_summary import densities, window_shares  # noqa: E402

R1 = C.R1
N2 = R1 / "night3" / "n2" / "base"
OUT = R1 / "night3" / "analysis"
YEARS = (2006, 2011, 2012, 2018, 2020)
DEBBY3 = ["Tb", "vo", "precip"]


def runs():
    """(case, family, run_dir, vars, real, null_name, design)"""
    out = []
    for s in ("z500", "tcwv", "olr", "slhf"):
        nm = PNW_STEM[s]
        for k in (-30, -15, 15, 30):
            out.append(("pnw", "time", N2 / f"pnw_{s}_S{k}", PNW6 + [f"{nm}_P2021S{k:+d}"], nm, f"{nm}_P2021S{k:+d}", "add"))
        out.append(("pnw", "space", N2 / f"pnw_{s}_L20", PNW6 + [f"{nm}_P2021L+20"], nm, f"{nm}_P2021L+20", "add"))
    for s in ("tcwv", "olr", "slhf"):
        nm = PNW_STEM[s]
        for y in YEARS:
            v = [f"{nm}_P{y}" if x == nm else x for x in PNW6]
            out.append(("pnw", "replace_year", N2 / f"pnw_{s}_rep{y}", v, nm, f"{nm}_P{y}", "replace"))
    for n, fam in (("SM10", "time"), ("SM5", "time"), ("SP5", "time"), ("SP10", "time"), ("L10", "space"), ("L20", "space")):
        out.append(("debby", fam, N2 / f"debby_add_{n}", DEBBY3 + [f"VO_{n}"], "vo", f"VO_{n}", "add"))
    for y in (2003, 2004, 2005, 2007, 2008):
        out.append(("debby", "replace_year", N2 / f"debby_rep{y}", DEBBY3, "vo", "vo", "replace"))
    for n in ("SM10", "SP10"):
        out.append(("debby", "replace_time", N2 / f"debby_rep{n}", DEBBY3, "vo", "vo", "replace"))
    return out


def per_lead_shares(D):
    tot = D.sum(1)
    return {L: D[L] / tot[L] for L in range(1, D.shape[0]) if tot[L] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=32)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = {k: retained_ids(k) for k in ("pnw", "debby")}
    base_dir = dict(pnw=R1 / "T6r_pnw", debby=R1 / "T4r_debby_vorebuilt")
    base_vars = dict(pnw=PNW6, debby=DEBBY3)
    # baseline shares per config (windows and leads)
    base = {}
    for kind in ("pnw", "debby"):
        Dd, _ = densities(kind, base_dir[kind], len(base_vars[kind]), ids[kind], a.nproc, 0)
        base[kind] = Dd
    rows = []
    for case, fam, d, vars_, real, null, design in runs():
        Dd, succ = densities(case, d, len(vars_), ids[case], a.nproc, 0)
        for cfg, D in Dd.items():
            if design == "add":
                ws = window_shares(D, case); ls = per_lead_shares(D)
                for w, sh in list(ws.items()) + [(str(L), sh) for L, sh in ls.items()]:
                    rows.append(dict(case=case, family=fam, design=design, run=d.name, cfg=cfg, window=w, var=real,
                                     share_x=sh[vars_.index(real)], share_null=sh[vars_.index(null)]))
            else:
                if cfg not in base[case]:
                    continue
                Db = base[case][cfg]
                wb, lb = window_shares(Db, case), per_lead_shares(Db)
                ws, ls = window_shares(D, case), per_lead_shares(D)
                slot = vars_.index(null)
                for w in ws:
                    if w in wb:
                        rows.append(dict(case=case, family=fam, design=design, run=d.name, cfg=cfg, window=w, var=real,
                                         share_x=wb[w][base_vars[case].index(real)], share_null=ws[w][slot]))
                for L in ls:
                    if L in lb:
                        rows.append(dict(case=case, family=fam, design=design, run=d.name, cfg=cfg, window=str(L),
                                         var=real, share_x=lb[L][base_vars[case].index(real)], share_null=ls[L][slot]))
    long = pd.DataFrame(rows)
    long.to_csv(OUT / "n2_long.csv.gz", index=False)
    if long.empty:
        print("no N2 outputs yet"); return
    long["diff"] = long["share_x"] - long["share_null"]
    S = []
    for (case, var, fam, w), g in long.groupby(["case", "var", "family", "window"]):
        pc = g.groupby("cfg")["diff"].mean()
        m, lo, hi = boot_ci(pc.values)
        # rank p-value: X (per config, mean over runs) vs each null run's share
        pr = []
        for cfg, gg in g.groupby("cfg"):
            x = gg["share_x"].mean(); nulls = gg["share_null"].to_numpy()
            pr.append((1 + np.sum(nulls >= x)) / (1 + nulls.size))
        S.append(dict(case=case, var=var, family=fam, window=w, n_cfg=int(pc.size), n_runs=int(g["run"].nunique()),
                      x_mean=float(g["share_x"].mean()), null_mean=float(g["share_null"].mean()), diff_mean=m, lo=lo,
                      hi=hi, frac_x_gt_null=float((pc > 0).mean()), rank_p_mean=float(np.mean(pr))))
    S = pd.DataFrame(S)
    win = S[S["window"].str.startswith("w")]
    lead = S[~S["window"].str.startswith("w")].copy(); lead["lead"] = lead["window"].astype(int)
    win.to_csv(OUT / "n2_window_summary.csv", index=False)
    lead.sort_values(["case", "var", "family", "lead"]).to_csv(OUT / "n2_lead_summary.csv", index=False)
    hz = []
    for (case, var, fam), g in lead.groupby(["case", "var", "family"]):
        h1, h2 = horizon(g.sort_values("lead"))
        hz.append(dict(case=case, var=var, family=fam, horizon_contiguous=h1, last_lead_ci_above_0=h2))
    pd.DataFrame(hz).to_csv(OUT / "n2_horizon.csv", index=False)
    print(win.round(3).to_string(), "\n", pd.DataFrame(hz).to_string(), flush=True)


if __name__ == "__main__":
    main()
