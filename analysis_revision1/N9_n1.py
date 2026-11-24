"""Night-9 N1: Debby final set + typical-field decoy pool (night9/n1/deb_p0..7, paper draws 0..299, M 30, linear, harmonized retention).
Per retained config: pooled decoy share (all leads, step 0 excluded), R1 not-rank-1 / R2 equal-share (1/p, p = 7) flags, shares by
window (1-4, 5-15; renormalized among the real variables), track distance. Outputs night9/analysis/N1_debby_configs.csv (retained
configs; same columns as T1b_*_configs.csv), N1_debby_all.csv (all configs, incl. not retained), N1_debby.md/_check{1,2,3}.csv.
python N9_n1.py --nproc 32"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N8_T1c as C  # noqa: E402

R1 = C.R1
OUT = R1 / "night9" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--nproc", type=int, default=32)
a = ap.parse_args()
parts = sorted((R1 / "night9/n1").glob("deb_q[0-9]"))
R = C.case_debby(a.nproc, parts=parts)
# paper-range reference = night-6 final-set ensemble (no decoy); same loader as T1c
Rr = C.case_debby(a.nproc, parts=sorted((R1 / "night6/t2/debby_final").glob("part[0-9][0-9]")), nv=6)
Rr = Rr[Rr.retained]
# track distance per config (mean over members of track_dist_km_mean)
td = []
for p in parts:
    f = p / "results_members_final.csv"
    if not f.exists():
        f = p / "results_members_snapshot.csv"
    if f.exists():
        m = pd.read_csv(f)
        m = m[m["ok"].astype(bool)]
        td.append(m.groupby("trial_id")["track_dist_km_mean"].mean())
if td:
    R["track_dist_km"] = pd.concat(td).reindex(R.index)
R.to_csv(OUT / "N1_debby_all.csv")
Rk = R[R.retained].copy(); Rk["R1_drop"] = Rk.R1_drop.astype(bool); Rk["R2_drop"] = Rk.R2_drop.astype(bool)
Rk.to_csv(OUT / "N1_debby_configs.csv")
C.DRAWN["debby"] = 300
c1, c2, c3 = C.summarize("debby", R, {"night-6 final-set ensemble (no decoy)": Rr})
c1.to_csv(OUT / "N1_debby_check1.csv", index=False); c3.to_csv(OUT / "N1_debby_check3.csv", index=False)
p = len(C.REAL["debby"]) + 1
ds = Rk["decoy_share"]
dist = dict(n_run=len(R), n_retained=len(Rk), R1_removed=int(Rk.R1_drop.sum()), R2_removed=int(Rk.R2_drop.sum()),
            R1_removed_frac=float(Rk.R1_drop.mean()), R2_removed_frac=float(Rk.R2_drop.mean()), decoy_share_mean=float(ds.mean()),
            decoy_share_median=float(ds.median()), decoy_share_p05=float(ds.quantile(.05)), decoy_share_p95=float(ds.quantile(.95)),
            decoy_share_max=float(ds.max()), one_over_p=1.0 / p, median_over_1p=float(ds.median() * p))
if "track_dist_km" in Rk:
    dist.update(track_km_all=float(Rk.track_dist_km.mean()), track_km_R1_kept=float(Rk[~Rk.R1_drop].track_dist_km.mean()),
                track_km_R2_kept=float(Rk[~Rk.R2_drop].track_dist_km.mean()))
pd.Series(dist).to_csv(OUT / "N1_debby_decoy_distribution.csv", header=["value"])
with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format, "display.max_rows", 400):
    txt = ["# N1 Debby with the typical-field decoy pool (M 30, linear, harmonized retention)", "```", pd.Series(dist).to_string(), "```",
           "\n## Survival (of the 300 paper draws)\n```", c1.to_string(index=False), "```",
           "\n## Headline windows: shares and rank-1 fractions (95 % CIs; decoy runs unfiltered / not-rank-1 kept / equal-share kept vs night-6 final-set ensemble)\n```",
           c3.to_string(index=False), "```"]
(OUT / "N1_debby.md").write_text("\n".join(txt))
print("\n".join(txt[:4]))
