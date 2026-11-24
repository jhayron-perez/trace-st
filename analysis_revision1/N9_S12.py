"""Night-9 N3 (fork F2): S12 original vs final set per case, both framings, figures_v9/S12_original_vs_final/.
B = night-6 final-set runs (no decoy, no criterion): fig_s12 of N6_T2 on night6/analysis/T2_window_shares.csv.
A = final set + typical-field decoy (night8 T1b: PNW, Pinatubo; Debby = N1, --debby-a-dir when ready), configs kept by the
not-rank-1 rule; real shares renormalized among the real variables; 95 % bootstrap CI over kept configs (10,000, seed 0).
The original-set bars are the night-6 paired original-set rows (same in A and B).
python N9_S12.py [--debby-a DIR_CONFIGS_CSV]   (PBS develop)"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
import N6_T2 as T2  # noqa: E402
from A4_twins import boot_ci  # noqa: E402

R1 = T2.R1
OUT = R1 / "figures_v9" / "S12_original_vs_final"
OUT.mkdir(parents=True, exist_ok=True)
MAP = dict(OLR="MTNLWRF", SLHF="MSLHF", SSHF="MSSHF", T2m="T2M")   # T1b names -> figure keys
REAL = dict(pnw=["Z500", "Z10", "OLR", "TCWV", "SLHF", "SSHF", "T2m"],
            pinatubo=["TMSO201", "BURDENSO401", "TMH2SO401", "AEROD_v", "TMSO203", "BURDENSO403", "T050"],
            debby=["Tb", "vo", "precip", "CAPE", "DIV200", "MSLP"])
ROLES = {c: T2.CASES[c]["roles"] for c in T2.CASES}


def a_rows(case, csv, nofilter=False):
    d = pd.read_csv(csv).set_index("id")
    if "R1_drop" in d and not nofilter:
        d = d[~d["R1_drop"].astype(bool)]
    d = d.dropna(subset=[c for c in d.columns if c.startswith("s_")][:1])
    rows = []
    for (a, b) in T2.WINDOWS[case]:
        cols = [f"s_{a}-{b}_{v}" for v in REAL[case]]
        if not all(c in d for c in cols):
            continue
        S_ = d[cols].to_numpy(float); ok = np.isfinite(S_).all(1); S_ = S_[ok]
        top = np.argmax(S_, 1)
        for j, v in enumerate(REAL[case]):
            m, lo, hi = boot_ci(S_[:, j])
            vv = MAP.get(v, v)
            rows.append(dict(case=case, set="final", cfgset="paired", window=f"w{a}-{b}", var=vv, role=ROLES[case].get(vv),
                             n_cfg=int(ok.sum()), share=m, lo=lo, hi=hi, frac_rank1=float((top == j).mean())))
    return pd.DataFrame(rows)


def move(case, tag):
    src = R1 / "figures_v9" / "S12_original_vs_final"
    for ext in ("pdf", "png"):
        f = src / f"s12_{case}_original_vs_final.{ext}"
        if f.exists():
            os.replace(f, src / f"s12_{case}_original_vs_final_{tag}.{ext}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", action="store_true", help="A0: no configuration filtered by the decoy (harmonized retention only); PNW only, files *_A0")
    ap.add_argument("--debby-a", default=None, help="Debby N1 per-config csv (T1c-style columns s_<a>-<b>_<var>, R1_drop)")
    a = ap.parse_args()
    S.set_style()
    W = pd.read_csv(R1 / "night6" / "analysis" / "T2_window_shares.csv")
    root = R1 / "figures_v9"
    if a.a0:
        # A0: final-set bars = share of ALL candidates (real + decoy) of the decoy runs, all harmonized-retained configs (window csvs of Figs 4-6 A0)
        WF = dict(pnw="Fig6_pnw/fig6_pnw_final_A0_window_shares.csv", pinatubo="Fig5_pinatubo/fig5_pinatubo_final_A0_window_shares.csv",
                  debby="Fig4_debby/fig4_debby_final_A0_window_shares.csv")
        for case, rel in WF.items():
            A = pd.read_csv(root / rel)
            A = A[A["var"] != "DECOY"].copy(); A["set"] = "final"; A["cfgset"] = "paired"
            W2 = pd.concat([W[~((W["case"] == case) & (W["set"] == "final") & (W["cfgset"] == "paired"))], A], ignore_index=True)
            T2.fig_s12(case, W2, T2.CASES[case], root); move(case, "A0")
            print(case, "A0 configs:", int(A["n_cfg"].max()), flush=True)
        return
    for case in ("debby", "pinatubo", "pnw"):
        T2.fig_s12(case, W, T2.CASES[case], root); move(case, "B")
    srcs = dict(pnw=R1 / "night8/analysis/T1b_pnw_configs.csv", pinatubo=R1 / "night8/analysis/T1b_pinatubo_configs.csv")
    if a.debby_a:
        srcs["debby"] = Path(a.debby_a)
    for case, csv in srcs.items():
        A = a_rows(case, csv)
        W2 = pd.concat([W[~((W["case"] == case) & (W["set"] == "final") & (W["cfgset"] == "paired"))], A], ignore_index=True)
        T2.fig_s12(case, W2, T2.CASES[case], root); move(case, "A")
        print(case, "A configs kept:", int(A["n_cfg"].max()), flush=True)


if __name__ == "__main__":
    main()
