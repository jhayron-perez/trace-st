"""Night-9 (A0 basis): per-config window shares as a share of ALL candidates (real variables + decoy; rows sum to 1), for the
decoy runs of the three cases (all harmonized-retained configs, no criterion) + the rule flags.
Uses N9_figs456.case_data_A with TST_A0=1 (no config removed). Output night9/analysis/A0_<case>_allcand_configs.csv:
id, decoy_share, R1_drop, R2_drop, s_<a>-<b>_<var> (var incl. DECOY).  Used by N9_n5_filterfig.py (S20). PBS/Casper only."""
import os
os.environ["TST_A0"] = "1"
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import N9_figs456 as G  # noqa: E402

A8, A9 = G.R1 / "night8/analysis", G.R1 / "night9/analysis"
G.N5T2.WINDOWS = G.N6T2.WINDOWS
for case in ("pnw", "pinatubo", "debby"):
    D = G.case_data_A(case, 32)
    real = D["vars"] + ["DECOY"]
    rows = []
    for i in D["ids"]:
        row = dict(id=i)
        for w, s in G.N5T2.shares_from_D(D["D_all"][i], case).items():
            for j, v in enumerate(real):
                row[f"s_{w[1:]}_{v}"] = s[j]
        rows.append(row)
    T = pd.DataFrame(rows).set_index("id")
    f = pd.read_csv(A8 / f"T1b_{case}_configs.csv" if case != "debby" else A9 / "N1_debby_configs.csv").set_index("id")
    T = f[["decoy_share", "R1_drop", "R2_drop"]].join(T, how="inner")
    T.to_csv(A9 / f"A0_{case}_allcand_configs.csv")
    print(case, len(T), "configs; row-sum check", float(T[[c for c in T.columns if c.startswith("s_11-30") or c.startswith("s_4-16") or c.startswith("s_5-15")][:len(real)]].sum(1).mean()) if len(T) else "", flush=True)
