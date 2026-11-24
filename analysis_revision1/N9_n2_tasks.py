"""Night-9 N2 (S11): choose 5 configs at random (default_rng(0)) from the harmonized-retained configs of the decoy run of a case
and write the task files night9/tasks/n2_<case>_d<dd>.txt (one line per (K, config); K = number of typical fields incl. the
decoy; K = 1 is the base decoy run and is not re-run). Draw dd of the extra fields: night9/extras/<case>/draw<dd>/x_<jj>.nc.
python N9_n2_tasks.py --case pnw|pinatubo|debby"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
KS = [3, 6, 9, 12, 15]
ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
a = ap.parse_args()
case = a.case
src = dict(pnw="night8/analysis/T1b_pnw_configs.csv", pinatubo="night8/analysis/T1b_pinatubo_configs.csv",
           debby="night9/analysis/N1_debby_configs.csv")[case]
ids = sorted(pd.read_csv(R1 / src)["id"].astype(int))
sel = sorted(int(i) for i in np.random.default_rng(0).choice(ids, 5, replace=False))
(R1 / "night9").mkdir(exist_ok=True)
json.dump(dict(case=case, source=src, n_retained=len(ids), seed=0, configs=sel), open(R1 / f"night9/n2_{case}_configs.json", "w"))
print(case, "retained", len(ids), "selected", sel)
pool = R1 / f"night{'9' if case == 'debby' else '8'}/pools/{case}"
ext = R1 / f"night9/extras/{case}"
for dd in range(3):
    lines = []
    for K in KS:
        for c in sel:
            tag = f"{case}_d{dd}_K{K:02d}_c{c:03d}"
            opt = f"--decoy-pool {pool} --decoy-extra {ext}|{dd}|{K - 1}"
            if case == "pnw":
                args = (f"--n-trials 300 --run-ids {c} --n-members 30 --nproc 30 --start-method fork --continue-if-exists "
                        f"--force-prob-rule linear --extra-vars t2m --extra-path {R1}/night5/inputs_pnw {opt}")
                lines.append(f"{case}_d{dd}/K{K:02d}_c{c:03d}|8_HW_PNW21_Runs_v2.py|{args}")
            elif case == "pinatubo":
                args = (f"--n-trials 1000 --trial-ids {c} --n-workers 30 --ensemble-size 30 --flush-every 1 --force-prob-rule linear "
                        f"--extra-vars TMSO203 BURDENSO403 T050 {opt}")
                lines.append(f"{case}_d{dd}/K{K:02d}_c{c:03d}|6_PinatuboRuns.py|{args}")
            else:
                args = (f"--n-trials 300 --trial-ids {c} --ensemble-size 30 --flush-every 1 --force-prob-rule linear --n-workers 24 "
                        f"--parallel-members --vo-input {R1}/inputs_debby/vo500700_2006_rebuilt.nc:vo --extra-inputs "
                        f"CAPE={R1}/inputs_debby/cape_2006.nc:cape DIV200={R1}/inputs_debby/d200_2006.nc:d MSLP={R1}/night5/inputs_debby/msl_2006.nc:msl {opt}")
                lines.append(f"{case}_d{dd}/K{K:02d}_c{c:03d}|5_DebbyRuns.py|{args}")
    # largest K first (longest tasks start first)
    lines = lines[::-1]
    (R1 / f"night9/tasks/n2_{case}_d{dd}.txt").write_text("\n".join(lines) + "\n")
