"""Precompute per-step, per-variable integrated trajectory density for every configuration of the
real cases (all trials, not only retained), from the stored bundles / ensemble pickles.

Output: analysis_v2/I_significance/step_density_<case>.npz with arrays
  ids (n_cfg,), D (n_cfg, n_steps, n_vars), vars
and a validation CSV comparing recomputed window sums to the stored ones
(Debby density_range 96-360 h, Pinatubo density_range 4-16 d, PNW window_stats days_1_10/11_20/21_30).
Run through PBS (reads ~4 GB of pickles for PNW).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402

out = C.outdir("I_significance")
val_rows = []

for case in ["debby", "debby_orig", "pinatubo"]:
    j = C.load_jobs(case)
    names = C.VARS[case]
    nmax = 361 if case.startswith("debby") else 17
    ids, Ds = [], []
    for _, r in j.iterrows():
        D = C.step_density_bundle(r["bundle_npz"], float(r["box_size"]), len(names))
        Dp = np.zeros((nmax, len(names)))
        Dp[:D.shape[0]] = D
        ids.append(int(r["trial_id"])); Ds.append(Dp)
        a, b = int(r["density_range_start_step"]), int(r["density_range_end_step_exclusive"])
        w = C.window_sum(Dp, a, b)
        stored = np.array([r[f"density_range::{v}"] for v in names], float)
        val_rows.append(dict(case=case, cfg=int(r["trial_id"]), window=f"{a}-{b}",
                             max_abs_diff=float(np.max(np.abs(w - stored))),
                             max_rel_diff=float(np.max(np.abs(w - stored) / np.maximum(stored, 1)))))
    np.savez_compressed(out / f"step_density_{case}.npz", ids=np.array(ids), D=np.stack(Ds), vars=np.array(names))
    print(case, len(ids), flush=True)

for case in ["pnw", "pnw_wide"]:
    s = C.load_pnw_summary(case)
    names = C.VARS["pnw"]
    ids, Ds = [], []
    for _, r in s.iterrows():
        rid = int(r["run_id"])
        rd = C.RUNS[case] / f"run_{rid:04d}"
        import json
        box = float(json.load(open(rd / "params.json"))["box_size"])
        D = C.step_density_pnw(rd, box)
        ids.append(rid); Ds.append(D)
        for (a, b), lab in zip(((0, 10), (10, 20), (20, 30)), ("days_1_10", "days_11_20", "days_21_30")):
            w = C.window_sum(D, a, b)
            stored = np.array([r[f"{lab}_integrated_{v}"] for v in names], float)
            val_rows.append(dict(case=case, cfg=rid, window=lab, max_abs_diff=float(np.max(np.abs(w - stored))),
                                 max_rel_diff=float(np.max(np.abs(w - stored) / np.maximum(stored, 1)))))
    np.savez_compressed(out / f"step_density_{case}.npz", ids=np.array(ids), D=np.stack(Ds), vars=np.array(names))
    print(case, len(ids), flush=True)

v = pd.DataFrame(val_rows)
v.to_csv(out / "step_density_validation.csv", index=False)
print(v.groupby(["case", "window"])[["max_abs_diff", "max_rel_diff"]].max().to_string())
print("n cfg-windows with any mismatch > 0:", int((v.max_abs_diff > 0).sum()), "of", len(v))
