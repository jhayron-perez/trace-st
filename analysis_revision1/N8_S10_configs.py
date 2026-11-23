"""Night-8 S10 pre-declaration: 'best' 5 (smallest mean |alpha error| over alpha 0.2/0.6 in night5/t3/p03) and 'random' 5
(numpy default_rng(0).choice over the sorted 50 trial ids, replace=False). -> night8/analysis/S10_configs.{csv,md}"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_T3 as T3  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
OUT = R1 / "night8" / "analysis"
d, x = T3.load(3)
pc = T3.per_config(3, x)
e = pc.groupby("trial_id")["alpha_err"].agg(["mean", "count"]).rename(columns={"mean": "mean_alpha_err"})
assert (e["count"] == 2).all() and len(e) == 50, e["count"].value_counts()
e = e.sort_values(["mean_alpha_err", ], kind="stable")
best = [int(i) for i in e.index[:5]]
ids = sorted(int(i) for i in e.index)
rnd = [int(i) for i in np.random.default_rng(0).choice(ids, size=5, replace=False)]
e["rank"] = np.arange(1, 51)
e["best"] = e.index.isin(best); e["random"] = e.index.isin(rnd)
e["selected"] = e["best"] | e["random"]
OUT.mkdir(parents=True, exist_ok=True)
e.to_csv(OUT / "S10_configs.csv")
ov = sorted(set(best) & set(rnd))
txt = ["# S10 configs (pre-declared)", f"source: {d}/results_alphas_final.csv; mean |alpha error| over alpha 0.2, 0.6; 50 configs",
       f"best 5 (trial ids, ascending error): {best}", f"random 5 (default_rng(0).choice(sorted ids, 5, replace=False)): {rnd}",
       f"overlap: {ov}; union: {sorted(set(best) | set(rnd))} ({len(set(best) | set(rnd))} configs)", "```", e.head(12).to_string(), "```"]
(OUT / "S10_configs.md").write_text("\n".join(txt)); print("\n".join(txt))
