"""Night-8 T1c amendment: which wide-range (10^[-3,0] x 10^[-4,0]) configs fall inside the union range
(lambda_a >= 10^-2.5, l1_ratio >= 0.1; upper bounds equal). PNW/Pinatubo draws regenerated with the run scripts
(--search-space common, seeds 2026112301/02); Pinatubo: done = trial ids in the pieces' results_jobs.jsonl.
Writes night8/analysis/T1c_range_*.csv/json and night8/tasks/t1c2_pin_missing.txt (missing in-range ids)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_fraction_full_length import load_script  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
OUT = R1 / "night8/analysis"
LAM_LO, L1_LO = -2.5, -1.0


def inr(t):
    return (np.log10(t["lambda_a"]) >= LAM_LO - 1e-12) & (np.log10(t["l1_ratio"]) >= L1_LO - 1e-12)


res = {}
m = load_script("8_HW_PNW21_Runs_v2.py"); m.use_common_search_space()
t = pd.DataFrame([dict(id=i, **tr) for i, tr in m.build_pnw_jobs(300, seed=2026112301)])
t["in_range"] = inr(t)
t.to_csv(OUT / "T1c_range_pnw.csv", index=False)
res["pnw"] = dict(drawn=300, in_range=int(t.in_range.sum()))
m = load_script("6_PinatuboRuns.py"); m.use_common_search_space()
t = pd.DataFrame([dict(id=i, **tr) for i, tr in m.build_pinatubo_jobs(300, seed=2026112302)])
t["in_range"] = inr(t)
done = set()
for p in sorted((R1 / "night8/t1c").glob("pin_p[0-9]")):
    for line in open(p / "results_jobs.jsonl"):
        line = line.strip()
        if line:
            done.add(int(json.loads(line)["trial_id"]))
t["done"] = t["id"].isin(done)
t.to_csv(OUT / "T1c_range_pinatubo.csv", index=False)
miss = t[t.in_range & ~t.done]
res["pinatubo"] = dict(drawn=300, done=int(t.done.sum()), in_range=int(t.in_range.sum()), in_range_done=int((t.in_range & t.done).sum()),
                       in_range_missing=int(len(miss)), out_range_done=int((~t.in_range & t.done).sum()), out_range_missing=int((~t.in_range & ~t.done).sum()),
                       missing_ids=[int(i) for i in miss.id])
(OUT / "T1c_range.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: {a: b for a, b in v.items() if a != "missing_ids"} for k, v in res.items()}, indent=1))
print("missing in-range pinatubo ids:", res["pinatubo"]["missing_ids"])
