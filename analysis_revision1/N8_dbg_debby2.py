import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_fraction_full_length import load_script
R = "/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1"
m = load_script("5_DebbyRuns.py"); m.use_union_search_space()
ex = [("CAPE", f"{R}/inputs_debby/cape_2006.nc", "cape"), ("DIV200", f"{R}/inputs_debby/d200_2006.nc", "d"), ("MSLP", f"{R}/night5/inputs_debby/msl_2006.nc", "msl")]
vo = (f"{R}/inputs_debby/vo500700_2006_rebuilt.nc", "vo")
dec = ("DECOY", f"{R}/night8/pools/debby/decoy_00.nc", "DECOY")
jobs = m.build_debby_jobs(40, seed=2026112313)
for label, extra in (("nodecoy", ex), ("decoy", ex + [dec])):
    case = m.build_case_debby(extra_inputs=extra, vo_input=vo)
    fd = case["full_data"]
    for tid in (0, 1, 2, 3):
        p = m.build_params_from_trial_debby(jobs[tid][1], case)
        p["starting_lat"] = float(case["event_lat"]); p["starting_lon"] = float(case["event_lon_0360"]); p["seed"] = 1
        t0 = time.time()
        try:
            c, pa, out = m.run_one_track(fd, p, case["date_end"])
            print(label, tid, "OK len", len(c), "sec", round(time.time() - t0), "lam", p["cd_kwargs"]["lambda_a"], "l1", p["cd_kwargs"]["l1_ratio"], "box", p["box_size"], "r", p["radius"], "tw", p["timewindow"], flush=True)
        except Exception as e:
            print(label, tid, "FAIL", str(e)[:80], "sec", round(time.time() - t0), "box", p["box_size"], "r", p["radius"], "tw", p["timewindow"], "start", p["starting_lat"], p["starting_lon"], flush=True)
