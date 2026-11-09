"""Blocking regression (task 2f): re-run 5 deterministic T1 configs with this code and
compare centers and parents with the original univariate_v0 outputs. Exact match required.

python -m tests.regress_t1 [ORIG_DIR] [N_WORKERS]
Exit code 0 = all match, 1 = mismatch.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from tests.test_fraction_full_length import load_script

PICK = [("granger", 0), ("granger", 500), ("pcmci", 0), ("pcmci", 500), ("dynotears", 0)]
ORIG = sys.argv[1] if len(sys.argv) > 1 else "/glade/work/jhayron/trace-st_data/original_submission/ResultsSynthetic/univariate_v0"


def _run(job):
    t1 = load_script("1_SyntheticRuns_Univariate.py")
    return t1.evaluate_job(job)


def main():
    t1 = load_script("1_SyntheticRuns_Univariate.py")
    jobs = {(m, i): (m, i, tr) for m, i, tr in t1.build_jobs(n_per_method=1000, seed=t1.SEARCH_SEED)}
    sel = [jobs[k] for k in PICK]
    orig = {}
    with open(f"{ORIG}/results_tracks.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if (r["method"], r["trial_id"]) in PICK:
                orig[(r["method"], r["trial_id"], r["track_idx"])] = r
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(_run, sel))
    ok_all = True
    for res in results:
        for r in res["track_rows"]:
            r = json.loads(json.dumps(r, default=float))
            key = (r["method"], r["trial_id"], r["track_idx"])
            o = orig[key]
            same_c = o["centers_raw"] == r["centers_raw"]
            same_p = o["parents_raw"] == r["parents_raw"]
            ok_all &= same_c and same_p
            print(f"{key}: len={len(r['centers_raw'])} centers_match={same_c} parents_match={same_p}", flush=True)
    print("T1_REGRESSION", "PASS" if ok_all else "FAIL", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
