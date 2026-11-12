"""Night-3 N0 task 3, Debby link level: DESIGN ONLY (not run; PI decides).

Debby bundles (A2_debby_twin/vo_<Y>/bundles/*.npz) keep centers/parents only; the per-step graphs
(run_track debug tables) were dropped by 5_DebbyRuns.serialize_ensemble_bundle_debby. This script re-traces a
subset of configs for one twin year with the exact A2 recipe and keeps, for every member and step, the
non-zero links parent -> child (parent variable, stencil position, coefficient).

Reproduction: uses the A2 code (worktree trace-st-rev1-run-v5, tag rev1-fixes-v5, e744c79) and the same
inputs and seeds as A2_debby_twin.pbs (vo_input = vo500700_2006_rebuilt.nc, extra VO_P<Y>, ensemble seed
SEARCH_SEED + 1000*trial_id, member seeds from it), so centers/parents should match the A2 bundles bit for
bit on Derecho CPU nodes; the script checks this per member and records the result.
Config subset: 20 of the 75 retained (rebuilt baseline) configs, 5 per quartile of N_samples/N_features
(seed 0), M = 30 members.
Cost estimate (A2 vo_2004 results_jobs_final.csv elapsed_sec: mean 2879 s, max 7426 s per 30-member trial,
one worker per trial): 20 trials ~ 16 worker-hours; with 20 workers on a 24-core develop slice the wall time
is set by the slowest trial (~2.1 h) => ~50 core-h (develop, x1 charge) or 1 main node x 2.1 h x 128 =
~270 core-h (x1.5 premium). Memory: A2 used 48 workers in 235 GB; 20 workers -> request 120 GB.

python N0_task3_debby_retrace.py --year 2004 --n-cfg 20 --nproc 20 [--M 30]
Output: revision1/night3/N0/task3/debby_retrace_<Y>/trial_XXXXX_links.npz + check.csv
"""
import argparse
import importlib.util
import sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
CODE = Path("/glade/u/home/jhayron/Causality_CaStLe/trace-st-rev1-run-v5")   # A2 code (rev1-fixes-v5)
IN = R1 / "inputs_debby"
sys.path.insert(0, str(CODE))
_spec = importlib.util.spec_from_file_location("debbyruns", CODE / "Scripts_Paper/5_DebbyRuns.py")
D = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(D)
from trace_st import retention  # noqa: E402  (from CODE)

CASE = None


def init(year):
    global CASE
    CASE = D.build_case_debby(extra_inputs=[(f"VO_P{year}", f"{IN}/vo500700_{year}to2006.nc", "vo")],
                              vo_input=(f"{IN}/vo500700_2006_rebuilt.nc", "vo"))


def one(args):
    trial_id, trial, M, year, outdir = args
    params = D.build_params_from_trial_debby(trial, CASE)
    ens = D.run_ensemble(CASE, params, M=M, seed=int(D.SEARCH_SEED + 1000 * int(trial_id)))
    rec = dict(member=[], step=[], child=[], parent=[], pos=[], coef=[])
    cen, par, ok = [], [], []
    for m in ens["members"]:
        cen.append(m["centers"]); par.append(m["parents"]); ok.append(m["ok"])
        if not m["ok"] or m["out"] is None:
            continue
        for k, d in enumerate(m["out"].get("debug") or []):
            if d is None or len(d) == 0:
                continue
            n = len(d)
            rec["member"] += [m["member_id"]] * n; rec["step"] += [k] * n
            rec["child"] += [int(m["parents"][k])] * n
            rec["parent"] += list(d["parent_p"].astype(int)); rec["pos"] += list(d["pos_raw"].astype(int))
            rec["coef"] += list(d["causal_value"].astype(float))
    np.savez_compressed(Path(outdir) / f"trial_{trial_id:05d}_links.npz",
                        **{k: np.asarray(v) for k, v in rec.items()},
                        centers=np.array(cen, dtype=object), parents=np.array(par, dtype=object), ok=np.array(ok))
    # reproduction check against the A2 bundle
    b = np.load(R1 / f"A2_debby_twin/vo_{year}/bundles/debby_trial_{trial_id:05d}.npz", allow_pickle=True)
    same = []
    for i, (c, p) in enumerate(zip(cen, par)):
        L = int(b["lengths"][i])
        same.append(len(p) == L and np.array_equal(np.asarray(p, int), b["parents"][i, :L].astype(int))
                    and np.allclose(np.asarray(c, np.float32), b["centers"][i, :L], atol=1e-4))
    return dict(trial_id=trial_id, n_links=len(rec["coef"]), frac_members_identical=float(np.mean(same)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2004); ap.add_argument("--n-cfg", type=int, default=20)
    ap.add_argument("--nproc", type=int, default=20); ap.add_argument("--M", type=int, default=30)
    a = ap.parse_args()
    out = R1 / f"night3/N0/task3/debby_retrace_{a.year}"; out.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(R1 / "T4r_debby_vorebuilt/results_jobs_final.csv")
    ids = sorted(base.loc[retention.debby_retained(base), "trial_id"].astype(int))
    J = pd.read_csv(R1 / f"A2_debby_twin/vo_{a.year}/results_jobs_final.csv")
    J = J[J.trial_id.isin(ids)].copy()
    W = (pd.to_timedelta(J.timewindow) / pd.Timedelta("1h")).round().astype(int)
    nref = np.clip((J.box_size / 0.25).round().astype(int) + 1 - 2 * J.radius, 0, None) ** 2
    J["ratio"] = nref * (W - 1) / (4 * (2 * J.radius + 1) ** 2)
    J["q"] = pd.qcut(J.ratio, 4, labels=False, duplicates="drop")
    sel = J.groupby("q", group_keys=False).apply(lambda g: g.sample(min(len(g), a.n_cfg // 4), random_state=0))
    trials = dict(D.build_debby_jobs(n_trials=300, seed=D.SEARCH_SEED))   # same as A2 (--n-trials 300)
    tasks = [(int(t), trials[int(t)], a.M, a.year, str(out)) for t in sel.trial_id]
    sel[["trial_id", "ratio", "q", "box_size", "radius", "timewindow"]].to_csv(out / "selected_configs.csv", index=False)
    ctx = get_context("fork")
    init(a.year)  # build once in the parent (fails fast; fork shares CASE)
    with ctx.Pool(a.nproc) as pool:
        res = pool.map(one, tasks, chunksize=1)
    pd.DataFrame(res).to_csv(out / "check.csv", index=False)
    print(pd.DataFrame(res).to_string())


if __name__ == "__main__":
    main()
