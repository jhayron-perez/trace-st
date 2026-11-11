"""Task F (R1 medium 1): exact enumeration of the TraCE-ST selection tree vs Monte Carlo.

For T2-EN (alpha_mix 0.6), the first 5 T2r trial ids (configs drawn exactly as
Scripts_Paper/3_SyntheticRuns_Multivariate.py draws them with --seed 11), with
prob_rule forced to "linear" (selection probability S/sum S) and n_steps_time = N:

1. Exact tree. Starting from the event, at each node we execute the body of
   trace_st.trajectory.run_track for one step (same window slicing, same
   get_mcastle_box call, same candidate table, same pick_direction_then_weighted_delta
   call). The only change is the random generator: a recording/forcing stand-in
   captures (clusters, probs) passed to rng.choice in movement._pick_cluster and then
   forces each candidate cluster in turn. Each child state therefore has probability
   prob(parent) * S_c / sum S. Children of all nodes at the same depth with identical
   (lat, lon, child variable, terminal flag) are merged (exact: the process is Markov in
   that state because the date is fixed by the depth).
2. Monte Carlo. A member with seed s is equivalent to walking the stored tree with
   np.random.default_rng(s).choice(clusters, p=probs) at every node (the only rng use in
   run_track). This equivalence is asserted against real run_track calls for a few
   members per config before any MC numbers are produced.
3. Densities per lag k = 0..N: variable shares (with an "absent" category for tracks that
   stopped earlier), the joint (variable, location) distribution, and the T2 box density
   (_accumulate_density_box of the T2 script, per variable, normalised jointly).
   Errors are total-variation distances, TV = 0.5 * L1.

Usage: python F_exact_vs_mc.py --outdir ... --figdir ... --nproc 64 [--n-steps 4]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import trace_st as tst  # noqa: E402
from trace_st.trajectory import get_mcastle_box, run_track  # noqa: E402
from trace_st.movement import pick_direction_then_weighted_delta, snap_delta_to_stencil  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"),
                                                  ROOT / "Scripts_Paper" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T2 = load_script("3_SyntheticRuns_Multivariate.py")

ALPHA_MIX = 0.6
ALPHA_J = T2.ALPHA_MIX_GRID.index(ALPHA_MIX)
M_LIST = [10, 30, 100, 300]
N_REP = 10


# -----------------------------------------------------------------------------
# Forcing / recording stand-in for np.random.Generator (only .choice is used)
# -----------------------------------------------------------------------------
class ForcedRng:
    def __init__(self, forced=None):
        self.forced = forced
        self.calls = []

    def choice(self, clusters, p=None):
        clusters = np.asarray(clusters)
        self.calls.append((clusters.copy(), np.asarray(p, dtype=float).copy()))
        if self.forced is None:
            return clusters[0]
        return self.forced


# -----------------------------------------------------------------------------
# One run_track step, branching (body copied from trace_st/trajectory.py run_track)
# -----------------------------------------------------------------------------
G = {}  # worker globals: full_data, params per config


def _unpack(params):
    return dict(
        timeres=params["timeres"], spaceres=float(params["spaceres"]), box_size=float(params["box_size"]),
        r=int(params["radius"]), timewindow=params["timewindow"],
        montecarlo=bool(params.get("montecarlo", False)),
        beta_softmax=100.0 if params.get("beta_softmax", 100.0) is None else float(params.get("beta_softmax", 100.0)),
        prob_rule=str(params.get("prob_rule", "linear")), strength_mode=str(params.get("strength_mode", "native")),
        eps=float(params.get("eps_dbscan", 0.5)), min_samples=int(params.get("min_samples_dbscan", 3)),
        averaging_winners=bool(params.get("averaging_winners", True)),
        snap_to_grid=bool(params.get("snap_to_grid", True)),
        score_mode=str(params.get("score_mode", "sum_over_n")), gamma=float(params.get("gamma", 0.5)),
        alpha=float(params.get("alpha", 2.0)), prefer_sign=str(params.get("prefer_sign", "auto")),
        winner_summary=str(params.get("winner_summary", "mean")),
        cd_method=str(params.get("cd_method", "pcmci")).lower(), cd_kwargs=params.get("cd_kwargs", None),
    )


def _child_from_choice(list_candidates, delta_lat, delta_lon, chosen_sign, chosen_cluster, df_debug):
    """Exactly the child-update block of run_track (averaging_winners=True)."""
    if df_debug is None or len(df_debug) == 0 or chosen_cluster is None:
        winner = list_candidates.sort_values("causal_value_abs", ascending=False).iloc[0]
        return int(str(winner.vars).split("_")[0].replace("p", ""))
    cluster_col = "cluster_global" if "cluster_global" in df_debug.columns else "cluster"
    df_ch = df_debug.copy()
    if chosen_sign == "pos":
        df_ch = df_ch[df_ch["sign"] == 1.0]
    elif chosen_sign == "neg":
        df_ch = df_ch[df_ch["sign"] == -1.0]
    df_ch = df_ch[df_ch[cluster_col] == chosen_cluster].copy()
    if len(df_ch) == 0:
        winner = list_candidates.sort_values("causal_value_abs", ascending=False).iloc[0]
        return int(str(winner.vars).split("_")[0].replace("p", ""))
    parents = df_ch["vars"].apply(lambda s: int(str(s).split("_")[0].replace("p", ""))).unique()
    if len(parents) != 1:
        raise ValueError(f"[BUG] chosen cluster not unique in parent: {parents}")
    return int(parents[0])


def expand_state(args):
    """Expand one tree node. state = (lat, lon, child, date_end_str).

    Returns dict(status, clusters, probs, children=[(cluster, prob, lat, lon, child, out_of_domain)]).
    status in {"ok", "stop_time", "stop_noparents", "stop_cd"}.
    """
    cfg, lat, lon, child, date_end = args
    full_data = G["full_data"]
    P = G["unpacked"][cfg]
    dt_step = pd.to_timedelta(P["timeres"])
    expected_len = int(np.round(pd.to_timedelta(P["timewindow"]) / dt_step))
    date_end = pd.to_datetime(date_end)
    r = P["r"]
    current_center = [float(lat), float(lon)]
    current_child = int(child)

    date_start = date_end - pd.to_timedelta(P["timewindow"]) + dt_step
    full_data_win = full_data.sel(time=slice(date_start, date_end))
    if full_data_win.sizes.get("time", 0) < expected_len:
        return dict(status="stop_time", clusters=None, probs=None, children=[])
    var_list = list(full_data["var"].values)
    try:
        data_list = [full_data_win.sel(var=v).drop_vars("var") for v in var_list]
    except Exception:
        data_list = [full_data_win.sel(var=v) for v in var_list]
    try:
        output_mcastle = get_mcastle_box(current_center, P["box_size"], data_list, radius=r,
                                         child_variable=current_child, cd_method=P["cd_method"],
                                         cd_kwargs=P["cd_kwargs"], strength_mode=P["strength_mode"])
    except ValueError as e:
        if "No valid reference point" in str(e):
            return dict(status="stop_cd", clusters=None, probs=None, children=[])
        raise
    rows = []
    for p in range(1, len(data_list) + 1):
        g, vals = output_mcastle[f"p{p}_c{current_child}"]
        ii, jj, kk = np.where(g == "-->")
        for a, b, c in zip(ii, jj, kk):
            rows.append((f"p{p}_c{current_child}", float(vals[a, b, c]), (int(a), int(b), int(c))))
    if len(rows) == 0:
        return dict(status="stop_noparents", clusters=None, probs=None, children=[])
    list_candidates = pd.DataFrame(rows, columns=["vars", "causal_value", "position"])
    list_candidates["causal_value_abs"] = np.abs(list_candidates["causal_value"].to_numpy(dtype=float))

    assert P["averaging_winners"]
    kw = dict(radius=r, spaceres=P["spaceres"], eps=P["eps"], min_samples=P["min_samples"],
              montecarlo=P["montecarlo"], beta_softmax=P["beta_softmax"], prob_rule=P["prob_rule"],
              prefer_sign=P["prefer_sign"], alpha=P["alpha"], score_mode=P["score_mode"], gamma=P["gamma"],
              winner_summary=P["winner_summary"])

    rec = ForcedRng(None)
    _ = pick_direction_then_weighted_delta(list_candidates, rng=rec, **kw)
    lat_coord = data_list[0].lat
    latmax, latmin = float(lat_coord.max()), float(lat_coord.min())

    def _finish(forced):
        rr = ForcedRng(forced)
        dlat, dlon, csign, ccl, dfd = pick_direction_then_weighted_delta(list_candidates, rng=rr, **kw)
        newchild = _child_from_choice(list_candidates, dlat, dlon, csign, ccl, dfd)
        if P["snap_to_grid"]:
            dlat, dlon = snap_delta_to_stencil(dlat, dlon, spaceres=P["spaceres"], radius=r)
        new_lat = float(current_center[0]) + float(dlat)
        new_lon = (float(current_center[1]) + float(dlon)) % 360.0
        ood = (new_lat + P["box_size"] / 2 > latmax) or (new_lat - P["box_size"] / 2 < latmin)
        return new_lat, new_lon, newchild, bool(ood)

    children = []
    if len(rec.calls) == 0:  # deterministic path (no rng call)
        nl, nlo, nc, ood = _finish(None)
        children.append((None, 1.0, nl, nlo, nc, ood))
        return dict(status="ok", clusters=None, probs=None, children=children)
    assert len(rec.calls) == 1
    clusters, probs = rec.calls[0]
    for c, pc in zip(clusters, probs):
        if pc <= 0:
            continue
        nl, nlo, nc, ood = _finish(c)
        children.append((int(c), float(pc), nl, nlo, nc, ood))
    return dict(status="ok", clusters=clusters, probs=probs, children=children)


def _init_worker(full_data, unpacked, params_full=None):
    G["full_data"] = full_data
    G["unpacked"] = unpacked
    G["params_full"] = params_full


def skey(lat, lon, child, term):
    return (round(float(lat), 9), round(float(lon), 9), int(child), bool(term))


# -----------------------------------------------------------------------------
# Tree building
# -----------------------------------------------------------------------------
def build_tree(cfg, params, date_end0, n_steps, pool, log):
    """Returns levels: list over depth of dict key->{prob, lat, lon, child, term, n_paths}
    and node table: key(depth)->expansion result (with child keys)."""
    dt_step = pd.to_timedelta(params["timeres"])
    lat0, lon0 = float(params["starting_lat"]), float(params["starting_lon"]) % 360.0
    k0 = skey(lat0, lon0, params["child_of_interest"], False)
    levels = [{k0: dict(prob=1.0, lat=lat0, lon=lon0, child=int(params["child_of_interest"]), term=False,
                        n_paths=1, status=None)}]
    expansions = []
    n_unmerged_paths = 1
    for d in range(n_steps):
        de = str(pd.to_datetime(date_end0) - d * dt_step)
        cur = levels[-1]
        to_expand = [k for k, v in cur.items() if not v["term"]]
        t0 = time.time()
        res = pool.map(expand_state, [(cfg, cur[k]["lat"], cur[k]["lon"], cur[k]["child"], de) for k in to_expand],
                       chunksize=1)
        exp_d = {}
        nxt = {}
        n_paths_next = 0
        for k, rr in zip(to_expand, res):
            node = cur[k]
            node["status"] = rr["status"]
            ch_keys = []
            for (c, pc, nl, nlo, nc, ood) in rr["children"]:
                kk = skey(nl, nlo, nc, ood)
                ch_keys.append((c, kk))
                if kk not in nxt:
                    nxt[kk] = dict(prob=0.0, lat=nl, lon=nlo, child=nc, term=ood, n_paths=0, status=None)
                nxt[kk]["prob"] += node["prob"] * pc
                nxt[kk]["n_paths"] += node["n_paths"]
                n_paths_next += node["n_paths"]
            exp_d[k] = dict(status=rr["status"], clusters=rr["clusters"], probs=rr["probs"], child_keys=ch_keys)
        expansions.append(exp_d)
        levels.append(nxt)
        stopped_mass = sum(cur[k]["prob"] for k in to_expand if exp_d[k]["status"] != "ok")
        log(f"[cfg {cfg}] depth {d}->{d+1}: expanded {len(to_expand)} nodes in {time.time()-t0:.1f}s; "
            f"{len(nxt)} merged states, {n_paths_next} unmerged paths; mass stopped at this step {stopped_mass:.4g}; "
            f"mass next {sum(v['prob'] for v in nxt.values()):.6f}")
        if len(nxt) > 1e5:
            log(f"[cfg {cfg}] WARNING: >1e5 merged states at depth {d+1}")
    return levels, expansions


# -----------------------------------------------------------------------------
# MC walk on the tree (equivalent to run_track with params['seed'] = s)
# -----------------------------------------------------------------------------
def walk(levels, expansions, seed, n_steps):
    rng = np.random.default_rng(int(seed))
    k = next(iter(levels[0]))
    path = [k]
    for d in range(n_steps):
        if k[3]:  # out of domain: run_track broke after appending
            break
        e = expansions[d][k]
        if e["status"] != "ok":
            break
        if e["clusters"] is None:
            k = e["child_keys"][0][1]
        else:
            c = int(rng.choice(e["clusters"], p=e["probs"]))
            k = dict(e["child_keys"])[c]
        path.append(k)
    return path


# -----------------------------------------------------------------------------
# Distributions and distances
# -----------------------------------------------------------------------------
def exact_distributions(levels, n_steps, nvar):
    """Per lag: var shares (index 0 = absent, 1..nvar) and joint state probs {state_key: p}."""
    out = []
    for k in range(n_steps + 1):
        joint = {}
        var = np.zeros(nvar + 1)
        for kk, v in levels[k].items():
            key = (kk[0], kk[1], kk[2])
            joint[key] = joint.get(key, 0.0) + v["prob"]
            var[v["child"]] += v["prob"]
        var[0] = max(0.0, 1.0 - var[1:].sum())
        out.append(dict(var=var, joint=joint))
    return out


def mc_distributions(paths, n_steps, nvar):
    out = []
    M = len(paths)
    for k in range(n_steps + 1):
        joint = {}
        var = np.zeros(nvar + 1)
        for p in paths:
            if len(p) > k:
                kk = p[k]
                key = (kk[0], kk[1], kk[2])
                joint[key] = joint.get(key, 0.0) + 1.0 / M
                var[kk[2]] += 1.0 / M
            else:
                var[0] += 1.0 / M
        out.append(dict(var=var, joint=joint))
    return out


def tv_dict(a, b):
    keys = set(a) | set(b)
    alive_a = sum(a.values()); alive_b = sum(b.values())
    s = sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) + abs((1 - alive_a) - (1 - alive_b))
    return 0.5 * s


def box_density(joint, case, box_size, nvar):
    xs, ys = case["xs"], case["ys"]
    dens = np.zeros((nvar, len(ys), len(xs)))
    for (lat, lon, child), p in joint.items():
        tmp = np.zeros((len(ys), len(xs)))
        T2._accumulate_density_box(tmp, xs, ys, np.array([[lat, lon]]), box_size)
        dens[child - 1] += p * tmp
    tot = dens.sum()
    return dens / tot if tot > 0 else dens


class DensityCache:
    """Box density exactly as T2._accumulate_density_box (xs in [xc-h, xc+h], ys in [yc-h, yc+h]),
    stored as index ranges (the masks are contiguous because xs, ys are sorted)."""
    def __init__(self, case, box_size, nvar):
        self.xs, self.ys = np.asarray(case["xs"]), np.asarray(case["ys"])
        self.half = box_size / 2.0
        self.nvar = nvar
        self.cache = {}
        # self-check of the range representation against the T2 function
        for (yc, xc) in [(100.0, 100.0), (3.3, 197.1), (150.25, 12.7)]:
            ref = np.zeros((len(self.ys), len(self.xs)))
            T2._accumulate_density_box(ref, self.xs, self.ys, np.array([[yc, xc]]), box_size)
            tst_ = np.zeros_like(ref)
            y0, y1, x0, x1 = self.rng_(yc, xc)
            tst_[y0:y1, x0:x1] += 1.0
            assert np.array_equal(ref, tst_)

    def rng_(self, lat, lon):
        k = (lat, lon)
        if k not in self.cache:
            xm = np.where((self.xs >= lon - self.half) & (self.xs <= lon + self.half))[0]
            ym = np.where((self.ys >= lat - self.half) & (self.ys <= lat + self.half))[0]
            if len(xm) == 0 or len(ym) == 0:
                self.cache[k] = (0, 0, 0, 0)
            else:
                self.cache[k] = (ym[0], ym[-1] + 1, xm[0], xm[-1] + 1)
        return self.cache[k]

    def density(self, joint):
        dens = np.zeros((self.nvar, len(self.ys), len(self.xs)))
        for (lat, lon, child), p in joint.items():
            y0, y1, x0, x1 = self.rng_(lat, lon)
            dens[child - 1, y0:y1, x0:x1] += p
        tot = dens.sum()
        return dens / tot if tot > 0 else dens


def pooled(dlist, n_steps, nvar):
    """Pool lags 1..N (like T2 step-based shares): average of per-lag joints."""
    joint = {}
    var = np.zeros(nvar + 1)
    for k in range(1, n_steps + 1):
        for kk, p in dlist[k]["joint"].items():
            joint[kk] = joint.get(kk, 0.0) + p / n_steps
        var += dlist[k]["var"] / n_steps
    return dict(var=var, joint=joint)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--nproc", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=4)
    ap.add_argument("--configs", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-verify", type=int, default=4, help="run_track members per config for the equivalence assert")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    logf = open(outdir / "F_log.txt", "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    N = int(args.n_steps)
    case = T2.build_multivariate_case(alpha_mix=ALPHA_MIX, T=40, N_inject=8, seed=T2.GLOBAL_SEED)
    full_data = case["full_data_run"]
    nvar = int(full_data.sizes["var"])
    jobs = dict(T2.build_multivar_jobs(n_trials=300, seed=T2.SEARCH_SEED))

    # check configs against the T2r table
    t2r = pd.read_csv("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/T2r_multivariate_en/results_alphas_final.csv")
    params_by_cfg, unpacked, cfg_rows = {}, {}, []
    for cfg in args.configs:
        trial = jobs[cfg]
        row = t2r[(t2r.trial_id == cfg) & (t2r.alpha_mix == ALPHA_MIX)].iloc[0]
        for k in ["timewindow", "box_size", "radius", "eps_dbscan", "score_mode", "alpha", "prob_rule", "beta_softmax"]:
            assert str(row[k]) == str(trial[k]) or np.isclose(float(row[k]), float(trial[k])), (cfg, k, row[k], trial[k])
        assert np.isclose(row["lambda_a"], trial["lambda_a"]) and np.isclose(row["l1_ratio"], trial["l1_ratio"])
        p = T2.build_params_from_trial_multivar(trial)
        orig_rule = p["prob_rule"]
        p["prob_rule"] = "linear"
        p["n_steps_time"] = N
        p["starting_lat"] = float(case["event"][0]); p["starting_lon"] = float(case["event"][1])
        params_by_cfg[cfg] = p
        unpacked[cfg] = _unpack(p)
        cfg_rows.append(dict(trial_id=cfg, original_prob_rule=orig_rule, used_prob_rule="linear",
                             **{k: trial[k] for k in ["timewindow", "box_size", "radius", "eps_dbscan", "score_mode",
                                                      "alpha", "beta_softmax", "lambda_a", "l1_ratio"]}))
    pd.DataFrame(cfg_rows).to_csv(outdir / "F_configs.csv", index=False)
    log(f"configs: {cfg_rows}")

    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.nproc, initializer=_init_worker, initargs=(full_data, unpacked, params_by_cfg))
    _init_worker(full_data, unpacked, params_by_cfg)

    rows_err, rows_tree, rows_exact = [], [], []
    for cfg in args.configs:
        p = params_by_cfg[cfg]
        t0 = time.time()
        levels, expansions = build_tree(cfg, p, case["date_end"], N, pool, log)
        t_tree = time.time() - t0

        # --- equivalence check against run_track (real calls, parallel)
        vseeds = [int(x) for x in np.random.default_rng(999 + cfg).integers(0, 2_000_000_000, args.n_verify)]
        vres = pool.map(_verify_member, [(cfg, s, case["date_end"]) for s in vseeds], chunksize=1)
        for s, (centers, parents) in zip(vseeds, vres):
            path = walk(levels, expansions, s, N)
            pc = np.array([[k[0], k[1]] for k in path]); pp = np.array([k[2] for k in path])
            assert len(path) == len(centers), (cfg, s, len(path), len(centers))
            assert np.allclose(pc, np.round(centers, 9), atol=1e-8), (cfg, s, pc, centers)
            assert np.array_equal(pp, parents), (cfg, s, pp, parents)
        log(f"[cfg {cfg}] run_track equivalence OK for {len(vseeds)} seeded members: {vseeds}")

        ex = exact_distributions(levels, N, nvar)
        ex_pool = pooled(ex, N, nvar)
        dc = DensityCache(case, float(p["box_size"]), nvar)
        ex_dens = [dc.density(ex[k]["joint"]) for k in range(N + 1)]
        ex_dens_pool = dc.density(ex_pool["joint"])
        for k in range(N + 1):
            rows_tree.append(dict(trial_id=cfg, lag=k, n_states=len(levels[k]),
                                  n_unmerged_paths=int(sum(v["n_paths"] for v in levels[k].values())),
                                  mass=float(sum(v["prob"] for v in levels[k].values())),
                                  entropy_states=float(-sum(v["prob"] * np.log(v["prob"]) for v in levels[k].values() if v["prob"] > 0)),
                                  tree_seconds=t_tree))
            rows_exact.append(dict(trial_id=cfg, lag=k, **{f"share_var{i}": float(ex[k]["var"][i]) for i in range(1, nvar + 1)},
                                   share_absent=float(ex[k]["var"][0])))
        rows_exact.append(dict(trial_id=cfg, lag="pooled_1_N", **{f"share_var{i}": float(ex_pool["var"][i]) for i in range(1, nvar + 1)},
                               share_absent=float(ex_pool["var"][0])))

        # --- MC replicate ensembles
        for M in M_LIST:
            for rep in range(N_REP):
                rng_local = np.random.default_rng(int(T2.SEARCH_SEED + 1000 * cfg + ALPHA_J) * 1000 + 10 * M + rep)
                seeds = [int(rng_local.integers(0, 2_000_000_000)) for _ in range(M)]
                paths = [walk(levels, expansions, s, N) for s in seeds]
                mc = mc_distributions(paths, N, nvar)
                mc_pool = pooled(mc, N, nvar)
                for k in list(range(1, N + 1)) + ["pooled"]:
                    e = ex_pool if k == "pooled" else ex[k]
                    m_ = mc_pool if k == "pooled" else mc[k]
                    ed = ex_dens_pool if k == "pooled" else ex_dens[k]
                    md = dc.density(m_["joint"])
                    rows_err.append(dict(trial_id=cfg, M=M, rep=rep, lag=k,
                                         tv_var=0.5 * float(np.abs(e["var"] - m_["var"]).sum()),
                                         tv_joint=tv_dict(e["joint"], m_["joint"]),
                                         tv_density=0.5 * float(np.abs(ed - md).sum())))
        log(f"[cfg {cfg}] done in {time.time()-t0:.1f}s")
        pd.DataFrame(rows_err).to_csv(outdir / "F_mc_errors.csv", index=False)
        pd.DataFrame(rows_tree).to_csv(outdir / "F_tree_sizes.csv", index=False)
        pd.DataFrame(rows_exact).to_csv(outdir / "F_exact_shares.csv", index=False)
        # tree dump (states per lag) for the record
        st = [dict(trial_id=cfg, lag=k, lat=v["lat"], lon=v["lon"], var=v["child"], prob=v["prob"],
                   out_of_domain=v["term"], status_next=v["status"], n_paths=v["n_paths"])
              for k in range(N + 1) for v in levels[k].values()]
        pd.DataFrame(st).to_csv(outdir / f"F_tree_states_trial{cfg:05d}.csv", index=False)
    pool.close()
    summarize_and_plot(outdir, Path(args.figdir), log)


def _verify_member(args):
    """Real run_track call with params['seed'] = s (as run_ensemble in the T2 script)."""
    cfg, s, date_end = args
    p = copy.deepcopy(G["params_full"][cfg])
    p["seed"] = int(s)
    out = run_track(G["full_data"], p, date_end=date_end, return_debug=False, return_cluster_summaries=False)
    return np.asarray(out["centers"], dtype=float), np.asarray(out["parents"], dtype=int)


def summarize_and_plot(outdir, figdir, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(outdir / "F_mc_errors.csv")
    df["lag"] = df["lag"].astype(str)
    agg = df.groupby(["trial_id", "M", "lag"])[["tv_var", "tv_joint", "tv_density"]].mean().reset_index()
    agg.to_csv(outdir / "F_mc_errors_mean_over_reps.csv", index=False)
    # slopes per metric on the median over configs (mean over reps, all lags 1..N averaged)
    lagsel = agg[agg.lag != "pooled"]
    per_cfg = lagsel.groupby(["trial_id", "M"])[["tv_var", "tv_joint", "tv_density"]].mean().reset_index()
    summ = []
    for met in ["tv_var", "tv_joint", "tv_density"]:
        med = per_cfg.groupby("M")[met].median()
        lo = per_cfg.groupby("M")[met].min(); hi = per_cfg.groupby("M")[met].max()
        slope = np.polyfit(np.log(med.index.values), np.log(med.values), 1)[0]
        slopes_cfg = [np.polyfit(np.log(g.M.values), np.log(g[met].values), 1)[0] for _, g in per_cfg.groupby("trial_id")]
        for M in med.index:
            summ.append(dict(metric=met, M=int(M), median=float(med[M]), min=float(lo[M]), max=float(hi[M]),
                             slope_median=float(slope), slope_cfg_min=float(np.min(slopes_cfg)),
                             slope_cfg_max=float(np.max(slopes_cfg))))
    S = pd.DataFrame(summ)
    S.to_csv(outdir / "F_summary.csv", index=False)
    log(S.to_string())

    plt.rcParams.update({"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18, "legend.fontsize": 15})
    colors = {"tv_var": "#0072B2", "tv_joint": "#D55E00", "tv_density": "#CC79A7"}
    labels = {"tv_var": "variable shares", "tv_joint": "joint (variable, location)", "tv_density": "box density (T2 definition)"}
    markers = {"tv_var": "o", "tv_joint": "s", "tv_density": "^"}
    fig, ax = plt.subplots(figsize=(16, 9))
    for met in ["tv_var", "tv_joint", "tv_density"]:
        s = S[S.metric == met]
        ax.fill_between(s.M, s["min"], s["max"], color=colors[met], alpha=0.18, lw=0)
        ax.plot(s.M, s["median"], marker=markers[met], ms=10, lw=2.5, color=colors[met],
                label=f"{labels[met]} (slope {s.slope_median.iloc[0]:.2f})")
    Ms = np.array([10, 300])
    ref0 = S[(S.metric == "tv_var") & (S.M == 10)]["median"].iloc[0]
    ax.plot(Ms, ref0 * np.sqrt(10 / Ms), "k--", lw=2, label=r"$1/\sqrt{M}$ reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Ensemble size M (members)")
    ax.set_ylabel("Total-variation distance to exact (unitless)")
    ax.set_title("T2-EN, alpha_mix 0.6: Monte Carlo vs exact selection tree (N = 4 steps)")
    ax.set_xticks([10, 30, 100, 300]); ax.set_xticklabels(["10", "30", "100", "300"])
    ax.legend(loc="lower left", frameon=False)
    ax.text(0.99, 0.98, "lines: median over 5 configs; bands: min-max\nerror: mean over 10 replicates and lags 1-4",
            transform=ax.transAxes, ha="right", va="top", fontsize=14)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(figdir / f"F_error_vs_M.{ext}", dpi=150)
    log(f"figure written: {figdir / 'F_error_vs_M.png'}")


if __name__ == "__main__":
    main()
