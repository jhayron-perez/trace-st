from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import copy
import json
import pickle
import warnings
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.exceptions import ConvergenceWarning

# -----------------------------------------------------------------------------
# Project import
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import trace_st as tst  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
GLOBAL_SEED = 11
SEARCH_SEED = 11

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
PATH_FILES = "/glade/derecho/scratch/jhayron/DataCaStLeBTs/JointFiles/"
TIME_SLICE = ("2021-05-20", "2021-06-30")

DATE_END = "2021-06-30"
EVENT_LAT = 52.5
EVENT_LON = 240.0

RES_PNW = 2.5
BOX_SIZE_PNW_BASE = 30.0

VAR_NAME_MAP = {
    "z500": "Z500",
    "olr": "MTNLWRF",
    "skt": "SKT",
    "slhf": "MSLHF",
    "sshf": "MSSHF",
    "tclw": "TCLW",
    "tcwv": "TCWV",
    "vimd": "MVIMD",
    "w500": "W500",
    "z100": "Z100",
    "z10": "Z10",
}

VAR_COMBOS: List[List[str]] = [
    ["z500", "z10", "olr", "vimd", "tcwv", "slhf", "sshf"],
]

# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_full_data_for_vars(varstems: List[str]) -> xr.DataArray:
    list_names = [VAR_NAME_MAP[v] for v in varstems]
    full_data = tst.data_io.load_full_data(
        path_files=PATH_FILES,
        varstems=varstems,
        list_names_vars=list_names,
        time_slice=TIME_SLICE,
        join="inner",
    )
    return full_data


def regrid_to_2p5(full_data: xr.DataArray) -> xr.DataArray:
    new_lats = np.arange(-30.0, 90.0 + RES_PNW, RES_PNW)
    new_lons = np.arange(0.0, 360.0, RES_PNW)

    full_data_2p5 = full_data.interp(
        lat=new_lats,
        lon=new_lons,
        method="linear",
    )
    return full_data_2p5


def build_case_pnw(var_combo: List[str] | None = None) -> dict:
    if var_combo is None:
        var_combo = VAR_COMBOS[0]

    full_data = load_full_data_for_vars(var_combo)
    full_data = regrid_to_2p5(full_data)

    params_base = dict(
        timeres="1d",
        spaceres=RES_PNW,
        box_size=BOX_SIZE_PNW_BASE,
        radius=5,
        starting_lat=EVENT_LAT,
        starting_lon=EVENT_LON,
        timewindow="5d",
        child_of_interest=1,  # z500
        n_steps_time=30,
        averaging_winners=True,
        snap_to_grid=False,
        montecarlo=False,
        beta_softmax=8.0,
        eps_dbscan=0.10,
        min_samples_dbscan=3,
        score_mode="mean",
        gamma=0.5,
        alpha=2.0,
        prefer_sign="both",
        verbose=False,
        prob_rule="softmax",
        winner_summary="mean",
        cd_method="granger",
        cd_kwargs=dict(
            lambda_a=0.5,
            l1_ratio=0.5,
            dependence_threshold=1e-8,
            max_iter=10000,
            fit_intercept=False,
            refit_ridge=False,
            ridge_alpha=1e-2,
        ),
    )

    return dict(
        case_name="pnw21",
        full_data=full_data,
        date_end=DATE_END,
        event_lat=EVENT_LAT,
        event_lon=EVENT_LON,
        var_combo=list(var_combo),
        var_names=list(full_data["var"].values),
        params_base=params_base,
    )


# -----------------------------------------------------------------------------
# Search space
# Keep consistent with prior cases, but sensible for daily 2.5° data.
# Do not drift too far from your working notebook ranges.
# -----------------------------------------------------------------------------
SEARCH_SPACE_SHARED_PNW = dict(
    timewindow=["4d", "5d", "6d"],
    box_size=[20.0, 25.0, 30.0, 35.0, 40.0],
    radius=[3, 4, 5, 6, 7],
    eps_dbscan=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    min_samples_dbscan=[2, 3, 4],
    score_mode=["mean", "sum"],
    winner_summary=["mean"],
    alpha=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32, 64],
    prefer_sign=["both"],
)

SEARCH_SPACE_PROB_PNW = dict(
    prob_rule=["linear", "softmax"],
    beta_softmax=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32, 64],
)

SEARCH_SPACE_GRANGER_PNW = dict(
    lambda_a=lambda rng: float(10 ** rng.uniform(-2.5, 0.0)),
    l1_ratio=lambda rng: float(10 ** rng.uniform(-1.0, 0.0)),
    dependence_threshold=[1e-7],
    max_iter=[10000],
    fit_intercept=[False],
    refit_ridge=[False],
    ridge_alpha=[1e-2],
)

SEARCH_SPACE_EXTRA_PNW = dict(
    gamma=[0.5],
)


def _draw_from_space(rng: np.random.Generator, x):
    if callable(x):
        return x(rng)
    v = rng.choice(x)
    if isinstance(v, np.str_):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def sample_pnw_trial(rng: np.random.Generator) -> dict:
    trial = {}

    for space in [
        SEARCH_SPACE_SHARED_PNW,
        SEARCH_SPACE_PROB_PNW,
        SEARCH_SPACE_GRANGER_PNW,
        SEARCH_SPACE_EXTRA_PNW,
    ]:
        for k, v in space.items():
            trial[k] = _draw_from_space(rng, v)

    if trial["prob_rule"] == "linear":
        trial["beta_softmax"] = 0.0

    return trial


def build_pnw_jobs(n_trials: int, seed: int = SEARCH_SEED):
    rng = np.random.default_rng(seed)
    jobs = []

    for trial_id in range(n_trials):
        trial = sample_pnw_trial(rng)
        trial["run_id"] = int(trial_id)
        jobs.append((trial_id, trial))

    return jobs


def build_params_from_trial_pnw(trial: dict, case: dict) -> dict:
    p = copy.deepcopy(case["params_base"])

    for k in [
        "timewindow",
        "box_size",
        "radius",
        "eps_dbscan",
        "min_samples_dbscan",
        "score_mode",
        "winner_summary",
        "alpha",
        "prefer_sign",
        "prob_rule",
        "beta_softmax",
        "gamma",
    ]:
        p[k] = trial[k]

    p["montecarlo"] = True
    p["snap_to_grid"] = False
    p["averaging_winners"] = True
    p["verbose"] = False
    p["cd_method"] = "granger"

    p["cd_kwargs"]["lambda_a"] = trial["lambda_a"]
    p["cd_kwargs"]["l1_ratio"] = trial["l1_ratio"]
    p["cd_kwargs"]["dependence_threshold"] = trial["dependence_threshold"]
    p["cd_kwargs"]["max_iter"] = trial["max_iter"]
    p["cd_kwargs"]["fit_intercept"] = trial["fit_intercept"]
    p["cd_kwargs"]["refit_ridge"] = trial["refit_ridge"]
    p["cd_kwargs"]["ridge_alpha"] = trial["ridge_alpha"]

    p["run_id"] = int(trial["run_id"])

    return p


# -----------------------------------------------------------------------------
# Run helpers
# -----------------------------------------------------------------------------
def run_one_track(full_data, params, date_end):
    """
    Single TraCE-ST run; returns centers, parents, and full output dict.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        out = tst.trajectory.run_track(
            full_data,
            params,
            date_end=date_end,
            return_debug=True,
            return_cluster_summaries=False,
        )

    centers = np.asarray(out["centers"], dtype=float)
    parents = np.asarray(out["parents"], dtype=int)
    return centers, parents, out


def _run_one_track_worker(args):
    full_data, params, date_end, member_id = args

    try:
        centers, parents, out = run_one_track(full_data, params, date_end)
        return dict(
            success=True,
            member_id=int(member_id),
            centers=centers,
            parents=parents,
            out=out,
            error=None,
        )
    except Exception as e:
        return dict(
            success=False,
            member_id=int(member_id),
            centers=None,
            parents=None,
            out=None,
            error=repr(e),
        )


def run_ensemble_tracks(
    full_data,
    params,
    date_end,
    n_members=20,
    nproc=None,
    start_method="spawn",
    drop_failed=False,
    verbose=True,
):
    """
    Run an ensemble of TraCE-ST tracks in parallel.
    """
    if nproc is None:
        nproc = min(n_members, os.cpu_count() or 1)

    args_list = [
        (full_data, copy.deepcopy(params), date_end, i)
        for i in range(n_members)
    ]

    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=nproc) as pool:
        results = pool.map(_run_one_track_worker, args_list)

    n_failed = sum(not r["success"] for r in results)
    if verbose and n_failed > 0:
        print(f"{n_failed} / {n_members} ensemble members failed.", flush=True)

    if drop_failed:
        results = [r for r in results if r["success"]]

    return results


# -----------------------------------------------------------------------------
# Density / window helpers
# -----------------------------------------------------------------------------
def density_by_variable_for_time_window_boxes(
    centers_ens,
    parents_ens,
    *,
    full_data_run,
    box_size,
    t_start,
    t_end,
):
    """
    Box-footprint density accumulated by variable over a time window [t_start, t_end).
    """
    lat_name = "lat" if "lat" in full_data_run.coords else "latitude"
    lon_name = "lon" if "lon" in full_data_run.coords else "longitude"

    lats = np.asarray(full_data_run[lat_name].values, dtype=float)
    lons = np.asarray(full_data_run[lon_name].values, dtype=float) % 360.0
    var_names = [str(v) for v in full_data_run["var"].values]
    n_vars = len(var_names)

    counts = np.zeros((n_vars, len(lats), len(lons)), dtype=np.float64)
    half = 0.5 * float(box_size)

    for centers, parents in zip(centers_ens, parents_ens):
        centers = np.asarray(centers, dtype=float)
        parents = np.asarray(parents, dtype=int)

        nT = min(len(centers), len(parents))
        a = max(0, int(t_start))
        b = min(nT, int(t_end))
        if b <= a:
            continue

        for (lat_c, lon_c), v in zip(centers[a:b], parents[a:b]):
            k = int(v) - 1
            if not (0 <= k < n_vars):
                continue

            lat_c = float(lat_c)
            lon_c = float(lon_c) % 360.0

            lat_lo, lat_hi = lat_c - half, lat_c + half
            ilats = np.where((lats >= lat_lo) & (lats <= lat_hi))[0]
            if ilats.size == 0:
                continue

            lon_lo = (lon_c - half) % 360.0
            lon_hi = (lon_c + half) % 360.0

            if lon_lo <= lon_hi:
                ilons = np.where((lons >= lon_lo) & (lons <= lon_hi))[0]
            else:
                ilons = np.where((lons >= lon_lo) | (lons <= lon_hi))[0]

            if ilons.size == 0:
                continue

            counts[k][np.ix_(ilats, ilons)] += 1.0

    return counts, var_names, lats, lons


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in km.
    lon can be in degrees on any consistent convention.
    """
    R = 6371.0

    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lon1 = np.deg2rad(np.asarray(lon1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def cumulative_path_length_km(centers):
    """
    Sum of step-to-step great-circle distances along a trajectory.
    centers[:, 0] = lat, centers[:, 1] = lon
    """
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        return 0.0

    lats = centers[:, 0]
    lons = centers[:, 1]
    seg = haversine_km(lats[:-1], lons[:-1], lats[1:], lons[1:])
    return float(np.nansum(seg))


def mean_step_length_km(centers):
    """
    Mean step-to-step great-circle distance along a trajectory.
    """
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        return 0.0

    lats = centers[:, 0]
    lons = centers[:, 1]
    seg = haversine_km(lats[:-1], lons[:-1], lats[1:], lons[1:])
    return float(np.nanmean(seg))


def net_displacement_start_to_end_km(centers):
    """
    Great-circle distance from trajectory start to trajectory end.
    """
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        return 0.0

    return float(
        haversine_km(
            centers[0, 0], centers[0, 1],
            centers[-1, 0], centers[-1, 1],
        )
    )


# -----------------------------------------------------------------------------
# Summary helpers
# -----------------------------------------------------------------------------
def summarize_ensemble_results(
    ensemble_results_all,
    *,
    params,
    full_data_run,
    event_lat,
    event_lon,
    windows=((0, 10), (10, 20), (20, 30)),
    window_labels=("days_1_10", "days_11_20", "days_21_30"),
):
    """
    Summarize one hyperparameter configuration.
    ensemble_results_all should include both successes and failures.
    """
    n_requested = len(ensemble_results_all)
    success_mask = [bool(r["success"]) for r in ensemble_results_all]
    n_success = int(sum(success_mask))
    n_failed = int(n_requested - n_success)
    success_rate = n_success / n_requested if n_requested > 0 else np.nan

    failed_member_ids = [int(r["member_id"]) for r in ensemble_results_all if not r["success"]]
    failed_errors = [str(r["error"]) for r in ensemble_results_all if not r["success"]]

    successful = [r for r in ensemble_results_all if r["success"]]

    summary = {
        "run_id": params.get("run_id", None),
        "n_requested": n_requested,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": success_rate,
        "failed_member_ids": failed_member_ids,
        "failed_errors": failed_errors,
    }

    if n_success == 0:
        summary["mean_steps"] = np.nan
        summary["median_steps"] = np.nan
        summary["min_steps"] = np.nan
        summary["max_steps"] = np.nan
        summary["fraction_full_length"] = np.nan

        summary["mean_displacement_km"] = np.nan
        summary["median_displacement_km"] = np.nan
        summary["max_displacement_km"] = np.nan

        summary["mean_path_length_km"] = np.nan
        summary["median_path_length_km"] = np.nan
        summary["max_path_length_km"] = np.nan

        summary["mean_step_length_km"] = np.nan
        summary["median_step_length_km"] = np.nan
        summary["mean_net_displacement_km"] = np.nan
        summary["median_net_displacement_km"] = np.nan

        summary["mean_unique_parents"] = np.nan
        summary["median_unique_parents"] = np.nan
        summary["parent_entropy_all"] = np.nan
        summary["window_stats"] = {}
        return summary

    centers_ens = [np.asarray(r["centers"], dtype=float) for r in successful]
    parents_ens = [np.asarray(r["parents"], dtype=int) for r in successful]

    # ---------- trajectory lengths ----------
    steps_arr = np.array([len(c) for c in centers_ens], dtype=float)
    summary["mean_steps"] = float(np.mean(steps_arr))
    summary["median_steps"] = float(np.median(steps_arr))
    summary["min_steps"] = int(np.min(steps_arr))
    summary["max_steps"] = int(np.max(steps_arr))
    summary["fraction_full_length"] = float(np.mean(steps_arr == params["n_steps_time"]))

    # ---------- displacement from target event to final point ----------
    end_lats = np.array([c[-1, 0] for c in centers_ens if len(c) > 0], dtype=float)
    end_lons = np.array([c[-1, 1] for c in centers_ens if len(c) > 0], dtype=float)

    disp_arr = haversine_km(
        np.full_like(end_lats, float(event_lat)),
        np.full_like(end_lons, float(event_lon)),
        end_lats,
        end_lons,
    )
    summary["mean_displacement_km"] = float(np.mean(disp_arr))
    summary["median_displacement_km"] = float(np.median(disp_arr))
    summary["max_displacement_km"] = float(np.max(disp_arr))

    # ---------- cumulative path length ----------
    path_arr = np.array([cumulative_path_length_km(c) for c in centers_ens], dtype=float)
    summary["mean_path_length_km"] = float(np.mean(path_arr))
    summary["median_path_length_km"] = float(np.median(path_arr))
    summary["max_path_length_km"] = float(np.max(path_arr))

    # ---------- new: mean displacement per step ----------
    mean_step_arr = np.array([mean_step_length_km(c) for c in centers_ens], dtype=float)
    summary["mean_step_length_km"] = float(np.mean(mean_step_arr))
    summary["median_step_length_km"] = float(np.median(mean_step_arr))

    # ---------- new: net displacement from trajectory start to end ----------
    net_disp_arr = np.array([net_displacement_start_to_end_km(c) for c in centers_ens], dtype=float)
    summary["mean_net_displacement_km"] = float(np.mean(net_disp_arr))
    summary["median_net_displacement_km"] = float(np.median(net_disp_arr))

    # ---------- parent diversity ----------
    unique_parent_counts = np.array([len(np.unique(p)) for p in parents_ens], dtype=float)
    summary["mean_unique_parents"] = float(np.mean(unique_parent_counts))
    summary["median_unique_parents"] = float(np.median(unique_parent_counts))

    all_parents = np.concatenate([p for p in parents_ens if len(p) > 0]).astype(int)
    if len(all_parents) > 0:
        vals, counts = np.unique(all_parents, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-12))
    else:
        entropy = np.nan
    summary["parent_entropy_all"] = float(entropy)

    # ---------- per-window densities / importances ----------
    window_stats = {}
    for (a, b), label in zip(windows, window_labels):
        counts_box, var_names, _, _ = density_by_variable_for_time_window_boxes(
            centers_ens,
            parents_ens,
            full_data_run=full_data_run,
            box_size=params["box_size"],
            t_start=a,
            t_end=b,
        )

        integrated_density = counts_box.sum(axis=(1, 2)).astype(float)
        total_density = integrated_density.sum()

        if total_density > 0:
            integrated_density_norm = integrated_density / total_density
        else:
            integrated_density_norm = np.full_like(integrated_density, np.nan, dtype=float)

        parent_counts = np.zeros(len(var_names), dtype=float)
        for p in parents_ens:
            nT = len(p)
            aa = max(0, int(a))
            bb = min(nT, int(b))
            if bb <= aa:
                continue

            chunk = p[aa:bb]
            for j in range(len(var_names)):
                parent_counts[j] += np.sum(chunk == (j + 1))

        parent_total = parent_counts.sum()
        if parent_total > 0:
            parent_freq_norm = parent_counts / parent_total
        else:
            parent_freq_norm = np.full_like(parent_counts, np.nan, dtype=float)

        flat = counts_box.reshape(len(var_names), -1)
        density_top5_frac = {}
        for j, vname in enumerate(var_names):
            arr = flat[j]
            arr = arr[np.isfinite(arr)]

            if arr.size == 0 or np.nansum(arr) == 0:
                density_top5_frac[vname] = np.nan
            else:
                thr = np.percentile(arr, 95.0)
                density_top5_frac[vname] = float(arr[arr >= thr].sum() / arr.sum())

        window_stats[label] = {
            "integrated_density_by_var": {
                str(v): float(x) for v, x in zip(var_names, integrated_density)
            },
            "integrated_density_by_var_norm": {
                str(v): float(x) for v, x in zip(var_names, integrated_density_norm)
            },
            "parent_count_by_var": {
                str(v): float(x) for v, x in zip(var_names, parent_counts)
            },
            "parent_freq_by_var_norm": {
                str(v): float(x) for v, x in zip(var_names, parent_freq_norm)
            },
            "density_top5_fraction_by_var": density_top5_frac,
            "total_integrated_density": float(total_density),
        }

    summary["window_stats"] = window_stats

    # ---------- spread at selected steps ----------
    for step_name, step_idx in [("step_10", 9), ("step_20", 19), ("step_30", 29)]:
        pts = []
        for c in centers_ens:
            if len(c) > step_idx:
                pts.append(c[step_idx])

        if len(pts) == 0:
            summary[f"{step_name}_n_members"] = 0
            summary[f"{step_name}_mean_lat"] = np.nan
            summary[f"{step_name}_mean_lon"] = np.nan
            summary[f"{step_name}_mean_radius_km"] = np.nan
        else:
            pts = np.asarray(pts, dtype=float)
            mean_lat = float(np.mean(pts[:, 0]))
            mean_lon = float(np.mean(pts[:, 1]))
            rad = haversine_km(
                np.full(len(pts), mean_lat),
                np.full(len(pts), mean_lon),
                pts[:, 0],
                pts[:, 1],
            )

            summary[f"{step_name}_n_members"] = int(len(pts))
            summary[f"{step_name}_mean_lat"] = mean_lat
            summary[f"{step_name}_mean_lon"] = mean_lon
            summary[f"{step_name}_mean_radius_km"] = float(np.mean(rad))

    return summary


# -----------------------------------------------------------------------------
# JSON / saving helpers
# -----------------------------------------------------------------------------
def make_jsonable(obj):
    """
    Recursively convert numpy objects into JSON-serializable Python objects.
    """
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def atomic_write_json(path, obj):
    """
    Safely write json to disk.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(obj), f, indent=2)
    tmp.replace(path)


def atomic_write_pickle(path, obj):
    """
    Safely write pickle to disk.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


def atomic_write_csv(path, df):
    """
    Safely write csv to disk.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def save_run_outputs(
    out_dir,
    *,
    params,
    summary,
    ensemble_results,
):
    """
    Save params, summary, and full ensemble results for one run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = int(params.get("run_id", -1))
    run_dir = out_dir / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_json(run_dir / "params.json", params)
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_pickle(run_dir / "ensemble_results.pkl", ensemble_results)

    return run_dir


# -----------------------------------------------------------------------------
# Flatten summary for master table
# -----------------------------------------------------------------------------
def flatten_summary_for_table(summary):
    """
    Turn nested summary dict into one flat row for a DataFrame.
    """
    row = {
        "run_id": summary.get("run_id"),
        "n_requested": summary.get("n_requested"),
        "n_success": summary.get("n_success"),
        "n_failed": summary.get("n_failed"),
        "success_rate": summary.get("success_rate"),

        "mean_steps": summary.get("mean_steps"),
        "median_steps": summary.get("median_steps"),
        "min_steps": summary.get("min_steps"),
        "max_steps": summary.get("max_steps"),
        "fraction_full_length": summary.get("fraction_full_length"),

        "mean_displacement_km": summary.get("mean_displacement_km"),
        "median_displacement_km": summary.get("median_displacement_km"),
        "max_displacement_km": summary.get("max_displacement_km"),

        "mean_path_length_km": summary.get("mean_path_length_km"),
        "median_path_length_km": summary.get("median_path_length_km"),
        "max_path_length_km": summary.get("max_path_length_km"),

        "mean_step_length_km": summary.get("mean_step_length_km"),
        "median_step_length_km": summary.get("median_step_length_km"),

        "mean_net_displacement_km": summary.get("mean_net_displacement_km"),
        "median_net_displacement_km": summary.get("median_net_displacement_km"),

        "mean_unique_parents": summary.get("mean_unique_parents"),
        "median_unique_parents": summary.get("median_unique_parents"),
        "parent_entropy_all": summary.get("parent_entropy_all"),

        "step_10_n_members": summary.get("step_10_n_members"),
        "step_10_mean_lat": summary.get("step_10_mean_lat"),
        "step_10_mean_lon": summary.get("step_10_mean_lon"),
        "step_10_mean_radius_km": summary.get("step_10_mean_radius_km"),

        "step_20_n_members": summary.get("step_20_n_members"),
        "step_20_mean_lat": summary.get("step_20_mean_lat"),
        "step_20_mean_lon": summary.get("step_20_mean_lon"),
        "step_20_mean_radius_km": summary.get("step_20_mean_radius_km"),

        "step_30_n_members": summary.get("step_30_n_members"),
        "step_30_mean_lat": summary.get("step_30_mean_lat"),
        "step_30_mean_lon": summary.get("step_30_mean_lon"),
        "step_30_mean_radius_km": summary.get("step_30_mean_radius_km"),
    }

    for wlabel, wstats in summary.get("window_stats", {}).items():
        row[f"{wlabel}_total_integrated_density"] = wstats.get("total_integrated_density", np.nan)

        for vname, val in wstats.get("integrated_density_by_var", {}).items():
            row[f"{wlabel}_integrated_{vname}"] = val

        for vname, val in wstats.get("integrated_density_by_var_norm", {}).items():
            row[f"{wlabel}_integrated_norm_{vname}"] = val

        for vname, val in wstats.get("parent_count_by_var", {}).items():
            row[f"{wlabel}_parent_count_{vname}"] = val

        for vname, val in wstats.get("parent_freq_by_var_norm", {}).items():
            row[f"{wlabel}_parent_freq_norm_{vname}"] = val

        for vname, val in wstats.get("density_top5_fraction_by_var", {}).items():
            row[f"{wlabel}_top5frac_{vname}"] = val

    return row


# -----------------------------------------------------------------------------
# Main hyperparameter search loop
# -----------------------------------------------------------------------------
def run_hyperparam_ensemble_search(
    *,
    case,
    param_samples,
    out_root,
    n_members=100,
    nproc=100,
    start_method="spawn",
    verbose=True,
    windows=((0, 10), (10, 20), (20, 30)),
    window_labels=("days_1_10", "days_11_20", "days_21_30"),
    continue_if_exists=True,
):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    full_data_run = case["full_data"]

    for i, trial in enumerate(param_samples):
        run_id = int(trial.get("run_id", i))
        run_dir = out_root / f"run_{run_id:04d}"

        if continue_if_exists and (run_dir / "summary.json").exists():
            print(f"[skip] run_id={run_id} already exists", flush=True)
            with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
            summary_rows.append(flatten_summary_for_table(summary))
            continue

        params_run = build_params_from_trial_pnw(trial, case)
        params_run["montecarlo"] = True

        print(
            f"\n=== Running hyperparameter set {i + 1}/{len(param_samples)} | "
            f"run_id={run_id} ===",
            flush=True,
        )
        if verbose:
            print(f"params: {params_run}", flush=True)

        ensemble_results_all = run_ensemble_tracks(
            full_data=full_data_run,
            params=params_run,
            date_end=case["date_end"],
            n_members=n_members,
            nproc=nproc,
            start_method=start_method,
            drop_failed=False,
            verbose=verbose,
        )

        summary = summarize_ensemble_results(
            ensemble_results_all,
            params=params_run,
            full_data_run=full_data_run,
            event_lat=case["event_lat"],
            event_lon=case["event_lon"],
            windows=windows,
            window_labels=window_labels,
        )

        save_run_outputs(
            out_root,
            params=params_run,
            summary=summary,
            ensemble_results=ensemble_results_all,
        )

        summary_rows.append(flatten_summary_for_table(summary))

        df = pd.DataFrame(summary_rows)
        atomic_write_csv(out_root / "hyperparam_search_summary.csv", df)
        atomic_write_pickle(out_root / "hyperparam_search_summary.pkl", df)

    return pd.DataFrame(summary_rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="PNW21 TraCE-ST hyperparameter ensemble search"
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output root directory")
    parser.add_argument("--n-trials", type=int, default=500, help="Number of hyperparameter trials")
    parser.add_argument("--n-members", type=int, default=100, help="Ensemble members per trial")
    parser.add_argument("--nproc", type=int, default=100, help="Parallel workers for each ensemble")
    parser.add_argument("--seed", type=int, default=SEARCH_SEED, help="Random seed")
    parser.add_argument(
        "--start-method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method",
    )
    parser.add_argument(
        "--continue-if-exists",
        action="store_true",
        help="Skip runs whose summary.json already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading PNW21 data...", flush=True)
    case = build_case_pnw()
    full_data_run = case["full_data"]

    print(full_data_run, flush=True)
    print("Dims:", full_data_run.dims, flush=True)
    print("Coords:", list(full_data_run.coords), flush=True)
    print("Shape:", list(full_data_run.shape), flush=True)

    print("Sampling hyperparameters...", flush=True)
    jobs = build_pnw_jobs(n_trials=args.n_trials, seed=args.seed)
    param_samples = [trial for _, trial in jobs]

    print(f"Number of hyperparameter samples: {len(param_samples)}", flush=True)
    print("First sample:", param_samples[0], flush=True)

    print("Starting hyperparameter search...", flush=True)
    df_summary = run_hyperparam_ensemble_search(
        case=case,
        param_samples=param_samples,
        out_root=args.outdir,
        n_members=args.n_members,
        nproc=args.nproc,
        start_method=args.start_method,
        verbose=True,
        windows=((0, 10), (10, 20), (20, 30)),
        window_labels=("days_1_10", "days_11_20", "days_21_30"),
        continue_if_exists=args.continue_if_exists,
    )

    print("\nFinished.", flush=True)
    print(df_summary.shape, flush=True)
    print(df_summary.head(), flush=True)

if __name__ == "__main__":
    main()