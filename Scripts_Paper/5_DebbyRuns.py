# Manuscript workflow: causal evolution of Tropical Storm Debby (2006).
#
# TraCE-ST starts from the precipitation target near 28 N, 46 W on
# 26 August 2006 and traces hourly causal influence backward through infrared
# brightness temperature (Tb), mid-level relative vorticity (vo), and
# precipitation fields. Fixed-parameter Monte Carlo ensembles sample competing
# causal pathways; separate hyperparameter trials characterize sensitivity
# within physically admissible spatial and temporal scales. The observed
# IBTrACS storm track is retained as an external physical-reference diagnostic,
# not supplied to the causal-discovery algorithm.

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import copy
import json
import time
import traceback
import argparse
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -----------------------------------------------------------------------------
# Project import
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import trace_st as tst  # noqa: E402

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
GLOBAL_SEED = 11
SEARCH_SEED = 11

# -----------------------------------------------------------------------------
# Debby input paths, target child, and native analysis scales
# -----------------------------------------------------------------------------
PATH_FILES_DEBBY = "/glade/derecho/scratch/jhayron/DataCaStLeBTs/FilesDebby"
PATH_IBTRACS = "/glade/u/home/jhayron/Causality_CaStLe/PaperCausalBTs/Scripts_Paper/IBTrACS.NA.v04r01.nc"

EVENT_LAT_DEBBY = 28.0
EVENT_LON_DEBBY = -46.0
DATE_END_DEBBY = "2006-08-26 00:00:00"

RES_DEBBY = 0.25
BOX_SIZE_DEBBY_BASE = 5.0

# -----------------------------------------------------------------------------
# Longitude, distance, and serialization utilities
# -----------------------------------------------------------------------------
def lon0360_to_m180_180(lon):
    lon = np.asarray(lon, dtype=float) % 360.0
    return ((lon + 180.0) % 360.0) - 180.0


def wrap_lon_diff_deg(lon1, lon2):
    return ((np.asarray(lon1) - np.asarray(lon2) + 180.0) % 360.0) - 180.0


def approx_distance_km(lat1, lon1_deg, lat2, lon2_deg):
    lat1 = np.asarray(lat1, dtype=float)
    lon1_deg = np.asarray(lon1_deg, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2_deg = np.asarray(lon2_deg, dtype=float)

    dlat = lat1 - lat2
    dlon = wrap_lon_diff_deg(lon1_deg, lon2_deg)
    mean_lat = 0.5 * (lat1 + lat2)

    km_per_deg = 111.0
    dx = km_per_deg * np.cos(np.deg2rad(mean_lat)) * dlon
    dy = km_per_deg * dlat
    return np.sqrt(dx**2 + dy**2)


def flatten_params(params):
    flat = {}
    for k, v in params.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    return flat


def canonicalize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): canonicalize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonicalize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

# -----------------------------------------------------------------------------
# Event fields and independent IBTrACS reference track
# -----------------------------------------------------------------------------
def load_debby_data(path_files_debby=PATH_FILES_DEBBY):
    var_tb = xr.open_dataset(f"{path_files_debby}/BT_anomalies_TC_argon.nc4").Tb
    var_vo = xr.open_dataset(f"{path_files_debby}/Vo_anomalies_TC_500-700.nc4").vo
    var_pr = xr.open_dataset(f"{path_files_debby}/P_anomalies_TC_argon2.nc4").precipitation

    # Select the precipitation hour coordinate matching each valid timestamp.
    datalist = []
    for timetemp in var_pr.time:
        hourtemp = pd.to_datetime(timetemp.values).hour
        datatemp = var_pr.sel(time=timetemp, hour=hourtemp)
        datalist.append(datatemp)
    var_pr = xr.concat(datalist, dim="time").fillna(0)

    var_vo = var_vo.rename({"latitude": "lat", "longitude": "lon"})
    var_tb = var_tb.assign_coords(time=var_tb.time.dt.round("1H"))
    var_vo = var_vo.assign_coords(time=var_vo.time.dt.round("1H"))
    var_pr = var_pr.assign_coords(time=var_pr.time.dt.round("1H"))

    full_data = xr.concat(
        [var_tb, var_vo, var_pr],
        dim=xr.IndexVariable("var", ["Tb", "vo", "precip"]),
    )
    full_data = full_data.assign_coords(lon=(full_data.lon % 360)).sortby("lon")
    return full_data


def load_debby_observed_track(
    ibtracs_path=PATH_IBTRACS,
    sid=b"2006234N12338",
    date_end=DATE_END_DEBBY,
):
    ds_tracks = xr.open_dataset(ibtracs_path)

    idx = np.where(ds_tracks.sid == sid)[0][0]
    time_storm = pd.to_datetime(ds_tracks.time.values[idx])
    lat_storm = np.asarray(ds_tracks.lat.values[idx], dtype=float)
    lon_storm = np.asarray(ds_tracks.lon.values[idx], dtype=float)

    valid = (~pd.isna(time_storm)) & np.isfinite(lat_storm) & np.isfinite(lon_storm)
    time_storm = pd.to_datetime(time_storm[valid])
    lat_storm = lat_storm[valid]
    lon_storm = lon_storm[valid]

    date_end_dt = pd.Timestamp(date_end)
    hours_back = (date_end_dt - time_storm) / pd.Timedelta(hours=1)
    keep = hours_back >= 0

    time_storm = time_storm[keep]
    lat_storm = lat_storm[keep]
    lon_storm = lon_storm[keep]
    hours_back = np.asarray(hours_back[keep], dtype=float)

    order = np.argsort(hours_back)
    return dict(
        time=time_storm[order],
        lat=lat_storm[order],
        lon_0360=(lon_storm[order] % 360.0),
        lon_plot=lon0360_to_m180_180(lon_storm[order]),
        hours_back=hours_back[order],
    )


def build_case_debby():
    full_data = load_debby_data()
    obs_track = load_debby_observed_track()

    params_base = dict(
        timeres="1h",
        spaceres=RES_DEBBY,
        box_size=BOX_SIZE_DEBBY_BASE,
        radius=6,
        starting_lat=EVENT_LAT_DEBBY,
        starting_lon=EVENT_LON_DEBBY,
        timewindow="6h",
        child_of_interest=3,  # precip
        n_steps_time=24 * 15,
        averaging_winners=True,
        snap_to_grid=False,
        montecarlo=True,
        beta_softmax=1.0,
        eps_dbscan=0.15,
        min_samples_dbscan=2,
        score_mode="mean",
        gamma=0.5,
        alpha=4.0,
        prefer_sign="both",
        verbose=False,
        prob_rule="softmax",
        winner_summary="mean",
        cd_method="granger",
        cd_kwargs=dict(
            lambda_a=0.1,
            l1_ratio=0.4,
            dependence_threshold=1e-7,
            max_iter=100000,
            fit_intercept=False,
            refit_ridge=False,
            ridge_alpha=1e-2,
        ),
    )

    return dict(
        case_name="debby",
        full_data=full_data,
        date_end=DATE_END_DEBBY,
        params_base=params_base,
        event_lat=EVENT_LAT_DEBBY,
        event_lon_0360=EVENT_LON_DEBBY % 360.0,
        var_names=list(full_data["var"].values),
        observed_track=obs_track,
    )


# -----------------------------------------------------------------------------
# Search space
# Preserve case-specific storm-motion scales for the time window, region, and
# stencil radius while using the common multivariate TraCE-ST parameterization.
# -----------------------------------------------------------------------------
SEARCH_SPACE_SHARED_DEBBY = dict(
    timewindow=["5h", "6h", "7h"],
    box_size=[5.0, 6.0, 7.0],
    radius=[4, 5, 6],
    eps_dbscan=[0.05, 0.10, 0.15, 0.20, 0.25],
    min_samples_dbscan=[2],
    score_mode=["mean", "sum"],
    winner_summary=["mean"],
    alpha=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    prefer_sign=["both"],
)

SEARCH_SPACE_PROB_DEBBY = dict(
    prob_rule=["linear", "softmax"],
    beta_softmax=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
)

SEARCH_SPACE_GRANGER_DEBBY = dict(
    lambda_a=lambda rng: float(10 ** rng.uniform(-2.5, -1.0)),
    l1_ratio=lambda rng: float(10 ** rng.uniform(-1.0, 0.0)),
    dependence_threshold=[1e-7],
    max_iter=[100000],
    fit_intercept=[False],
    refit_ridge=[False],
    ridge_alpha=[1e-2],
)

SEARCH_SPACE_EXTRA_DEBBY = dict(
    gamma=[0.5],   
)


def _draw_from_space(rng, x):
    if callable(x):
        return x(rng)
    v = rng.choice(x)
    if isinstance(v, np.str_):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def sample_debby_trial(rng):
    trial = {}
    for space in [
        SEARCH_SPACE_SHARED_DEBBY,
        SEARCH_SPACE_PROB_DEBBY,
        SEARCH_SPACE_GRANGER_DEBBY,
        SEARCH_SPACE_EXTRA_DEBBY,
    ]:
        for k, v in space.items():
            trial[k] = _draw_from_space(rng, v)
    return trial


def build_debby_jobs(n_trials, seed=SEARCH_SEED):
    rng = np.random.default_rng(seed)
    return [(trial_id, sample_debby_trial(rng)) for trial_id in range(n_trials)]


def build_params_from_trial_debby(trial, case):
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

    return p


# -----------------------------------------------------------------------------
# Single-trajectory and fixed-parameter Monte Carlo ensemble execution
# -----------------------------------------------------------------------------
def run_one_track(full_data, params, date_end):
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


def run_ensemble(case, params, M=30, seed=SEARCH_SEED):
    rng_local = np.random.default_rng(seed)
    members = []

    for m in range(M):
        p = copy.deepcopy(params)
        p["starting_lat"] = float(case["event_lat"])
        p["starting_lon"] = float(case["event_lon_0360"])

        np.random.seed(int(rng_local.integers(0, 2_000_000_000)))
        t0 = time.perf_counter()

        try:
            centers, parents, out = run_one_track(case["full_data"], p, case["date_end"])
            ok = True
            err = ""
        except Exception as e:
            centers = np.empty((0, 2), dtype=float)
            parents = np.empty((0,), dtype=int)
            out = None
            ok = False
            err = str(e)

        members.append({
            "member_id": m,
            "ok": ok,
            "error": err,
            "elapsed_sec": float(time.perf_counter() - t0),
            "centers": np.asarray(centers, dtype=float),
            "parents": np.asarray(parents, dtype=int),
            "out": out,
        })

    return {
        "members": members,
        "params": copy.deepcopy(params),
        "M": int(M),
    }


# -----------------------------------------------------------------------------
# Interpolation to the independent storm track and distance diagnostics
# -----------------------------------------------------------------------------
def get_timeres_hours(timeres: str) -> float:
    timeres = str(timeres).strip().lower()
    if timeres.endswith("h"):
        return float(timeres[:-1])
    if timeres.endswith("d"):
        return 24.0 * float(timeres[:-1])
    raise ValueError(f"Unsupported timeres: {timeres}")


def interpolate_observed_track_lonlat(query_hours_back, obs_track):
    q = np.asarray(query_hours_back, dtype=float)
    h = np.asarray(obs_track["hours_back"], dtype=float)
    lat = np.asarray(obs_track["lat"], dtype=float)
    lon = np.asarray(obs_track["lon_0360"], dtype=float)

    valid = np.isfinite(h) & np.isfinite(lat) & np.isfinite(lon)
    h = h[valid]
    lat = lat[valid]
    lon = lon[valid]

    if len(h) < 2:
        return (
            np.full_like(q, np.nan, dtype=float),
            np.full_like(q, np.nan, dtype=float),
            np.zeros_like(q, dtype=bool),
        )

    order = np.argsort(h)
    h = h[order]
    lat = lat[order]
    lon = lon[order]

    lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon)))
    lat_i = np.interp(q, h, lat, left=np.nan, right=np.nan)
    lon_i_unwrapped = np.interp(q, h, lon_unwrapped, left=np.nan, right=np.nan)
    lon_i = lon_i_unwrapped % 360.0

    valid_q = (q >= np.nanmin(h)) & (q <= np.nanmax(h))
    lat_i[~valid_q] = np.nan
    lon_i[~valid_q] = np.nan

    return lat_i, lon_i, valid_q


def debby_member_track_distances_km(case, centers):
    """
    Compare one trajectory to the observed Debby track.

    If the trajectory ends early, the last available model point
    is used to compute distances against the remaining observed track.
    """
    centers = np.asarray(centers, dtype=float)
    n_steps = len(centers)

    if n_steps == 0:
        return np.array([]), np.array([], dtype=bool), np.array([])

    dt_hours = get_timeres_hours(case["params_base"]["timeres"])

    obs = case["observed_track"]
    max_obs_hour = float(np.nanmax(obs["hours_back"]))
    obs_hours_back = np.arange(0.0, max_obs_hour + 1e-9, dt_hours)

    obs_lat_i, obs_lon_i, valid_obs = interpolate_observed_track_lonlat(
        obs_hours_back,
        obs,
    )

    if len(obs_hours_back) <= n_steps:
        pred_centers_cmp = centers[:len(obs_hours_back)]
    else:
        pred_centers_cmp = np.vstack([
            centers,
            np.repeat(centers[-1:, :], len(obs_hours_back) - n_steps, axis=0),
        ])

    pred_lat = pred_centers_cmp[:, 0]
    pred_lon = pred_centers_cmp[:, 1] % 360.0

    valid = valid_obs & np.isfinite(pred_lat) & np.isfinite(pred_lon)
    dist_km = np.full(len(obs_hours_back), np.nan, dtype=float)
    dist_km[valid] = approx_distance_km(
        pred_lat[valid],
        lon0360_to_m180_180(pred_lon[valid]),
        obs_lat_i[valid],
        lon0360_to_m180_180(obs_lon_i[valid]),
    )

    return dist_km, valid, obs_hours_back


def pairwise_step_lengths(centers):
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        return np.array([], dtype=float)
    diffs = np.diff(centers, axis=0)
    return np.sqrt(np.sum(diffs**2, axis=1))


def _member_geometry_stats_debby(centers, parents):
    centers = np.asarray(centers, dtype=float)
    parents = np.asarray(parents, dtype=int)

    if len(centers) == 0 or len(parents) == 0:
        return dict(
            n_steps=0,
            path_length=np.nan,
            net_displacement=np.nan,
            mean_step=np.nan,
            stagnant_frac=np.nan,
            end_parent=np.nan,
            frac_parent1=np.nan,
            frac_parent2=np.nan,
            frac_parent3=np.nan,
        )

    steps = pairwise_step_lengths(centers)
    path_length = float(np.sum(steps)) if len(steps) else 0.0
    net_displacement = (
        float(np.sqrt(np.sum((centers[-1] - centers[0]) ** 2)))
        if len(centers) >= 2 else 0.0
    )
    mean_step = float(np.mean(steps)) if len(steps) else 0.0
    stagnant_frac = float(np.mean(steps < 1e-6)) if len(steps) else 1.0

    return dict(
        n_steps=int(len(parents)),
        path_length=path_length,
        net_displacement=net_displacement,
        mean_step=mean_step,
        stagnant_frac=stagnant_frac,
        end_parent=int(parents[-1]),
        frac_parent1=float(np.mean(parents == 1)),
        frac_parent2=float(np.mean(parents == 2)),
        frac_parent3=float(np.mean(parents == 3)),
    )


# -----------------------------------------------------------------------------
# Variable-specific spatial trajectory-density accumulation
# -----------------------------------------------------------------------------
def _accumulate_density_box(density, xs, ys, centers, box_size):
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 0:
        return density

    half = float(box_size) / 2.0

    for yc, xc in centers:
        x0 = xc - half
        x1 = xc + half
        y0 = yc - half
        y1 = yc + half

        xmask = (xs >= x0) & (xs <= x1)
        ymask = (ys >= y0) & (ys <= y1)

        if np.any(xmask) and np.any(ymask):
            density[np.ix_(ymask, xmask)] += 1.0

    return density


def _integrated_density_by_parent(
    density_by_parent,
    start_step=None,
    end_step_exclusive=None,
    density_steps_by_parent=None,
):
    if density_steps_by_parent is None or start_step is None or end_step_exclusive is None:
        return {
            1: float(np.nansum(density_by_parent[1])),
            2: float(np.nansum(density_by_parent[2])),
            3: float(np.nansum(density_by_parent[3])),
        }

    out = {}
    for p in [1, 2, 3]:
        dens = np.zeros_like(density_by_parent[p], dtype=float)
        for step_i, arr in density_steps_by_parent[p]:
            if step_i >= start_step and step_i < end_step_exclusive:
                dens += arr
        out[p] = float(np.nansum(dens))
    return out


# -----------------------------------------------------------------------------
# Causal-contribution, track-distance, completion, and geometry diagnostics
# -----------------------------------------------------------------------------
def summarize_ensemble_debby(
    case,
    ensemble_out,
    params,
    density_range_start_hour=96.0,
    density_range_end_hour=360.0,
):
    xs = np.asarray(case["full_data"].lon.values, dtype=float)
    ys = np.asarray(case["full_data"].lat.values, dtype=float)
    shape2d = (len(ys), len(xs))
    box_size = float(params["box_size"])
    dt_hours = get_timeres_hours(params["timeres"])

    density_by_parent = {
        1: np.zeros(shape2d, dtype=float),
        2: np.zeros(shape2d, dtype=float),
        3: np.zeros(shape2d, dtype=float),
    }
    density_steps_by_parent = {1: [], 2: [], 3: []}

    end_parent_counts = {1: 0, 2: 0, 3: 0}
    n_ok = 0
    rows = []

    final_points = []
    final_parents = []

    for member in ensemble_out["members"]:
        member_id = int(member["member_id"])

        if not member["ok"]:
            rows.append({
                "member_id": member_id,
                "ok": 0,
                "error": member.get("error", ""),
                "track_dist_km_mean": np.nan,
                "track_dist_km_median": np.nan,
                "track_dist_km_max": np.nan,
                "track_dist_km_final": np.nan,
                "n_valid_track_points": 0,
                **_member_geometry_stats_debby([], []),
            })
            continue

        centers = np.asarray(member["centers"], dtype=float)
        parents = np.asarray(member["parents"], dtype=int)

        if len(centers) == 0 or len(parents) == 0:
            rows.append({
                "member_id": member_id,
                "ok": 0,
                "error": member.get("error", ""),
                "track_dist_km_mean": np.nan,
                "track_dist_km_median": np.nan,
                "track_dist_km_max": np.nan,
                "track_dist_km_final": np.nan,
                "n_valid_track_points": 0,
                **_member_geometry_stats_debby([], []),
            })
            continue

        n_ok += 1
        stats = _member_geometry_stats_debby(centers, parents)

        end_parent = stats["end_parent"]
        if end_parent in end_parent_counts:
            end_parent_counts[end_parent] += 1

        dist_km, valid_track, obs_hours_back = debby_member_track_distances_km(case, centers)

        # accumulate density by full box footprint
        for step_i, (center, p) in enumerate(zip(centers, parents)):
            arr = np.zeros(shape2d, dtype=float)
            arr = _accumulate_density_box(arr, xs, ys, np.asarray([center]), box_size)
            density_by_parent[int(p)] += arr
            density_steps_by_parent[int(p)].append((step_i, arr))

        final_points.append(centers[-1].copy())
        final_parents.append(int(parents[-1]))

        valid_dist = dist_km[np.isfinite(dist_km)]
        rows.append({
            "member_id": member_id,
            "ok": 1,
            "error": "",
            "track_dist_km_mean": float(np.nanmean(valid_dist)) if len(valid_dist) else np.nan,
            "track_dist_km_median": float(np.nanmedian(valid_dist)) if len(valid_dist) else np.nan,
            "track_dist_km_max": float(np.nanmax(valid_dist)) if len(valid_dist) else np.nan,
            "track_dist_km_final": float(valid_dist[-1]) if len(valid_dist) else np.nan,
            "n_valid_track_points": int(np.sum(np.isfinite(dist_km))),
            **stats,
        })

    df_members = pd.DataFrame(rows)

    total_end = sum(end_parent_counts.values())
    end_parent_frac = {
        p: (end_parent_counts[p] / total_end if total_end > 0 else np.nan)
        for p in [1, 2, 3]
    }

    density_full = _integrated_density_by_parent(
        density_by_parent=density_by_parent,
        density_steps_by_parent=None,
    )

    start_step = int(np.floor(density_range_start_hour / dt_hours))
    end_step_exclusive = int(np.ceil(density_range_end_hour / dt_hours))

    density_range = _integrated_density_by_parent(
        density_by_parent=density_by_parent,
        start_step=start_step,
        end_step_exclusive=end_step_exclusive,
        density_steps_by_parent=density_steps_by_parent,
    )

    valid_members = df_members[df_members["ok"] == 1].copy()

    def _safe_mean(col):
        return float(valid_members[col].mean()) if len(valid_members) else np.nan

    def _safe_median(col):
        return float(valid_members[col].median()) if len(valid_members) else np.nan

    def _safe_std(col):
        return float(valid_members[col].std(ddof=0)) if len(valid_members) else np.nan

    def _safe_min(col):
        return float(valid_members[col].min()) if len(valid_members) else np.nan

    # final-point spread
    if len(final_points):
        final_points_arr = np.asarray(final_points, dtype=float)
        final_lat = final_points_arr[:, 0]
        final_lon = final_points_arr[:, 1] % 360.0

        mean_final_lat = float(np.nanmean(final_lat))
        mean_final_lon = float(np.nanmean(final_lon))
        std_final_lat = float(np.nanstd(final_lat))
        std_final_lon = float(np.nanstd(lon0360_to_m180_180(final_lon)))

        dist_to_centroid_km = approx_distance_km(
            final_lat,
            lon0360_to_m180_180(final_lon),
            mean_final_lat,
            lon0360_to_m180_180(mean_final_lon),
        )
        std_final_dist_to_centroid_km = float(np.nanstd(dist_to_centroid_km))
    else:
        mean_final_lat = np.nan
        mean_final_lon = np.nan
        std_final_lat = np.nan
        std_final_lon = np.nan
        std_final_dist_to_centroid_km = np.nan

    # simple north/south split diagnostic
    if len(final_points):
        split_lat = float(np.nanmedian(np.asarray(final_points)[:, 0]))
        north_mask = np.asarray(final_points)[:, 0] >= split_lat
        south_mask = ~north_mask

        pct_final_points_north_branch = float(np.mean(north_mask))
        pct_final_points_south_branch = float(np.mean(south_mask))

        north_branch_mean_lat = float(np.nanmean(np.asarray(final_points)[north_mask, 0])) if np.any(north_mask) else np.nan
        south_branch_mean_lat = float(np.nanmean(np.asarray(final_points)[south_mask, 0])) if np.any(south_mask) else np.nan
        north_branch_mean_lon = float(np.nanmean(np.asarray(final_points)[north_mask, 1] % 360.0)) if np.any(north_mask) else np.nan
        south_branch_mean_lon = float(np.nanmean(np.asarray(final_points)[south_mask, 1] % 360.0)) if np.any(south_mask) else np.nan

        if np.any(north_mask) and np.any(south_mask):
            branch_centroid_separation_km = float(approx_distance_km(
                north_branch_mean_lat,
                lon0360_to_m180_180(north_branch_mean_lon),
                south_branch_mean_lat,
                lon0360_to_m180_180(south_branch_mean_lon),
            ))
        else:
            branch_centroid_separation_km = np.nan
    else:
        split_lat = np.nan
        pct_final_points_north_branch = np.nan
        pct_final_points_south_branch = np.nan
        north_branch_mean_lat = np.nan
        south_branch_mean_lat = np.nan
        north_branch_mean_lon = np.nan
        south_branch_mean_lon = np.nan
        branch_centroid_separation_km = np.nan

    # final variable diagnostics
    if len(final_parents):
        final_parents_arr = np.asarray(final_parents, dtype=int)
        counts = pd.Series(final_parents_arr).value_counts()
        most_repeated_parent_final = int(counts.index[0])
        most_repeated_var_final = case["var_names"][most_repeated_parent_final - 1]
        pct_traj_ended_on_most_repeated_var = float(np.mean(final_parents_arr == most_repeated_parent_final))
        pct_traj_ended_on_Tb = float(np.mean(final_parents_arr == 1))
        pct_traj_ended_on_vo = float(np.mean(final_parents_arr == 2))
        pct_traj_ended_on_precip = float(np.mean(final_parents_arr == 3))
    else:
        most_repeated_parent_final = np.nan
        most_repeated_var_final = np.nan
        pct_traj_ended_on_most_repeated_var = np.nan
        pct_traj_ended_on_Tb = np.nan
        pct_traj_ended_on_vo = np.nan
        pct_traj_ended_on_precip = np.nan

    summary = {
        "n_members_requested": int(len(ensemble_out["members"])),
        "n_members_success": int(n_ok),
        "n_members_failed": int(len(ensemble_out["members"]) - n_ok),
        "success_rate": float(n_ok / len(ensemble_out["members"])) if len(ensemble_out["members"]) else np.nan,

        "density_by_parent": density_by_parent,
        "density_full": density_full,
        "density_range": density_range,
        "density_range_start_hour": float(density_range_start_hour),
        "density_range_end_hour": float(density_range_end_hour),
        "density_range_start_step": int(start_step),
        "density_range_end_step_exclusive": int(end_step_exclusive),

        "end_parent_counts": end_parent_counts,
        "end_parent_frac": end_parent_frac,

        "mean_traj_length": _safe_mean("n_steps"),
        "median_traj_length": _safe_median("n_steps"),
        "std_traj_length": _safe_std("n_steps"),
        "min_traj_length": _safe_min("n_steps"),
        "max_traj_length": float(valid_members["n_steps"].max()) if len(valid_members) else np.nan,

        "mean_of_member_mean_track_dist_km": _safe_mean("track_dist_km_mean"),
        "median_of_member_mean_track_dist_km": _safe_median("track_dist_km_mean"),
        "mean_of_member_median_track_dist_km": _safe_mean("track_dist_km_median"),
        "median_of_member_median_track_dist_km": _safe_median("track_dist_km_median"),

        "mean_final_lat": mean_final_lat,
        "mean_final_lon": mean_final_lon,
        "std_final_lat": std_final_lat,
        "std_final_lon": std_final_lon,
        "std_final_dist_to_centroid_km": std_final_dist_to_centroid_km,

        "final_lat_split": split_lat,
        "pct_final_points_north_branch": pct_final_points_north_branch,
        "pct_final_points_south_branch": pct_final_points_south_branch,
        "north_branch_mean_lat": north_branch_mean_lat,
        "south_branch_mean_lat": south_branch_mean_lat,
        "north_branch_mean_lon": north_branch_mean_lon,
        "south_branch_mean_lon": south_branch_mean_lon,
        "branch_centroid_separation_km": branch_centroid_separation_km,

        "most_repeated_parent_final": most_repeated_parent_final,
        "most_repeated_var_final": most_repeated_var_final,
        "pct_traj_ended_on_most_repeated_var": pct_traj_ended_on_most_repeated_var,
        "pct_traj_ended_on_Tb": pct_traj_ended_on_Tb,
        "pct_traj_ended_on_vo": pct_traj_ended_on_vo,
        "pct_traj_ended_on_precip": pct_traj_ended_on_precip,

        "n_members_with_valid_track": int(len(valid_members)),
        "df_members": df_members,
    }
    return summary


def summarize_trial_debby(case, trial_id, params, summary):
    row = {
        "case_name": case["case_name"],
        "trial_id": int(trial_id),
        **flatten_params(params),

        "n_members_requested": summary["n_members_requested"],
        "n_members_success": summary["n_members_success"],
        "n_members_failed": summary["n_members_failed"],
        "success_rate": summary["success_rate"],

        "mean_traj_length": summary["mean_traj_length"],
        "median_traj_length": summary["median_traj_length"],
        "std_traj_length": summary["std_traj_length"],
        "min_traj_length": summary["min_traj_length"],
        "max_traj_length": summary["max_traj_length"],

        "n_members_with_valid_track": summary["n_members_with_valid_track"],
        "mean_of_member_mean_track_dist_km": summary["mean_of_member_mean_track_dist_km"],
        "median_of_member_mean_track_dist_km": summary["median_of_member_mean_track_dist_km"],
        "mean_of_member_median_track_dist_km": summary["mean_of_member_median_track_dist_km"],
        "median_of_member_median_track_dist_km": summary["median_of_member_median_track_dist_km"],

        "mean_final_lat": summary["mean_final_lat"],
        "mean_final_lon": summary["mean_final_lon"],
        "std_final_lat": summary["std_final_lat"],
        "std_final_lon": summary["std_final_lon"],
        "std_final_dist_to_centroid_km": summary["std_final_dist_to_centroid_km"],

        "final_lat_split": summary["final_lat_split"],
        "pct_final_points_north_branch": summary["pct_final_points_north_branch"],
        "pct_final_points_south_branch": summary["pct_final_points_south_branch"],
        "north_branch_mean_lat": summary["north_branch_mean_lat"],
        "south_branch_mean_lat": summary["south_branch_mean_lat"],
        "north_branch_mean_lon": summary["north_branch_mean_lon"],
        "south_branch_mean_lon": summary["south_branch_mean_lon"],
        "branch_centroid_separation_km": summary["branch_centroid_separation_km"],

        "most_repeated_parent_final": summary["most_repeated_parent_final"],
        "most_repeated_var_final": summary["most_repeated_var_final"],
        "pct_traj_ended_on_most_repeated_var": summary["pct_traj_ended_on_most_repeated_var"],
        "pct_traj_ended_on_Tb": summary["pct_traj_ended_on_Tb"],
        "pct_traj_ended_on_vo": summary["pct_traj_ended_on_vo"],
        "pct_traj_ended_on_precip": summary["pct_traj_ended_on_precip"],

        "density_full::Tb": summary["density_full"][1],
        "density_full::vo": summary["density_full"][2],
        "density_full::precip": summary["density_full"][3],

        "density_range_start_hour": summary["density_range_start_hour"],
        "density_range_end_hour": summary["density_range_end_hour"],
        "density_range_start_step": summary["density_range_start_step"],
        "density_range_end_step_exclusive": summary["density_range_end_step_exclusive"],
        "density_range::Tb": summary["density_range"][1],
        "density_range::vo": summary["density_range"][2],
        "density_range::precip": summary["density_range"][3],
    }
    return row


# -----------------------------------------------------------------------------
# Full ensemble artifacts used by downstream manuscript analysis
# -----------------------------------------------------------------------------
def _bundle_stem_debby(trial_id):
    return f"debby_trial_{int(trial_id):05d}"


def serialize_ensemble_bundle_debby(trial_id, trial, case, params, ensemble_out, summary):
    members = ensemble_out["members"]
    M = len(members)

    lengths = np.array([len(np.asarray(m["parents"])) for m in members], dtype=np.int32)
    max_len = int(lengths.max()) if len(lengths) else 0

    centers_pad = np.full((M, max_len, 2), np.nan, dtype=np.float32)
    parents_pad = np.full((M, max_len), -1, dtype=np.int16)
    ok_arr = np.zeros(M, dtype=np.int8)
    elapsed_arr = np.full(M, np.nan, dtype=np.float32)
    member_id_arr = np.full(M, -1, dtype=np.int32)
    error_list = []

    for i, m in enumerate(members):
        member_id_arr[i] = int(m["member_id"])
        ok_arr[i] = int(bool(m["ok"]))
        elapsed_arr[i] = float(m.get("elapsed_sec", np.nan))
        error_list.append(str(m.get("error", "")))

        centers = np.asarray(m["centers"], dtype=float)
        parents = np.asarray(m["parents"], dtype=int)
        L = len(parents)

        if L > 0:
            centers_pad[i, :L, :] = centers.astype(np.float32)
            parents_pad[i, :L] = parents.astype(np.int16)

    meta = {
        "case_name": case["case_name"],
        "trial_id": int(trial_id),
        "trial": canonicalize_for_json(trial),
        "params": canonicalize_for_json(params),
        "summary_small": {
            "n_members_requested": int(summary["n_members_requested"]),
            "n_members_success": int(summary["n_members_success"]),
            "n_members_failed": int(summary["n_members_failed"]),
            "success_rate": float(summary["success_rate"]),
            "mean_traj_length": summary["mean_traj_length"],
            "min_traj_length": summary["min_traj_length"],
            "mean_of_member_mean_track_dist_km": summary["mean_of_member_mean_track_dist_km"],
            "most_repeated_var_final": summary["most_repeated_var_final"],
            "pct_traj_ended_on_Tb": summary["pct_traj_ended_on_Tb"],
            "pct_traj_ended_on_vo": summary["pct_traj_ended_on_vo"],
            "pct_traj_ended_on_precip": summary["pct_traj_ended_on_precip"],
            "density_full": canonicalize_for_json(summary["density_full"]),
            "density_range": canonicalize_for_json(summary["density_range"]),
        },
        "errors": error_list,
    }

    arrays = {
        "centers": centers_pad,
        "parents": parents_pad,
        "lengths": lengths,
        "ok": ok_arr,
        "elapsed_sec": elapsed_arr,
        "member_id": member_id_arr,
        "density_parent1": summary["density_by_parent"][1].astype(np.float32),
        "density_parent2": summary["density_by_parent"][2].astype(np.float32),
        "density_parent3": summary["density_by_parent"][3].astype(np.float32),
        "xs": np.asarray(case["full_data"].lon.values, dtype=np.float32),
        "ys": np.asarray(case["full_data"].lat.values, dtype=np.float32),
        "obs_hours_back": np.asarray(case["observed_track"]["hours_back"], dtype=np.float32),
        "obs_lat": np.asarray(case["observed_track"]["lat"], dtype=np.float32),
        "obs_lon_0360": np.asarray(case["observed_track"]["lon_0360"], dtype=np.float32),
    }

    return {"meta": meta, "arrays": arrays}


def save_ensemble_bundle_debby(bundle, bundles_dir: Path):
    bundles_dir.mkdir(parents=True, exist_ok=True)

    trial_id = int(bundle["meta"]["trial_id"])
    stem = _bundle_stem_debby(trial_id)

    npz_path = bundles_dir / f"{stem}.npz"
    json_path = bundles_dir / f"{stem}.json"

    np.savez_compressed(npz_path, **bundle["arrays"])
    atomic_write_json(json_path, bundle["meta"])

    return npz_path, json_path


# -----------------------------------------------------------------------------
# Incremental output and restart support for long searches
# -----------------------------------------------------------------------------
def append_jsonl(path: Path, records):
    if not records:
        return
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_write_csv(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(path)


def load_jsonl_records(path: Path):
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                print(f"Warning: could not parse line {line_num} in {path}. Skipping.", flush=True)
    return records


def load_completed_trial_ids(jobs_jsonl: Path):
    records = load_jsonl_records(jobs_jsonl)
    completed = set()
    for r in records:
        try:
            trial_id = int(r["trial_id"])
            completed.add(trial_id)
        except Exception:
            continue
    return completed, records


def filter_pending_trials(all_jobs, completed_ids):
    pending = []
    for job in all_jobs:
        trial_id, _ = job
        if int(trial_id) not in completed_ids:
            pending.append(job)
    return pending


# -----------------------------------------------------------------------------
# Evaluate one physically admissible parameter configuration
# -----------------------------------------------------------------------------
def evaluate_trial_debby(trial_id, trial, ensemble_size=30, seed=SEARCH_SEED):
    case = build_case_debby()
    params = build_params_from_trial_debby(trial, case)

    ensemble_out = run_ensemble(
        case,
        params,
        M=ensemble_size,
        seed=int(seed + 1000 * int(trial_id)),
    )

    summary = summarize_ensemble_debby(
        case,
        ensemble_out,
        params,
        density_range_start_hour=96.0,
        density_range_end_hour=360.0,
    )

    trial_row = summarize_trial_debby(case, trial_id, params, summary)

    df_members = summary["df_members"].copy()
    df_members["case_name"] = case["case_name"]
    df_members["trial_id"] = int(trial_id)
    for k, v in flatten_params(params).items():
        df_members[k] = v

    bundle = serialize_ensemble_bundle_debby(
        trial_id=trial_id,
        trial=trial,
        case=case,
        params=params,
        ensemble_out=ensemble_out,
        summary=summary,
    )

    return {
        "trial_row": trial_row,
        "member_rows": df_members.to_dict(orient="records"),
        "bundle": bundle,
    }


# -----------------------------------------------------------------------------
# Multiprocessing wrapper for one parameter configuration
# -----------------------------------------------------------------------------
def evaluate_trial_worker(job, ensemble_size=30, seed=SEARCH_SEED):
    trial_id, trial = job
    t0 = time.perf_counter()

    try:
        out = evaluate_trial_debby(
            trial_id=trial_id,
            trial=trial,
            ensemble_size=ensemble_size,
            seed=seed,
        )

        trial_row = out["trial_row"]
        trial_row["elapsed_sec"] = float(time.perf_counter() - t0)
        trial_row["status"] = "ok"
        trial_row["ok"] = 1
        trial_row["error"] = ""

        return {
            "ok": True,
            "trial_row": trial_row,
            "member_rows": out["member_rows"],
            "bundle": out["bundle"],
        }

    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()

        trial_row = {
            "case_name": "debby",
            "trial_id": int(trial_id),
            **trial,
        
            "n_members_requested": np.nan,
            "n_members_success": np.nan,
            "n_members_failed": np.nan,
            "success_rate": np.nan,
        
            "mean_traj_length": np.nan,
            "median_traj_length": np.nan,
            "std_traj_length": np.nan,
            "min_traj_length": np.nan,
            "max_traj_length": np.nan,
        
            "n_members_with_valid_track": np.nan,
            "mean_of_member_mean_track_dist_km": np.nan,
            "median_of_member_mean_track_dist_km": np.nan,
            "mean_of_member_median_track_dist_km": np.nan,
            "median_of_member_median_track_dist_km": np.nan,
        
            "mean_final_lat": np.nan,
            "mean_final_lon": np.nan,
            "std_final_lat": np.nan,
            "std_final_lon": np.nan,
            "std_final_dist_to_centroid_km": np.nan,
        
            "final_lat_split": np.nan,
            "pct_final_points_north_branch": np.nan,
            "pct_final_points_south_branch": np.nan,
            "north_branch_mean_lat": np.nan,
            "south_branch_mean_lat": np.nan,
            "north_branch_mean_lon": np.nan,
            "south_branch_mean_lon": np.nan,
            "branch_centroid_separation_km": np.nan,
        
            "most_repeated_parent_final": np.nan,
            "most_repeated_var_final": np.nan,
            "pct_traj_ended_on_most_repeated_var": np.nan,
            "pct_traj_ended_on_Tb": np.nan,
            "pct_traj_ended_on_vo": np.nan,
            "pct_traj_ended_on_precip": np.nan,
        
            "density_full::Tb": np.nan,
            "density_full::vo": np.nan,
            "density_full::precip": np.nan,
        
            "density_range_start_hour": np.nan,
            "density_range_end_hour": np.nan,
            "density_range_start_step": np.nan,
            "density_range_end_step_exclusive": np.nan,
            "density_range::Tb": np.nan,
            "density_range::vo": np.nan,
            "density_range::precip": np.nan,
        
            "status": "failed",
            "ok": 0,
            "error": err,
            "elapsed_sec": float(time.perf_counter() - t0),
        }

        return {
            "ok": False,
            "trial_row": trial_row,
            "member_rows": [],
            "bundle": None,
        }


# -----------------------------------------------------------------------------
# Command-line orchestration and multiprocessing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Trace-ST Debby hyperparameter search")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--n-trials", type=int, default=1000, help="Number of hyperparameter trials")
    parser.add_argument("--n-workers", type=int, default=120, help="Parallel workers")
    parser.add_argument("--ensemble-size", type=int, default=30, help="Ensemble members per trial")
    parser.add_argument("--seed", type=int, default=SEARCH_SEED, help="Search seed")
    parser.add_argument("--flush-every", type=int, default=10, help="Write snapshot every N completed trials")
    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundles_dir = outdir / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)

    jobs_jsonl = outdir / "results_jobs.jsonl"
    members_jsonl = outdir / "results_members.jsonl"

    jobs_csv = outdir / "results_jobs_snapshot.csv"
    members_csv = outdir / "results_members_snapshot.csv"

    meta_path = outdir / "run_metadata.json"
    summary_path = outdir / "run_summary.json"

    all_jobs = build_debby_jobs(n_trials=args.n_trials, seed=args.seed)

    completed_ids, existing_job_rows = load_completed_trial_ids(jobs_jsonl)
    pending_jobs = filter_pending_trials(all_jobs, completed_ids)

    existing_member_rows = load_jsonl_records(members_jsonl)

    meta = {
        "case_name": "debby",
        "seed": int(args.seed),
        "n_trials": int(args.n_trials),
        "n_workers": int(args.n_workers),
        "ensemble_size": int(args.ensemble_size),
        "flush_every": int(args.flush_every),
        "n_total_trials": len(all_jobs),
        "n_completed_trials_found_at_start": len(completed_ids),
        "n_pending_trials_at_start": len(pending_jobs),
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }
    atomic_write_json(meta_path, meta)

    if existing_job_rows:
        atomic_write_csv(jobs_csv, pd.DataFrame(existing_job_rows))
    if existing_member_rows:
        atomic_write_csv(members_csv, pd.DataFrame(existing_member_rows))

    print(f"Total trials: {len(all_jobs)}", flush=True)
    print(f"Completed:    {len(completed_ids)}", flush=True)
    print(f"Pending:      {len(pending_jobs)}", flush=True)

    if len(pending_jobs) == 0:
        print("Nothing to do. All trials already completed.", flush=True)

        final_jobs = pd.DataFrame(existing_job_rows)
        final_members = pd.DataFrame(existing_member_rows)

        if len(final_jobs) > 0:
            sort_cols = [c for c in [
                "ok",
                "mean_of_member_mean_track_dist_km",
                "median_of_member_mean_track_dist_km",
                "mean_traj_length",
            ] if c in final_jobs.columns]
            
            asc = [(False if c == "ok" else True) for c in sort_cols]
            if len(sort_cols):
                final_jobs = final_jobs.sort_values(sort_cols, ascending=asc)

            atomic_write_csv(outdir / "results_jobs_final.csv", final_jobs)

        if len(final_members) > 0:
            atomic_write_csv(outdir / "results_members_final.csv", final_members)

        summary = {
            "finished_at": pd.Timestamp.utcnow().isoformat(),
            "elapsed_sec": 0.0,
            "n_trials_total": len(all_jobs),
            "n_trials_ok": int((final_jobs["ok"] == 1).sum()) if len(final_jobs) and "ok" in final_jobs.columns else 0,
            "n_trials_failed": int((final_jobs["ok"] == 0).sum()) if len(final_jobs) and "ok" in final_jobs.columns else 0,
            "resumed": True,
            "nothing_new_run": True,
        }
        atomic_write_json(summary_path, summary)
        return

    buffer_trial_rows = []
    buffer_member_rows = []

    t0 = time.perf_counter()
    n_done_this_run = 0
    n_total_done_now = len(completed_ids)

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as ex:
        futures = [
            ex.submit(
                evaluate_trial_worker,
                job,
                args.ensemble_size,
                args.seed,
            )
            for job in pending_jobs
        ]

        for fut in as_completed(futures):
            result = fut.result()

            trial_row = result["trial_row"]
            member_rows = result["member_rows"]
            bundle = result["bundle"]

            if bundle is not None:
                npz_path, json_path = save_ensemble_bundle_debby(bundle, bundles_dir=bundles_dir)
                trial_row["bundle_npz"] = str(npz_path)
                trial_row["bundle_json"] = str(json_path)

                for row in member_rows:
                    row["bundle_npz"] = str(npz_path)
                    row["bundle_json"] = str(json_path)

            buffer_trial_rows.append(trial_row)
            buffer_member_rows.extend(member_rows)

            n_done_this_run += 1
            n_total_done_now += 1

            if (n_done_this_run % args.flush_every == 0) or (n_done_this_run == len(pending_jobs)):
                append_jsonl(jobs_jsonl, buffer_trial_rows)
                append_jsonl(members_jsonl, buffer_member_rows)

                all_job_rows_now = load_jsonl_records(jobs_jsonl)
                all_member_rows_now = load_jsonl_records(members_jsonl)

                df_jobs_now = pd.DataFrame(all_job_rows_now) if all_job_rows_now else pd.DataFrame()
                df_members_now = pd.DataFrame(all_member_rows_now) if all_member_rows_now else pd.DataFrame()

                if len(df_jobs_now) > 0:
                    atomic_write_csv(jobs_csv, df_jobs_now)
                if len(df_members_now) > 0:
                    atomic_write_csv(members_csv, df_members_now)

                elapsed = time.perf_counter() - t0
                rate = n_done_this_run / elapsed if elapsed > 0 else np.nan

                n_ok_total = int(df_jobs_now["ok"].sum()) if len(df_jobs_now) > 0 and "ok" in df_jobs_now.columns else 0
                best_msg = "best=NA"

                if len(df_jobs_now) > 0 and "mean_of_member_mean_track_dist_km" in df_jobs_now.columns:
                    good = df_jobs_now[df_jobs_now.get("ok", 0) == 1].copy()
                    good = good[good["mean_of_member_mean_track_dist_km"].notna()]

                    if len(good) > 0:
                        sort_cols = [c for c in [
                            "mean_of_member_mean_track_dist_km",
                            "median_of_member_mean_track_dist_km",
                            "mean_traj_length",
                        ] if c in good.columns]
                        if len(sort_cols):
                            good = good.sort_values(sort_cols, ascending=[True] * len(sort_cols))
                            best = good.iloc[0]
                            best_msg = (
                                f"best_trial={int(best['trial_id'])} "
                                f"mean_track_km={best['mean_of_member_mean_track_dist_km']:.2f} "
                                f"mean_len={best['mean_traj_length']:.2f}"
                            )

                print(
                    f"[run {n_done_this_run:4d}/{len(pending_jobs)} | total {n_total_done_now:4d}/{len(all_jobs)}] "
                    f"ok_total={n_ok_total:4d} fail_total={n_total_done_now - n_ok_total:4d} "
                    f"rate={rate:.2f} trials/s {best_msg}",
                    flush=True,
                )

                buffer_trial_rows = []
                buffer_member_rows = []

    all_job_rows_final = load_jsonl_records(jobs_jsonl)
    all_member_rows_final = load_jsonl_records(members_jsonl)

    final_jobs = pd.DataFrame(all_job_rows_final)
    final_members = pd.DataFrame(all_member_rows_final)

    if len(final_jobs) > 0:
        sort_cols = [c for c in [
            "ok",
            "mean_of_member_mean_track_dist_km",
            "median_of_member_mean_track_dist_km",
            "mean_traj_length",
        ] if c in final_jobs.columns]

        asc = [(False if c == "ok" else True) for c in sort_cols]
        if len(sort_cols):
            final_jobs = final_jobs.sort_values(sort_cols, ascending=asc)

        atomic_write_csv(outdir / "results_jobs_final.csv", final_jobs)

    if len(final_members) > 0:
        atomic_write_csv(outdir / "results_members_final.csv", final_members)

    summary = {
        "finished_at": pd.Timestamp.utcnow().isoformat(),
        "elapsed_sec": float(time.perf_counter() - t0),
        "n_trials_total": len(all_jobs),
        "n_trials_ok": int(final_jobs["ok"].sum()) if len(final_jobs) and "ok" in final_jobs.columns else 0,
        "n_trials_failed": int((final_jobs["ok"] == 0).sum()) if len(final_jobs) and "ok" in final_jobs.columns else 0,
        "resumed": len(completed_ids) > 0,
        "n_trials_completed_before_this_run": len(completed_ids),
        "n_trials_completed_this_run": n_done_this_run,
    }
    atomic_write_json(summary_path, summary)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
