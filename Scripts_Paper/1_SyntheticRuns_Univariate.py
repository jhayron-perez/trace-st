# Manuscript workflow: controlled two-variable causal-trajectory experiment.
#
# A coherent Gaussian anomaly follows a prescribed path in V1. V2 depends on
# lagged V1 and is therefore correlated with it, but V2 is not a causal parent
# of V1. Starting from the V1 event at t0, this script evaluates whether
# TraCE-ST reconstructs the known V1 pathway without switching spuriously to
# V2. Elastic-Net Granger, PCMCI, and DYNOTEARS are evaluated over their
# respective hyperparameter spaces. Geometry and parent-selection diagnostics
# are written incrementally so interrupted high-performance-computing searches
# can be resumed safely.

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
from scipy.ndimage import gaussian_filter
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn

# -----------------------------------------------------------------------------
# Project import
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(Path(__file__).resolve().parent.parent)

import trace_st as tst  # noqa: E402

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
GLOBAL_SEED = 11
SEARCH_SEED = 11

# -----------------------------------------------------------------------------
# Controlled fields and prescribed trajectories
# -----------------------------------------------------------------------------
def gaussian_blob(y2d, x2d, y0, x0, sig_y=6.0, sig_x=10.0, amp=1.0):
    dy = y2d - float(y0)
    dx = x2d - float(x0)
    return amp * np.exp(-0.5 * (dy / sig_y) ** 2 - 0.5 * (dx / sig_x) ** 2)


def make_spatiotemporal_ar1_noise(
    shape,
    phi=0.8,
    sigma=0.1,
    smooth_sigma_y=1.5,
    smooth_sigma_x=1.5,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    T, NY, NX = shape
    noise = np.zeros(shape, dtype=np.float32)

    eps0 = rng.normal(0.0, sigma, size=(NY, NX)).astype(np.float32)
    eps0 = gaussian_filter(eps0, sigma=(smooth_sigma_y, smooth_sigma_x), mode="nearest")
    noise[0] = eps0 / np.sqrt(1.0 - phi**2 + 1e-12)

    for t in range(1, T):
        eps = rng.normal(0.0, sigma, size=(NY, NX)).astype(np.float32)
        eps = gaussian_filter(eps, sigma=(smooth_sigma_y, smooth_sigma_x), mode="nearest")
        noise[t] = phi * noise[t - 1] + eps

    return noise


def linear_centers_to_event(event_y, event_x, vy, vx, T):
    centers = np.zeros((T, 2), dtype=float)
    centers[-1] = (event_y, event_x)
    for t in range(T - 2, -1, -1):
        centers[t, 0] = centers[t + 1, 0] - vy
        centers[t, 1] = centers[t + 1, 1] - vx
    return centers


def bezier_centers(start, control1, control2, end, T):
    u = np.linspace(0.0, 1.0, T)
    centers = np.zeros((T, 2), dtype=float)
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(control1, dtype=float)
    p2 = np.asarray(control2, dtype=float)
    p3 = np.asarray(end, dtype=float)

    for i, s in enumerate(u):
        centers[i] = (
            (1 - s) ** 3 * p0
            + 3 * (1 - s) ** 2 * s * p1
            + 3 * (1 - s) * s ** 2 * p2
            + s ** 3 * p3
        )
    return centers


def build_synthetic_case(n_pre=0, pre_origin="hold"):
    """Paper T1 record (40 daily steps, event at the last day).

    revision-1 G (opt-in, n_pre > 0): prepend n_pre days *before* the true origin (centers[0])
    so that n_steps_time > 37 is feasible; days 0..39 stay byte-identical to the paper record.
    See extend_record_before_origin.
    """
    T = 40
    time_coord = pd.date_range("2001-01-01", periods=T, freq="D")

    Y_MIN, Y_MAX = 0, 200
    X_MIN, X_MAX = 0, 200
    NY = 121
    NX = 241
    ys = np.linspace(Y_MIN, Y_MAX, NY)
    xs = np.linspace(X_MIN, X_MAX, NX)
    x2d, y2d = np.meshgrid(xs, ys)
    
    track_specs = [
        dict(
            name="Track 1",
            event=(100.0, 50.0),
            centers=bezier_centers(
                start=(50.0, 10.0),
                control1=(70.0, 40.0),
                control2=(80.0, 55.0),
                end=(100.0, 50.0),
                T=T,
            ),
            amp=1.0,
        ),
    
        dict(
            name="Track 2",
            event=(110.0, 105.0),
            centers=bezier_centers(
                start=(160.0, 60.0),
                control1=(150.0, 70.0),
                control2=(130.0, 130.0),
                end=(110.0, 105.0),
                T=T,
            ),
            amp=1.0,
        ),
    
        dict(
            name="Track 3",
            event=(70.0, 100.0),
            centers=linear_centers_to_event(
                event_y=70.0,
                event_x=100.0,
                vy=(70.0 - 36.0) / (T - 1),
                vx=(100.0 - 168.0) / (T - 1),
                T=T,
            ),
            amp=1.0,
        ),
    ]

    sig_y = 10.0
    sig_x = 10.0
    amp = 1.0
    phi_A = 0.8
    sigma_A = 0.1
    beta_AB = 0.8
    phi_B = 0.8
    sigma_B = 0.1

    A_clean = np.zeros((T, NY, NX), dtype=np.float32)
    for spec in track_specs:
        centers = spec["centers"]
        amp_i = spec.get("amp", amp)
        for t in range(T):
            A_clean[t] += gaussian_blob(
                y2d,
                x2d,
                y0=centers[t, 0],
                x0=centers[t, 1],
                sig_y=sig_y,
                sig_x=sig_x,
                amp=amp_i,
            ).astype(np.float32)

    rng_A = np.random.default_rng(GLOBAL_SEED)
    noise_A = make_spatiotemporal_ar1_noise(
        A_clean.shape,
        phi=phi_A,
        sigma=sigma_A,
        rng=rng_A,
    )
    A = np.clip(A_clean + noise_A, 0.0, None)

    rng_B = np.random.default_rng(GLOBAL_SEED + 1)
    noise_B = make_spatiotemporal_ar1_noise(
        A.shape,
        phi=0.8,
        sigma=sigma_B,
        rng=rng_B,
    )
    B = np.zeros_like(A, dtype=np.float32)
    B[0] = noise_B[0]
    for t in range(1, T):
        B[t] = phi_B * B[t - 1] + beta_AB * A[t - 1] + noise_B[t]
    B = np.clip(B, 0.0, None)

    full_data_A = xr.DataArray(
        A[:, None, :, :].astype(np.float32),
        dims=("time", "var", "y", "x"),
        coords={"time": time_coord, "var": ["A"], "y": ys, "x": xs},
        name="synthetic",
    )
    full_data_B = xr.DataArray(
        B[:, None, :, :].astype(np.float32),
        dims=("time", "var", "y", "x"),
        coords={"time": time_coord, "var": ["B"], "y": ys, "x": xs},
        name="synthetic",
    )

    if int(n_pre) > 0:
        A, B, time_coord, track_specs = extend_record_before_origin(
            A, B, noise_A, time_coord, track_specs, int(n_pre), str(pre_origin),
            y2d=y2d, x2d=x2d, sig_y=sig_y, sig_x=sig_x, amp=amp,
            phi_A=phi_A, sigma_A=sigma_A, phi_B=phi_B, beta_AB=beta_AB, sigma_B=sigma_B,
        )
        full_data_A = xr.DataArray(A[:, None, :, :].astype(np.float32), dims=("time", "var", "y", "x"),
                                   coords={"time": time_coord, "var": ["A"], "y": ys, "x": xs}, name="synthetic")
        full_data_B = xr.DataArray(B[:, None, :, :].astype(np.float32), dims=("time", "var", "y", "x"),
                                   coords={"time": time_coord, "var": ["B"], "y": ys, "x": xs}, name="synthetic")

    full_data_run = xr.concat([full_data_A, full_data_B], dim="var")
    full_data_run = full_data_run.rename({"y": "lat", "x": "lon"})
    date_end = str(time_coord[-1].date())

    return full_data_run, track_specs, date_end


def extend_record_before_origin(A, B, noise_A, time_coord, track_specs, n_pre, pre_origin, *,
                                y2d, x2d, sig_y, sig_x, amp, phi_A, sigma_A, phi_B, beta_AB, sigma_B):
    """revision-1 G: prepend n_pre days before day 0 (the true origin of every track).

    - Source: pre_origin="hold" -> each blob sits at its origin centers[0] (truth before the
      origin = origin); "absent" -> no blob (noise only; truth still set to the origin).
    - A noise: the same AR(1) continued *backwards* from noise_A[0] (a stationary Gaussian AR(1)
      is time-reversible): n[-k] = phi_A n[-k+1] + smoothed eps, rng seed GLOBAL_SEED + 101.
    - B: same recursion as the paper, run forward over the prepended days from a fresh noise
      start (seed GLOBAL_SEED + 102): B[t] = phi_B B[t-1] + beta_AB A[t-1] + noise. Day 0 keeps
      the paper value B[0] = noise_B[0] (as in the paper, B has a cold start at day 0), so days
      0..39 of A and B are byte-identical to the paper record.
    - Track truth: centers prepended with n_pre copies of centers[0]; event unchanged.
    """
    if pre_origin not in ("hold", "absent"):
        raise ValueError("pre_origin must be 'hold' or 'absent'")
    _, NY, NX = A.shape
    A_clean_pre = np.zeros((n_pre, NY, NX), dtype=np.float32)
    if pre_origin == "hold":
        for spec in track_specs:
            y0, x0 = spec["centers"][0]
            blob = gaussian_blob(y2d, x2d, y0=y0, x0=x0, sig_y=sig_y, sig_x=sig_x,
                                 amp=spec.get("amp", amp)).astype(np.float32)
            A_clean_pre += blob[None]

    rng_pre = np.random.default_rng(GLOBAL_SEED + 101)
    noise_pre = np.zeros((n_pre, NY, NX), dtype=np.float32)
    nxt = noise_A[0]
    for k in range(n_pre - 1, -1, -1):  # fill day -1, -2, ... (index n_pre-1 is day -1)
        eps = rng_pre.normal(0.0, sigma_A, size=(NY, NX)).astype(np.float32)
        eps = gaussian_filter(eps, sigma=(1.5, 1.5), mode="nearest")
        noise_pre[k] = phi_A * nxt + eps
        nxt = noise_pre[k]
    A_pre = np.clip(A_clean_pre + noise_pre, 0.0, None)

    noise_B_pre = make_spatiotemporal_ar1_noise(A_pre.shape, phi=0.8, sigma=sigma_B,
                                                rng=np.random.default_rng(GLOBAL_SEED + 102))
    B_pre = np.zeros_like(A_pre, dtype=np.float32)
    B_pre[0] = noise_B_pre[0]
    for t in range(1, n_pre):
        B_pre[t] = phi_B * B_pre[t - 1] + beta_AB * A_pre[t - 1] + noise_B_pre[t]
    B_pre = np.clip(B_pre, 0.0, None)

    time_pre = pd.date_range(end=time_coord[0] - pd.Timedelta(days=1), periods=n_pre, freq="D")
    new_specs = []
    for spec in track_specs:
        sp = dict(spec)
        c = np.asarray(spec["centers"], dtype=float)
        sp["centers"] = np.vstack([np.repeat(c[:1], n_pre, axis=0), c])
        new_specs.append(sp)
    return (np.concatenate([A_pre, A], axis=0), np.concatenate([B_pre, B], axis=0),
            time_pre.append(time_coord), new_specs)


def n_pre_days_for(n_steps, record_length=None, max_window_days=4, paper_T=40):
    """Days to prepend: explicit record_length - 40, else the minimum so every step of every
    trial (timewindow up to 4 d) has a full window: max(0, n_steps + max_window - 1 - 40)."""
    if record_length is not None:
        if int(record_length) < paper_T:
            raise ValueError("record_length must be >= 40")
        return int(record_length) - paper_T
    return max(0, int(n_steps) + int(max_window_days) - 1 - paper_T)


# -----------------------------------------------------------------------------
# Baseline TraCE-ST configurations for each causal-discovery engine
# -----------------------------------------------------------------------------
def build_base_params():
    box_size = 20
    radius = 2
    timewindow = "4d"
    timeres = "1d"
    spaceres = 1

    params_elasticnet = dict(
        timeres=timeres,
        spaceres=spaceres,
        box_size=box_size,
        radius=radius,
        starting_lat=100.0,
        starting_lon=100.0,
        timewindow=timewindow,
        child_of_interest=1,
        n_steps_time=30,
        averaging_winners=True,
        snap_to_grid=False,
        montecarlo=False,
        beta_softmax=1.0,
        eps_dbscan=0.15,
        min_samples_dbscan=2,
        score_mode="mean",
        gamma=0.5,
        alpha=3.0,
        prefer_sign="both",
        verbose=False,
        winner_summary="mean",
        prob_rule="linear",
        cd_method="granger",
        cd_kwargs=dict(
            lambda_a=0.5,
            l1_ratio=0.4,
            dependence_threshold=1e-7,
            max_iter=100000,
            fit_intercept=False,
            refit_ridge=False,
            ridge_alpha=1e-2,
        ),
    )

    params_pcmci = dict(
        timeres=timeres,
        spaceres=spaceres,
        box_size=box_size,
        radius=radius,
        starting_lat=100.0,
        starting_lon=100.0,
        timewindow=timewindow,
        child_of_interest=1,
        n_steps_time=30,
        averaging_winners=True,
        snap_to_grid=False,
        montecarlo=False,
        beta_softmax=1.0,
        eps_dbscan=0.15,
        min_samples_dbscan=2,
        score_mode="mean",
        gamma=0.5,
        alpha=3.0,
        prefer_sign="both",
        verbose=False,
        winner_summary="mean",
        prob_rule="linear",
        cd_method="pcmci",
        cd_kwargs=dict(
            cd_function="run_pcmci",
            min_tau=1,
            pc_alpha=0.01,
            graph_p_threshold=0.01,
            cond_ind_test=ParCorr(),
            fdr_method=None,
            allow_center_directed_links=True,
            dependencies_wrap=False,
            y_ascending=True,
            strength_threshold=None,
        ),
    )

    params_dynotears = dict(
        timeres=timeres,
        spaceres=spaceres,
        box_size=box_size,
        radius=radius,
        starting_lat=100.0,
        starting_lon=100.0,
        timewindow=timewindow,
        child_of_interest=1,
        n_steps_time=30,
        averaging_winners=True,
        snap_to_grid=False,
        montecarlo=False,
        beta_softmax=1.0,
        eps_dbscan=0.15,
        min_samples_dbscan=2,
        score_mode="mean",
        gamma=0.5,
        alpha=3.0,
        prefer_sign="both",
        verbose=False,
        winner_summary="mean",
        prob_rule="linear",
        cd_method="dynotears",
        cd_kwargs=dict(
            lambda_a=0.01,
            max_iter=300,
            dependence_threshold=1e-3,
            strength_threshold=None,
            verbose=0,
            allow_center_directed_links=True,
            dependencies_wrap=False,
        ),
    )

    return params_elasticnet, params_pcmci, params_dynotears


# -----------------------------------------------------------------------------
# Physically and computationally admissible search domains
# -----------------------------------------------------------------------------
SEARCH_SPACE_SHARED = dict(
    timewindow=["2d", "3d", "4d"],
    box_size=[15, 20, 25, 30],
    radius=[2, 3],
    eps_dbscan=[0.05, 0.10, 0.15, 0.20, 0.25],
    min_samples_dbscan=[2],
    score_mode=["mean", "sum"],
    winner_summary=["mean"],
    alpha=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    prefer_sign=["both"],
    prob_rule=["linear", "softmax"],
    beta_softmax=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
)


def sample_shared_params(rng):
    out = {}
    for k, v in SEARCH_SPACE_SHARED.items():
        x = rng.choice(v)
        if isinstance(x, np.str_):
            x = str(x)
        elif isinstance(x, np.generic):
            x = x.item()
        out[k] = x
    return out


def sample_granger_trial(rng):
    return {
        "lambda_a": float(10 ** rng.uniform(-3.0, -0.5)),
        "l1_ratio": float(10 ** rng.uniform(-4.0, 0.0)),
        "dependence_threshold": float(rng.choice([1e-7])),
        "max_iter": int(rng.choice([100000])),
        "fit_intercept": bool(rng.choice([False])),
        "refit_ridge": bool(rng.choice([False])),
        "ridge_alpha": float(rng.choice([1e-2])),
    }


def sample_pcmci_trial(rng):
    cond_name = str(rng.choice(["parcorr", "robustparcorr"]))
    fdr = rng.choice([None, "bh"])
    if isinstance(fdr, np.str_):
        fdr = str(fdr)

    return {
        "pc_alpha": float(rng.choice([1e-4, 1e-3, 1e-2, 1e-1])),
        "graph_p_threshold": float(rng.choice([1e-4, 1e-3, 1e-2, 1e-1])),
        "cond_ind_test_name": cond_name,
        "fdr_method": fdr,
    }


def sample_dynotears_trial(rng):
    return {
        "lambda_a": float(10 ** rng.uniform(-3.0, -0.5)),
        "max_iter": int(rng.choice([100, 300, 500, 1000, 3000])),
        "dependence_threshold": float(rng.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])),
    }


def build_jobs(n_per_method=1000, seed=123):
    rng = np.random.default_rng(seed)
    jobs = []

    for i in range(n_per_method):
        shared = sample_shared_params(rng)
        jobs.append(("granger", i, {**shared, **sample_granger_trial(rng)}))

    for i in range(n_per_method):
        shared = sample_shared_params(rng)
        jobs.append(("pcmci", i, {**shared, **sample_pcmci_trial(rng)}))

    for i in range(n_per_method):
        shared = sample_shared_params(rng)
        jobs.append(("dynotears", i, {**shared, **sample_dynotears_trial(rng)}))

    return jobs


# -----------------------------------------------------------------------------
# Reconstruction metrics against the prescribed V1 pathway
# -----------------------------------------------------------------------------
def build_cond_ind_test(name):
    if name == "parcorr":
        return ParCorr()
    if name == "robustparcorr":
        return RobustParCorr()
    if name == "gpdc":
        return GPDC()
    if name == "cmiknn":
        return CMIknn(
            workers=1,
            knn=5,
            shuffle_neighbors=3,
            transform="ranks",
            model_selection_folds=2,
        )
    raise ValueError(f"Unknown cond_ind_test_name: {name}")


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


def pairwise_step_lengths(centers):
    if len(centers) < 2:
        return np.array([], dtype=float)
    diffs = np.diff(centers, axis=0)
    return np.sqrt((diffs ** 2).sum(axis=1))


def summarize_track(centers, parents, truth, target_len=31):
    centers_raw = np.asarray(centers, dtype=float)
    parents_raw = np.asarray(parents, dtype=int)
    truth_full = np.asarray(truth, dtype=float)

    raw_len = len(centers_raw)
    failed = int(raw_len == 0)

    if raw_len == 0:
        return {
            "mean_distance": np.nan,
            "max_distance": np.nan,
            "origin_distance": np.nan,
            "event_distance": np.nan,
            "raw_len": 0,
            "path_length": np.nan,
            "net_displacement": np.nan,
            "mean_step": np.nan,
            "stagnant_frac": np.nan,
            "parent2_frac": np.nan,
            "failed": 1,
        }

    # reverse traced path so it goes origin -> event
    pred = centers_raw[::-1].copy()
    parents_ord = parents_raw[::-1].copy()

    # take the final target_len truth points, also origin -> event
    T = len(truth_full)
    truth_cmp = truth_full[T - target_len:T]

    # pad or truncate pred to match target_len
    if len(pred) < target_len:
        pred = np.vstack([
            pred,
            np.repeat(pred[-1:, :], target_len - len(pred), axis=0)
        ])
        parents_ord = np.concatenate([
            parents_ord,
            np.repeat(parents_ord[-1:], target_len - len(parents_ord), axis=0)
        ])
    else:
        pred = pred[:target_len]
        parents_ord = parents_ord[:target_len]

    dy = pred[:, 0] - truth_cmp[:, 0]
    dx = pred[:, 1] - truth_cmp[:, 1]
    pointwise_dist = np.sqrt(dx**2 + dy**2)

    steps = np.sqrt(np.sum(np.diff(pred, axis=0) ** 2, axis=1))
    path_length = float(np.sum(steps))
    net_displacement = float(np.sqrt(np.sum((pred[-1] - pred[0]) ** 2)))
    mean_step = float(np.mean(steps)) if len(steps) > 0 else 0.0
    stagnant_frac = float(np.mean(steps < 1e-6)) if len(steps) > 0 else 1.0
    parent2_frac = float(np.mean(parents_ord == 2))

    return {
        "mean_distance": float(np.mean(pointwise_dist)),
        "max_distance": float(np.max(pointwise_dist)),
        "origin_distance": float(pointwise_dist[0]),
        "event_distance": float(pointwise_dist[-1]),
        "raw_len": raw_len,
        "path_length": path_length,
        "net_displacement": net_displacement,
        "mean_step": mean_step,
        "stagnant_frac": stagnant_frac,
        "parent2_frac": parent2_frac,
        "failed": 0,
    }


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
        json.dump(obj, f, indent=2)
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


def load_completed_job_keys(jobs_jsonl: Path):
    records = load_jsonl_records(jobs_jsonl)
    completed = set()
    for r in records:
        try:
            method = str(r["method"])
            trial_id = int(r["trial_id"])
            completed.add((method, trial_id))
        except Exception:
            continue
    return completed, records


def filter_pending_jobs(all_jobs, completed_keys):
    pending = []
    for job in all_jobs:
        method, trial_id, _ = job
        key = (str(method), int(trial_id))
        if key not in completed_keys:
            pending.append(job)
    return pending


# -----------------------------------------------------------------------------
# One independent hyperparameter evaluation
# -----------------------------------------------------------------------------
def evaluate_job(job, strength_mode="native", grouping="dbscan", n_steps=30, n_pre=0, pre_origin="hold"):
    method, trial_id, trial = job
    t0 = time.perf_counter()

    try:
        if int(n_pre) > 0:
            full_data_run, track_specs, date_end = build_synthetic_case(n_pre=int(n_pre), pre_origin=pre_origin)
        else:
            full_data_run, track_specs, date_end = build_synthetic_case()
        params_elasticnet, params_pcmci, params_dynotears = build_base_params()

        if method == "granger":
            pbase = copy.deepcopy(params_elasticnet)
        elif method == "pcmci":
            pbase = copy.deepcopy(params_pcmci)
        elif method == "dynotears":
            pbase = copy.deepcopy(params_dynotears)
        else:
            raise ValueError(f"Unknown method: {method}")

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
        ]:
            pbase[k] = trial[k]

        pbase["n_steps_time"] = int(n_steps)  # 30 = paper; revision-1 G: --n-steps
        pbase["child_of_interest"] = 1
        pbase["montecarlo"] = False
        pbase["snap_to_grid"] = False
        pbase["averaging_winners"] = True
        pbase["verbose"] = False

        if strength_mode != "native":
            pbase["strength_mode"] = str(strength_mode)  # revision-1 opt-in (T9)
        if grouping != "dbscan":
            pbase["grouping"] = str(grouping)  # revision-1 opt-in (E, R1 major 2)

        if method == "granger":
            pbase["cd_kwargs"]["lambda_a"] = trial["lambda_a"]
            pbase["cd_kwargs"]["l1_ratio"] = trial["l1_ratio"]
            pbase["cd_kwargs"]["dependence_threshold"] = trial["dependence_threshold"]
            pbase["cd_kwargs"]["max_iter"] = trial["max_iter"]
            pbase["cd_kwargs"]["fit_intercept"] = trial["fit_intercept"]
            pbase["cd_kwargs"]["refit_ridge"] = trial["refit_ridge"]
            pbase["cd_kwargs"]["ridge_alpha"] = trial["ridge_alpha"]

        elif method == "pcmci":
            pbase["cd_kwargs"]["pc_alpha"] = trial["pc_alpha"]
            pbase["cd_kwargs"]["graph_p_threshold"] = trial["graph_p_threshold"]
            pbase["cd_kwargs"]["fdr_method"] = trial["fdr_method"]
            pbase["cd_kwargs"]["cond_ind_test"] = build_cond_ind_test(trial["cond_ind_test_name"])

        elif method == "dynotears":
            pbase["cd_kwargs"]["lambda_a"] = trial["lambda_a"]
            pbase["cd_kwargs"]["max_iter"] = trial["max_iter"]
            pbase["cd_kwargs"]["dependence_threshold"] = trial["dependence_threshold"]

        track_rows = []
        mean_distances = []
        origin_distances = []
        event_distances = []
        path_lengths = []
        net_displacements = []
        stagnant_fracs = []
        parent2_fracs = []

        for i, spec in enumerate(track_specs, start=1):
            event_y, event_x = spec["event"]
            p = copy.deepcopy(pbase)
            p["starting_lat"] = float(event_y)
            p["starting_lon"] = float(event_x)

            centers, parents, _ = run_one_track(full_data_run, p, date_end)
            track_summary = summarize_track(
                centers,
                parents,
                spec["centers"],
                target_len=int(pbase["n_steps_time"]) + 1,
            )

            row = {
                "method": method,
                "trial_id": int(trial_id),
                "track_idx": int(i),
                "track_name": spec["name"],
                "event_y": float(event_y),
                "event_x": float(event_x),
                "centers_raw": centers.tolist(),
                "parents_raw": parents.tolist(),
                "elapsed_sec": time.perf_counter() - t0,
                **{k: v for k, v in trial.items()},
                **track_summary,
            }
            track_rows.append(row)

            mean_distances.append(track_summary["mean_distance"])
            origin_distances.append(track_summary["origin_distance"])
            event_distances.append(track_summary["event_distance"])
            path_lengths.append(track_summary["path_length"])
            net_displacements.append(track_summary["net_displacement"])
            stagnant_fracs.append(track_summary["stagnant_frac"])
            parent2_fracs.append(track_summary["parent2_frac"])

        elapsed = time.perf_counter() - t0

        job_row = {
            "method": method,
            "trial_id": int(trial_id),
            **{k: v for k, v in trial.items()},
            "score_mean_distance": float(np.nanmean(mean_distances)),
            "score_origin_distance": float(np.nanmean(origin_distances)),
            "score_event_distance": float(np.nanmean(event_distances)),
            "score_path_length": float(np.nanmean(path_lengths)),
            "score_net_displacement": float(np.nanmean(net_displacements)),
            "score_stagnant_frac": float(np.nanmean(stagnant_fracs)),
            "score_parent2_frac": float(np.nanmean(parent2_fracs)),
            "elapsed_sec": float(elapsed),
            "ok": 1,
            "error": "",
        }

        return {
            "ok": True,
            "job_row": job_row,
            "track_rows": track_rows,
        }

    except Exception as e:
        elapsed = time.perf_counter() - t0
        err = "".join(traceback.format_exception_only(type(e), e)).strip()

        job_row = {
            "method": method,
            "trial_id": int(trial_id),
            **{k: v for k, v in trial.items()},
            "score_mean_distance": np.nan,
            "score_origin_distance": np.nan,
            "score_event_distance": np.nan,
            "score_path_length": np.nan,
            "score_net_displacement": np.nan,
            "score_stagnant_frac": np.nan,
            "score_parent2_frac": np.nan,
            "elapsed_sec": float(elapsed),
            "ok": 0,
            "error": err,
        }

        return {
            "ok": False,
            "job_row": job_row,
            "track_rows": [],
        }


# -----------------------------------------------------------------------------
# Command-line orchestration and multiprocessing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Trace-ST synthetic hyperparameter search")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--trials-per-method", type=int, default=1000, help="Trials per method")
    parser.add_argument("--n-workers", type=int, default=128, help="Parallel workers")
    parser.add_argument("--seed", type=int, default=SEARCH_SEED, help="Search seed")
    parser.add_argument("--flush-every", type=int, default=25, help="Write snapshot every N completed jobs")
    parser.add_argument("--strength-mode", type=str, default="native", choices=["native", "refit"],
                        help="revision-1 T9: link-strength definition passed to run_track")
    parser.add_argument("--grouping", type=str, default="dbscan",
                        choices=["dbscan", "strongest", "variable_sign", "sectors8", "kmeans"],
                        help="revision-1 E: winner grouping passed to run_track (default dbscan = original)")
    parser.add_argument("--methods", type=str, nargs="+", default=None, choices=["granger", "pcmci", "dynotears"],
                        help="Subset of methods to run (default: all). Configs are drawn for all methods "
                             "first, so a subset gets the same configs (trial ids) as the full run.")
    parser.add_argument("--n-configs", type=int, default=None,
                        help="Run only trial ids < N of each selected method (default: all)")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="revision-1 G: n_steps_time (paper 30). >37 prepends days before the origin")
    parser.add_argument("--record-length", type=int, default=None,
                        help="revision-1 G: total record length in days (>= 40; default: 40 or the "
                             "minimum n_steps + 3 so every step has a full window)")
    parser.add_argument("--pre-origin", type=str, default="hold", choices=["hold", "absent"],
                        help="revision-1 G: source before the origin: hold at the origin or absent")
    parser.add_argument("--trial-ids", type=str, default=None,
                        help="Run only these trial ids: comma list '0,5,7' or '@FILE' (one id per line)")
    return parser.parse_args()


def parse_trial_ids(spec):
    if spec is None:
        return None
    if spec.startswith("@"):
        txt = Path(spec[1:]).read_text().replace(",", "\n").split()
    else:
        txt = spec.replace(",", " ").split()
    return sorted({int(x) for x in txt})


def select_jobs(all_jobs, methods=None, n_configs=None, trial_ids=None):
    """Filter the full paper draw (ids unchanged) by method, first N ids and/or explicit ids."""
    out = all_jobs
    if methods is not None:
        out = [j for j in out if j[0] in set(methods)]
    if n_configs is not None:
        out = [j for j in out if int(j[1]) < int(n_configs)]
    if trial_ids is not None:
        keep = set(trial_ids)
        out = [j for j in out if int(j[1]) in keep]
    return out


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs_jsonl = outdir / "results_jobs.jsonl"
    tracks_jsonl = outdir / "results_tracks.jsonl"
    jobs_csv = outdir / "results_jobs_snapshot.csv"
    tracks_csv = outdir / "results_tracks_snapshot.csv"
    meta_path = outdir / "run_metadata.json"
    summary_path = outdir / "run_summary.json"

    all_jobs = build_jobs(n_per_method=args.trials_per_method, seed=args.seed)
    trial_ids = parse_trial_ids(args.trial_ids)
    n_pre = n_pre_days_for(args.n_steps, args.record_length)
    if args.methods is not None or args.n_configs is not None or trial_ids is not None:
        all_jobs = select_jobs(all_jobs, args.methods, args.n_configs, trial_ids)

    completed_keys, existing_job_rows = load_completed_job_keys(jobs_jsonl)
    pending_jobs = filter_pending_jobs(all_jobs, completed_keys)

    existing_track_rows = load_jsonl_records(tracks_jsonl)

    meta = {
        "seed": args.seed,
        "trials_per_method": args.trials_per_method,
        "n_workers": args.n_workers,
        "flush_every": args.flush_every,
        "n_total_jobs": len(all_jobs),
        "n_completed_jobs_found_at_start": len(completed_keys),
        "n_pending_jobs_at_start": len(pending_jobs),
        "methods": ["granger", "pcmci", "dynotears"],
        "strength_mode": args.strength_mode,
        **({} if args.grouping == "dbscan" else {"grouping": args.grouping}),
        **({} if args.methods is None else {"methods_run": args.methods}),
        **({} if args.n_configs is None else {"n_configs": args.n_configs}),
        **({} if trial_ids is None else {"trial_ids": trial_ids}),
        **({} if (args.n_steps == 30 and n_pre == 0) else
           {"n_steps": args.n_steps, "n_pre_days": n_pre, "record_length": 40 + n_pre, "pre_origin": args.pre_origin}),
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }
    atomic_write_json(meta_path, meta)

    if existing_job_rows:
        df_jobs_existing = pd.DataFrame(existing_job_rows)
        atomic_write_csv(jobs_csv, df_jobs_existing)

    if existing_track_rows:
        df_tracks_existing = pd.DataFrame(existing_track_rows)
        atomic_write_csv(tracks_csv, df_tracks_existing)

    print(f"Total jobs:    {len(all_jobs)}", flush=True)
    print(f"Completed:     {len(completed_keys)}", flush=True)
    print(f"Pending:       {len(pending_jobs)}", flush=True)

    if len(pending_jobs) == 0:
        print("Nothing to do. All jobs already completed.", flush=True)

        final_jobs = pd.DataFrame(existing_job_rows)
        final_tracks = pd.DataFrame(existing_track_rows)

        if len(final_jobs) > 0:
            final_jobs = final_jobs.sort_values(
                ["ok", "score_mean_distance", "score_origin_distance", "score_event_distance"],
                ascending=[False, True, True, True],
            )
            atomic_write_csv(outdir / "results_jobs_final.csv", final_jobs)

        if len(final_tracks) > 0:
            atomic_write_csv(outdir / "results_tracks_final.csv", final_tracks)

        summary = {
            "finished_at": pd.Timestamp.utcnow().isoformat(),
            "elapsed_sec": 0.0,
            "n_jobs_total": len(all_jobs),
            "n_jobs_ok": int(final_jobs["ok"].sum()) if len(final_jobs) else 0,
            "n_jobs_failed": int((1 - final_jobs["ok"]).sum()) if len(final_jobs) else 0,
            "resumed": True,
            "nothing_new_run": True,
        }
        atomic_write_json(summary_path, summary)
        return

    buffer_job_rows = []
    buffer_track_rows = []

    t0 = time.perf_counter()
    n_done_this_run = 0
    n_total_done_now = len(completed_keys)

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as ex:
        futures = [ex.submit(evaluate_job, job, args.strength_mode, args.grouping, args.n_steps, n_pre, args.pre_origin)
                   for job in pending_jobs]

        for fut in as_completed(futures):
            result = fut.result()

            job_row = result["job_row"]
            track_rows = result["track_rows"]

            buffer_job_rows.append(job_row)
            buffer_track_rows.extend(track_rows)

            n_done_this_run += 1
            n_total_done_now += 1

            if (n_done_this_run % args.flush_every == 0) or (n_done_this_run == len(pending_jobs)):
                append_jsonl(jobs_jsonl, buffer_job_rows)
                append_jsonl(tracks_jsonl, buffer_track_rows)

                # reload full persisted state so snapshots reflect both prior runs and current one
                all_job_rows_now = load_jsonl_records(jobs_jsonl)
                all_track_rows_now = load_jsonl_records(tracks_jsonl)

                if all_job_rows_now:
                    df_jobs_now = pd.DataFrame(all_job_rows_now)
                    atomic_write_csv(jobs_csv, df_jobs_now)
                else:
                    df_jobs_now = pd.DataFrame()

                if all_track_rows_now:
                    df_tracks_now = pd.DataFrame(all_track_rows_now)
                    atomic_write_csv(tracks_csv, df_tracks_now)

                elapsed = time.perf_counter() - t0
                rate = n_done_this_run / elapsed if elapsed > 0 else np.nan

                n_ok_total = 0
                best_msg = "best=NA"
                if len(df_jobs_now) > 0:
                    n_ok_total = int(df_jobs_now["ok"].sum())
                    best_df = df_jobs_now[df_jobs_now["ok"] == 1]
                    if len(best_df) > 0:
                        best_df = best_df.sort_values(
                            ["score_mean_distance", "score_origin_distance", "score_event_distance"],
                            ascending=[True, True, True],
                        )
                        best_row = best_df.iloc[0].to_dict()
                        best_msg = (
                            f"best={best_row['method']} trial={best_row['trial_id']} "
                            f"mean_dist={best_row['score_mean_distance']:.4f}"
                        )

                print(
                    f"[run {n_done_this_run:4d}/{len(pending_jobs)} | total {n_total_done_now:4d}/{len(all_jobs)}] "
                    f"ok_total={n_ok_total:4d} fail_total={n_total_done_now - n_ok_total:4d} "
                    f"rate={rate:.2f} jobs/s {best_msg}",
                    flush=True,
                )

                buffer_job_rows = []
                buffer_track_rows = []

    all_job_rows_final = load_jsonl_records(jobs_jsonl)
    all_track_rows_final = load_jsonl_records(tracks_jsonl)

    final_jobs = pd.DataFrame(all_job_rows_final)
    final_tracks = pd.DataFrame(all_track_rows_final)

    if len(final_jobs) > 0:
        final_jobs = final_jobs.sort_values(
            ["ok", "score_mean_distance", "score_origin_distance", "score_event_distance"],
            ascending=[False, True, True, True],
        )
        atomic_write_csv(outdir / "results_jobs_final.csv", final_jobs)

    if len(final_tracks) > 0:
        atomic_write_csv(outdir / "results_tracks_final.csv", final_tracks)

    summary = {
        "finished_at": pd.Timestamp.utcnow().isoformat(),
        "elapsed_sec": time.perf_counter() - t0,
        "n_jobs_total": len(all_jobs),
        "n_jobs_ok": int(final_jobs["ok"].sum()) if len(final_jobs) else 0,
        "n_jobs_failed": int((1 - final_jobs["ok"]).sum()) if len(final_jobs) else 0,
        "resumed": len(completed_keys) > 0,
        "n_jobs_completed_before_this_run": len(completed_keys),
        "n_jobs_completed_this_run": n_done_this_run,
    }
    atomic_write_json(summary_path, summary)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
