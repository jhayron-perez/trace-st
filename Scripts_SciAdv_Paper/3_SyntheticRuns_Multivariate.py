# Manuscript workflow: controlled three-variable causal-mixture experiment.
#
# Independent coherent anomalies propagate in V1 and V3 before jointly
# generating the target V2 during the final event-development period. The
# prescribed alpha_mix values define the known relative contributions of V1
# and V3. This script runs fixed-parameter Monte Carlo trajectory ensembles
# with the Elastic-Net Granger M-CaStLe backend, summarizes variable-specific
# trajectory density and geometry, and tests whether recovered contribution
# diagnostics vary consistently with the prescribed mixture. Hyperparameter
# configurations are evaluated across every alpha_mix value so comparisons are
# made under a common model specification.

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

# -----------------------------------------------------------------------------
# Project import
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(PROJECT_ROOT, flush=True)

import trace_st as tst  # noqa: E402

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
GLOBAL_SEED = 11
SEARCH_SEED = 11

# -----------------------------------------------------------------------------
# Prescribed causal-mixture strengths evaluated in the manuscript
# -----------------------------------------------------------------------------
ALPHA_MIX_GRID = [0.2, 0.4, 0.6, 0.8]

# -----------------------------------------------------------------------------
# Controlled fields, prescribed paths, and lagged response construction
# -----------------------------------------------------------------------------
def gaussian_blob(y2d, x2d, y0, x0, sig_y=6.0, sig_x=10.0, amp=1.0):
    dy = y2d - float(y0)
    dx = x2d - float(x0)
    return amp * np.exp(-0.5 * (dy / sig_y) ** 2 - 0.5 * (dx / sig_x) ** 2)


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


def build_dataarray(var_name, arr, time_coord, ys, xs):
    return xr.DataArray(
        arr[:, None, :, :].astype(np.float32),
        dims=("time", "var", "y", "x"),
        coords={
            "time": time_coord,
            "var": [var_name],
            "y": ys,
            "x": xs,
        },
        name="synthetic",
    )


def build_multivariate_case(
    alpha_mix=0.6,
    T=40,
    N_inject=8,
    tau_mix=1,
    sig_y=10.0,
    sig_x=10.0,
    amp_A=1.0,
    amp_C=1.0,
    phi_A=0.85,
    phi_C=0.85,
    phi_B=0.80,
    sigma_A=0.10,
    sigma_C=0.10,
    sigma_B=0.10,
    clip_to_positive=True,
    seed=GLOBAL_SEED,
):
    time_coord = pd.date_range("2001-01-01", periods=T, freq="D")

    Y_MIN, Y_MAX = 0, 200
    X_MIN, X_MAX = 0, 200
    NY = 121
    NX = 241
    ys = np.linspace(Y_MIN, Y_MAX, NY)
    xs = np.linspace(X_MIN, X_MAX, NX)
    x2d, y2d = np.meshgrid(xs, ys)

    event_y = 100.0
    event_x = 100.0

    centers_A = bezier_centers(
        start=(145.0, 60.0),
        control1=(142.0, 95.0),
        control2=(120.0, 98.0),
        end=(event_y, event_x),
        T=T,
    )
    centers_C = bezier_centers(
        start=(40.0, 150.0),
        control1=(75.0, 148.0),
        control2=(98.0, 118.0),
        end=(event_y, event_x),
        T=T,
    )

    A_clean = np.zeros((T, NY, NX), dtype=np.float32)
    C_clean = np.zeros((T, NY, NX), dtype=np.float32)
    for t in range(T):
        A_clean[t] = gaussian_blob(
            y2d, x2d,
            y0=centers_A[t, 0],
            x0=centers_A[t, 1],
            sig_y=sig_y,
            sig_x=sig_x,
            amp=amp_A,
        ).astype(np.float32)
        C_clean[t] = gaussian_blob(
            y2d, x2d,
            y0=centers_C[t, 0],
            x0=centers_C[t, 1],
            sig_y=sig_y,
            sig_x=sig_x,
            amp=amp_C,
        ).astype(np.float32)

    rng_A = np.random.default_rng(seed + 0)
    rng_C = np.random.default_rng(seed + 2)
    noise_A = make_spatiotemporal_ar1_noise(A_clean.shape, phi=phi_A, sigma=sigma_A, rng=rng_A)
    noise_C = make_spatiotemporal_ar1_noise(C_clean.shape, phi=phi_C, sigma=sigma_C, rng=rng_C)

    A = A_clean + noise_A
    C = C_clean + noise_C
    if clip_to_positive:
        A = np.clip(A, 0.0, None)
        C = np.clip(C, 0.0, None)

    t0 = T - int(N_inject)
    if t0 < 0:
        raise ValueError("N_inject too large.")
    if t0 - tau_mix < 0:
        raise ValueError("Need t0 - tau_mix >= 0.")

    rng_B = np.random.default_rng(seed + 1)
    noise_B = make_spatiotemporal_ar1_noise((T, NY, NX), phi=phi_B, sigma=sigma_B, rng=rng_B)
    B = noise_B.copy()

    for t in range(t0, T):
        B[t] = (
            alpha_mix * A_clean[t - tau_mix]
            + (1.0 - alpha_mix) * C_clean[t - tau_mix]
            + noise_B[t]
        )
    if clip_to_positive:
        B = np.clip(B, 0.0, None)

    full_data_A = build_dataarray("A", A, time_coord, ys, xs)
    full_data_B = build_dataarray("B", B, time_coord, ys, xs)
    full_data_C = build_dataarray("C", C, time_coord, ys, xs)

    full_data_run = xr.concat([full_data_A, full_data_B, full_data_C], dim="var")
    full_data_run = full_data_run.rename({"y": "lat", "x": "lon"})
    date_end = str(time_coord[-1].date())

    case = dict(
        full_data_run=full_data_run,
        A=A,
        B=B,
        C=C,
        A_clean=A_clean,
        C_clean=C_clean,
        centers_A=centers_A,
        centers_C=centers_C,
        event=(event_y, event_x),
        xs=xs,
        ys=ys,
        x2d=x2d,
        y2d=y2d,
        time=time_coord,
        T=T,
        date_end=date_end,
        alpha_mix=float(alpha_mix),
        target_parent_fractions={1: float(alpha_mix), 3: float(1.0 - alpha_mix)},
        injection_start_idx=int(t0),
        tau_mix=int(tau_mix),
        N_inject=int(N_inject),
    )
    return case


# -----------------------------------------------------------------------------
# Baseline probabilistic TraCE-ST configuration
# -----------------------------------------------------------------------------
def build_base_params_multivar():
    return dict(
        timeres="1d",
        spaceres=1,
        box_size=20,
        radius=2,
        starting_lat=None,
        starting_lon=None,
        timewindow="4d",
        child_of_interest=2,  # B
        n_steps_time=30,
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
        winner_summary="mean",
        prob_rule="softmax",
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


# -----------------------------------------------------------------------------
# Admissible TraCE-ST and Elastic-Net parameter domains
# -----------------------------------------------------------------------------
SEARCH_SPACE_SHARED = dict(
    timewindow=["5d","6d","7d"],#
    box_size=[30, 35, 40],#
    radius=[4,5,6],#
    eps_dbscan=[0.05, 0.10, 0.15, 0.20, 0.25],#
    min_samples_dbscan=[2],#
    score_mode=["mean", "sum"],#
    winner_summary=["mean"],#
    alpha=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],#
    prefer_sign=["both"],
)

SEARCH_SPACE_PROB = dict(
    prob_rule=["linear", "softmax"],#
    beta_softmax=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],#
)

SEARCH_SPACE_GRANGER = dict(
    lambda_a=lambda rng: float(10 ** rng.uniform(-2.5, -1)),
    l1_ratio=lambda rng: float(10 ** rng.uniform(-1, 0.0)),
    dependence_threshold=[1e-7],
    max_iter=[100000],
    fit_intercept=[False],
    refit_ridge=[False],
    ridge_alpha=[1e-2],
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


def sample_multivar_trial(rng):
    trial = {}
    for k, v in SEARCH_SPACE_SHARED.items():
        trial[k] = _draw_from_space(rng, v)
    for k, v in SEARCH_SPACE_PROB.items():
        trial[k] = _draw_from_space(rng, v)
    for k, v in SEARCH_SPACE_GRANGER.items():
        trial[k] = _draw_from_space(rng, v)
    return trial


def build_multivar_jobs(n_trials, seed=SEARCH_SEED):
    rng = np.random.default_rng(seed)
    jobs = []
    for trial_id in range(n_trials):
        jobs.append((trial_id, sample_multivar_trial(rng)))
    return jobs


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


def run_ensemble(case, base_params, M=30, seed=123):
    rng_local = np.random.default_rng(seed)
    members = []

    for m in range(M):
        p = copy.deepcopy(base_params)
        p["starting_lat"] = float(case["event"][0])
        p["starting_lon"] = float(case["event"][1])

        np.random.seed(int(rng_local.integers(0, 2_000_000_000)))
        t0 = time.perf_counter()

        try:
            centers, parents, out = run_one_track(case["full_data_run"], p, case["date_end"])
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
            "elapsed_sec": time.perf_counter() - t0,
            "centers": np.asarray(centers, dtype=float),
            "parents": np.asarray(parents, dtype=int),
            "out": out,
        })

    return {"members": members, "params": copy.deepcopy(base_params), "M": M}


def build_params_from_trial_multivar(trial):
    p = build_base_params_multivar()
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
        p[k] = trial[k]

    p["child_of_interest"] = 2
    p["n_steps_time"] = 30
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
# Spatial density, parent contribution, and pathway-geometry diagnostics
# -----------------------------------------------------------------------------
def _accumulate_density_box(density, xs, ys, centers, box_size):
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 0:
        return density

    half = box_size / 2.0
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
def _member_geometry_stats(centers, parents):
    centers = np.asarray(centers, dtype=float)
    parents = np.asarray(parents, dtype=int)

    if len(centers) == 0 or len(parents) == 0:
        return dict(
            n_steps=0,
            path_length=np.nan,
            net_displacement=np.nan,
            mean_step=np.nan,
            stagnant_frac=np.nan,
            first_step_parent=np.nan,  # first non-trivial decision = parents[1]
            end_parent=np.nan,
            frac_parent1=np.nan,
            frac_parent2=np.nan,
            frac_parent3=np.nan,
            frac_nontrivial=np.nan,
            any_nontrivial=0,
            ended_nontrivial=0,
            unique_nontrivial_count=0,
            uses_parent1=0,
            uses_parent3=0,
            uses_both_nontrivial=0,
        )

    if len(centers) >= 2:
        diffs = np.diff(centers, axis=0)
        steps = np.sqrt(np.sum(diffs**2, axis=1))
        path_length = float(np.sum(steps))
        net_displacement = float(np.sqrt(np.sum((centers[-1] - centers[0]) ** 2)))
        mean_step = float(np.mean(steps))
        stagnant_frac = float(np.mean(steps < 1e-6))
    else:
        path_length = 0.0
        net_displacement = 0.0
        mean_step = 0.0
        stagnant_frac = 1.0

    first_step_parent = int(parents[1]) if len(parents) >= 2 else np.nan
    end_parent = int(parents[-1])

    nontrivial = parents[1:] if len(parents) >= 2 else np.array([], dtype=int)
    nontrivial_mask = np.isin(nontrivial, [1, 3])
    unique_nontrivial = np.unique(nontrivial[nontrivial_mask]) if len(nontrivial) else np.array([], dtype=int)

    return dict(
        n_steps=int(len(parents)),
        path_length=path_length,
        net_displacement=net_displacement,
        mean_step=mean_step,
        stagnant_frac=stagnant_frac,
        first_step_parent=first_step_parent,
        end_parent=end_parent,
        frac_parent1=float(np.mean(parents == 1)),
        frac_parent2=float(np.mean(parents == 2)),
        frac_parent3=float(np.mean(parents == 3)),
        frac_nontrivial=float(np.mean(np.isin(parents, [1, 3]))),
        any_nontrivial=int(np.any(np.isin(nontrivial, [1, 3]))),
        ended_nontrivial=int(end_parent in [1, 3]),
        unique_nontrivial_count=int(len(unique_nontrivial)),
        uses_parent1=int(np.any(parents == 1)),
        uses_parent3=int(np.any(parents == 3)),
        uses_both_nontrivial=int((1 in unique_nontrivial) and (3 in unique_nontrivial)),
    )


def summarize_ensemble(case, ensemble_out, params):
    xs = case["xs"]
    ys = case["ys"]
    shape2d = (len(ys), len(xs))
    box_size = float(params["box_size"])

    density_by_parent = {
        1: np.zeros(shape2d, dtype=float),
        2: np.zeros(shape2d, dtype=float),
        3: np.zeros(shape2d, dtype=float),
    }

    initial_parent_counts = {1: 0, 2: 0, 3: 0}
    end_parent_counts = {1: 0, 2: 0, 3: 0}
    step_parent_counts = {1: 0, 2: 0, 3: 0}
    n_ok = 0
    rows = []

    for member in ensemble_out["members"]:
        member_id = member["member_id"]

        if not member["ok"]:
            rows.append({
                "member_id": member_id,
                "ok": 0,
                "error": member.get("error", ""),
                **_member_geometry_stats([], []),
            })
            continue

        centers = np.asarray(member["centers"], dtype=float)
        parents = np.asarray(member["parents"], dtype=int)

        if len(centers) == 0 or len(parents) == 0:
            rows.append({
                "member_id": member_id,
                "ok": 0,
                "error": member.get("error", ""),
                **_member_geometry_stats([], []),
            })
            continue

        n_ok += 1
        stats = _member_geometry_stats(centers, parents)

        initial_parent = stats["first_step_parent"]
        end_parent = stats["end_parent"]

        if initial_parent in initial_parent_counts:
            initial_parent_counts[initial_parent] += 1
        if end_parent in end_parent_counts:
            end_parent_counts[end_parent] += 1

        for p in [1, 2, 3]:
            mask = (parents == p)
            step_parent_counts[p] += int(mask.sum())
            density_by_parent[p] = _accumulate_density_box(
                density_by_parent[p],
                xs,
                ys,
                centers[mask],
                box_size=box_size,
            )

        rows.append({
            "member_id": member_id,
            "ok": 1,
            "error": "",
            **stats,
        })

    df_members = pd.DataFrame(rows)

    total_initial = sum(initial_parent_counts.values())
    total_end = sum(end_parent_counts.values())
    total_steps = sum(step_parent_counts.values())

    initial_parent_frac = {
        p: (initial_parent_counts[p] / total_initial if total_initial > 0 else np.nan)
        for p in [1, 2, 3]
    }
    end_parent_frac = {
        p: (end_parent_counts[p] / total_end if total_end > 0 else np.nan)
        for p in [1, 2, 3]
    }
    step_parent_frac = {
        p: (step_parent_counts[p] / total_steps if total_steps > 0 else np.nan)
        for p in [1, 2, 3]
    }

    density_integral = {
        p: float(np.nansum(density_by_parent[p]))
        for p in [1, 2, 3]
    }
    density_integral_total = float(sum(density_integral.values()))
    density_integral_frac = {
        p: (density_integral[p] / density_integral_total if density_integral_total > 0 else np.nan)
        for p in [1, 2, 3]
    }

    valid_members = df_members[df_members["ok"] == 1].copy()

    def _safe_mean(col):
        return float(valid_members[col].mean()) if len(valid_members) else np.nan

    def _safe_std(col):
        return float(valid_members[col].std(ddof=0)) if len(valid_members) else np.nan

    def _safe_q(col, q):
        return float(valid_members[col].quantile(q)) if len(valid_members) else np.nan

    member_frac_mean = {
        p: _safe_mean(f"frac_parent{p}") for p in [1, 2, 3]
    }
    member_frac_std = {
        p: _safe_std(f"frac_parent{p}") for p in [1, 2, 3]
    }

    validity_metrics = {
        "ok_frac": (n_ok / len(ensemble_out["members"])) if len(ensemble_out["members"]) else np.nan,
        "mean_n_steps": _safe_mean("n_steps"),
        "median_n_steps": _safe_q("n_steps", 0.5),
        "q25_n_steps": _safe_q("n_steps", 0.25),
        "min_n_steps": float(valid_members["n_steps"].min()) if len(valid_members) else np.nan,
        "mean_path_length": _safe_mean("path_length"),
        "median_path_length": _safe_q("path_length", 0.5),
        "mean_net_displacement": _safe_mean("net_displacement"),
        "median_net_displacement": _safe_q("net_displacement", 0.5),
        "mean_mean_step": _safe_mean("mean_step"),
        "mean_stagnant_frac": _safe_mean("stagnant_frac"),
        "median_stagnant_frac": _safe_q("stagnant_frac", 0.5),
        "frac_members_any_nontrivial": _safe_mean("any_nontrivial"),
        "frac_members_end_nontrivial": _safe_mean("ended_nontrivial"),
        "frac_members_use_parent1": _safe_mean("uses_parent1"),
        "frac_members_use_parent3": _safe_mean("uses_parent3"),
        "frac_members_use_both_nontrivial": _safe_mean("uses_both_nontrivial"),
        "mean_unique_nontrivial_count": _safe_mean("unique_nontrivial_count"),
        "mean_frac_parent1": _safe_mean("frac_parent1"),
        "mean_frac_parent2": _safe_mean("frac_parent2"),
        "mean_frac_parent3": _safe_mean("frac_parent3"),
    }

    summary = {
        "n_members": len(ensemble_out["members"]),
        "n_ok": n_ok,
        "initial_parent_counts": initial_parent_counts,
        "initial_parent_frac": initial_parent_frac,
        "end_parent_counts": end_parent_counts,
        "end_parent_frac": end_parent_frac,
        "step_parent_counts": step_parent_counts,
        "step_parent_frac": step_parent_frac,
        "density_by_parent": density_by_parent,
        "density_integral": density_integral,
        "density_integral_total": density_integral_total,
        "density_integral_frac": density_integral_frac,
        "member_frac_mean": member_frac_mean,
        "member_frac_std": member_frac_std,
        "validity_metrics": validity_metrics,
        "df_members": df_members,
    }
    return summary


def summarize_against_truth(case, summary):
    target = case["target_parent_fractions"]
    vm = summary["validity_metrics"]

    row = {
        "alpha_mix": case["alpha_mix"],
        "target_var1": target[1],
        "target_var3": target[3],

        "initial_count_var1": summary["initial_parent_counts"][1],
        "initial_count_var2": summary["initial_parent_counts"][2],
        "initial_count_var3": summary["initial_parent_counts"][3],

        "initial_frac_var1": summary["initial_parent_frac"][1],
        "initial_frac_var2": summary["initial_parent_frac"][2],
        "initial_frac_var3": summary["initial_parent_frac"][3],

        "end_count_var1": summary["end_parent_counts"][1],
        "end_count_var2": summary["end_parent_counts"][2],
        "end_count_var3": summary["end_parent_counts"][3],

        "end_frac_var1": summary["end_parent_frac"][1],
        "end_frac_var2": summary["end_parent_frac"][2],
        "end_frac_var3": summary["end_parent_frac"][3],

        "step_count_var1": summary["step_parent_counts"][1],
        "step_count_var2": summary["step_parent_counts"][2],
        "step_count_var3": summary["step_parent_counts"][3],

        "step_frac_var1": summary["step_parent_frac"][1],
        "step_frac_var2": summary["step_parent_frac"][2],
        "step_frac_var3": summary["step_parent_frac"][3],

        "density_integral_var1": summary["density_integral"][1],
        "density_integral_var2": summary["density_integral"][2],
        "density_integral_var3": summary["density_integral"][3],

        "density_frac_var1": summary["density_integral_frac"][1],
        "density_frac_var2": summary["density_integral_frac"][2],
        "density_frac_var3": summary["density_integral_frac"][3],

        "member_mean_frac_var1": summary["member_frac_mean"][1],
        "member_mean_frac_var2": summary["member_frac_mean"][2],
        "member_mean_frac_var3": summary["member_frac_mean"][3],

        "member_std_frac_var1": summary["member_frac_std"][1],
        "member_std_frac_var2": summary["member_frac_std"][2],
        "member_std_frac_var3": summary["member_frac_std"][3],

        "abs_err_initial_var1": abs(summary["initial_parent_frac"][1] - target[1]),
        "abs_err_initial_var3": abs(summary["initial_parent_frac"][3] - target[3]),
        "abs_err_end_var1": abs(summary["end_parent_frac"][1] - target[1]),
        "abs_err_end_var3": abs(summary["end_parent_frac"][3] - target[3]),
        "abs_err_step_var1": abs(summary["step_parent_frac"][1] - target[1]),
        "abs_err_step_var3": abs(summary["step_parent_frac"][3] - target[3]),
        "abs_err_density_var1": abs(summary["density_integral_frac"][1] - target[1]),
        "abs_err_density_var3": abs(summary["density_integral_frac"][3] - target[3]),

        "mae_initial": np.nanmean([
            abs(summary["initial_parent_frac"][1] - target[1]),
            abs(summary["initial_parent_frac"][3] - target[3]),
        ]),
        "mae_end": np.nanmean([
            abs(summary["end_parent_frac"][1] - target[1]),
            abs(summary["end_parent_frac"][3] - target[3]),
        ]),
        "mae_step": np.nanmean([
            abs(summary["step_parent_frac"][1] - target[1]),
            abs(summary["step_parent_frac"][3] - target[3]),
        ]),
        "mae_density": np.nanmean([
            abs(summary["density_integral_frac"][1] - target[1]),
            abs(summary["density_integral_frac"][3] - target[3]),
        ]),

        "n_ok": summary["n_ok"],
        "M": summary["n_members"],
        "ok_frac": vm["ok_frac"],
        "mean_n_steps": vm["mean_n_steps"],
        "q25_n_steps": vm["q25_n_steps"],
        "min_n_steps": vm["min_n_steps"],
        "mean_path_length": vm["mean_path_length"],
        "mean_net_displacement": vm["mean_net_displacement"],
        "mean_mean_step": vm["mean_mean_step"],
        "mean_stagnant_frac": vm["mean_stagnant_frac"],
        "frac_members_any_nontrivial": vm["frac_members_any_nontrivial"],
        "frac_members_end_nontrivial": vm["frac_members_end_nontrivial"],
        "frac_members_use_parent1": vm["frac_members_use_parent1"],
        "frac_members_use_parent3": vm["frac_members_use_parent3"],
        "frac_members_use_both_nontrivial": vm["frac_members_use_both_nontrivial"],
        "mean_unique_nontrivial_count": vm["mean_unique_nontrivial_count"],
    }

    return pd.DataFrame([row])


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    if np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])

def _safe_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    if np.std(x[mask]) < 1e-12:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def _safe_intercept(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    if np.std(x[mask]) < 1e-12:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[1])


# -----------------------------------------------------------------------------
# Physical plausibility and ensemble-completion criteria
# -----------------------------------------------------------------------------
def evaluate_physical_validity(
    summary,
    min_ok_frac=0.80,
    min_q25_n_steps=20,
    min_mean_net_displacement=8.0,
    max_mean_stagnant_frac=0.75,
    max_step_var2_frac=0.85,
    max_density_var2_frac=0.85,
    min_frac_members_any_nontrivial=0.50,
    min_frac_members_end_nontrivial=0.50,
):
    vm = summary["validity_metrics"]

    checks = {
        "valid_ok_frac": bool(np.isfinite(vm["ok_frac"])) and (vm["ok_frac"] >= min_ok_frac),
        "valid_q25_n_steps": bool(np.isfinite(vm["q25_n_steps"])) and (vm["q25_n_steps"] >= min_q25_n_steps),
        "valid_mean_net_displacement": bool(np.isfinite(vm["mean_net_displacement"])) and (
            vm["mean_net_displacement"] >= min_mean_net_displacement
        ),
        "valid_mean_stagnant_frac": bool(np.isfinite(vm["mean_stagnant_frac"])) and (
            vm["mean_stagnant_frac"] <= max_mean_stagnant_frac
        ),
        "valid_step_var2_not_dominant": bool(np.isfinite(summary["step_parent_frac"][2])) and (
            summary["step_parent_frac"][2] <= max_step_var2_frac
        ),
        "valid_density_var2_not_dominant": bool(np.isfinite(summary["density_integral_frac"][2])) and (
            summary["density_integral_frac"][2] <= max_density_var2_frac
        ),
        "valid_any_nontrivial": bool(np.isfinite(vm["frac_members_any_nontrivial"])) and (
            vm["frac_members_any_nontrivial"] >= min_frac_members_any_nontrivial
        ),
        "valid_end_nontrivial": bool(np.isfinite(vm["frac_members_end_nontrivial"])) and (
            vm["frac_members_end_nontrivial"] >= min_frac_members_end_nontrivial
        ),
    }
    checks["is_valid_physical"] = all(checks.values())
    return checks


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
# Full ensemble artifacts used by downstream manuscript analysis
# -----------------------------------------------------------------------------
def _bundle_stem(trial_id, alpha_mix):
    alpha_tag = str(alpha_mix).replace(".", "p")
    return f"trial_{int(trial_id):05d}_alpha_{alpha_tag}"


def serialize_ensemble_bundle(trial_id, alpha_mix, trial, case, params, ensemble_out, summary):
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
        "trial_id": int(trial_id),
        "alpha_mix": float(alpha_mix),
        "trial": copy.deepcopy(trial),
        "params": copy.deepcopy(params),
        "case_meta": {
            "T": int(case["T"]),
            "N_inject": int(case["N_inject"]),
            "tau_mix": int(case["tau_mix"]),
            "event": [float(case["event"][0]), float(case["event"][1])],
            "target_parent_fractions": {
                "1": float(case["target_parent_fractions"][1]),
                "3": float(case["target_parent_fractions"][3]),
            },
        },
        "summary_small": {
            "n_members": int(summary["n_members"]),
            "n_ok": int(summary["n_ok"]),
            "initial_parent_counts": {str(k): int(v) for k, v in summary["initial_parent_counts"].items()},
            "end_parent_counts": {str(k): int(v) for k, v in summary["end_parent_counts"].items()},
            "step_parent_counts": {str(k): int(v) for k, v in summary["step_parent_counts"].items()},
            "initial_parent_frac": {str(k): float(v) for k, v in summary["initial_parent_frac"].items()},
            "end_parent_frac": {str(k): float(v) for k, v in summary["end_parent_frac"].items()},
            "step_parent_frac": {str(k): float(v) for k, v in summary["step_parent_frac"].items()},
            "density_integral": {str(k): float(v) for k, v in summary["density_integral"].items()},
            "density_integral_frac": {str(k): float(v) for k, v in summary["density_integral_frac"].items()},
            "validity_metrics": {
                k: (float(v) if np.isscalar(v) and np.isfinite(v) else v)
                for k, v in summary["validity_metrics"].items()
            },
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
        "xs": np.asarray(case["xs"], dtype=np.float32),
        "ys": np.asarray(case["ys"], dtype=np.float32),
        "truth_centers_A": np.asarray(case["centers_A"], dtype=np.float32),
        "truth_centers_C": np.asarray(case["centers_C"], dtype=np.float32),
    }

    return {"meta": meta, "arrays": arrays}


def save_ensemble_bundle(bundle, bundles_dir: Path):
    bundles_dir.mkdir(parents=True, exist_ok=True)

    trial_id = int(bundle["meta"]["trial_id"])
    alpha_mix = float(bundle["meta"]["alpha_mix"])
    stem = _bundle_stem(trial_id, alpha_mix)

    npz_path = bundles_dir / f"{stem}.npz"
    json_path = bundles_dir / f"{stem}.json"

    np.savez_compressed(npz_path, **bundle["arrays"])
    atomic_write_json(json_path, bundle["meta"])

    return npz_path, json_path


# -----------------------------------------------------------------------------
# Evaluate one configuration across all prescribed mixture strengths
# -----------------------------------------------------------------------------
def evaluate_trial_across_alphas_full(trial_id, trial, alpha_grid=ALPHA_MIX_GRID, M=30, seed=SEARCH_SEED):
    params = build_params_from_trial_multivar(trial)

    rows_alpha = []
    rows_members = []
    bundles = []

    for j, alpha_mix in enumerate(alpha_grid):
        case = build_multivariate_case(
            alpha_mix=float(alpha_mix),
            T=40,
            N_inject=8,
            seed=GLOBAL_SEED,
        )

        ensemble_out = run_ensemble(
            case,
            params,
            M=M,
            seed=int(seed + 1000 * trial_id + j),
        )

        summary = summarize_ensemble(case, ensemble_out, params)
        validity = evaluate_physical_validity(summary)
        row = summarize_against_truth(case, summary).iloc[0].to_dict()

        row_alpha = {
            "trial_id": int(trial_id),
            **copy.deepcopy(trial),
            **row,
            **validity,
            "estimated_alpha_end": row["end_frac_var1"],
            "estimated_alpha_density": row["density_frac_var1"],
        }
        rows_alpha.append(row_alpha)

        dfm = summary["df_members"].copy()
        dfm["trial_id"] = int(trial_id)
        dfm["alpha_mix"] = float(alpha_mix)
        for k, v in trial.items():
            dfm[k] = v
        rows_members.extend(dfm.to_dict(orient="records"))

        bundles.append(
            serialize_ensemble_bundle(
                trial_id=trial_id,
                alpha_mix=alpha_mix,
                trial=trial,
                case=case,
                params=params,
                ensemble_out=ensemble_out,
                summary=summary,
            )
        )

    df_alpha = pd.DataFrame(rows_alpha).sort_values("alpha_mix").reset_index(drop=True)

    agg = {
        "trial_id": int(trial_id),
        **copy.deepcopy(trial),
        "n_alpha": int(len(df_alpha)),
        "n_alpha_valid_physical": int(df_alpha["is_valid_physical"].sum()) if "is_valid_physical" in df_alpha.columns else 0,
        "frac_alpha_valid_physical": float(df_alpha["is_valid_physical"].mean()) if "is_valid_physical" in df_alpha.columns else np.nan,

        "mean_mae_initial": float(df_alpha["mae_initial"].mean()),
        "mean_mae_end": float(df_alpha["mae_end"].mean()),
        "mean_mae_step": float(df_alpha["mae_step"].mean()),
        "mean_mae_density": float(df_alpha["mae_density"].mean()),

        "max_mae_initial": float(df_alpha["mae_initial"].max()),
        "max_mae_end": float(df_alpha["mae_end"].max()),
        "max_mae_step": float(df_alpha["mae_step"].max()),
        "max_mae_density": float(df_alpha["mae_density"].max()),

        "mean_ok_frac": float(df_alpha["ok_frac"].mean()),
        "min_ok_frac": float(df_alpha["ok_frac"].min()),
        "mean_mean_n_steps": float(df_alpha["mean_n_steps"].mean()),
        "min_q25_n_steps": float(df_alpha["q25_n_steps"].min()),
        "mean_mean_net_displacement": float(df_alpha["mean_net_displacement"].mean()),
        "mean_mean_path_length": float(df_alpha["mean_path_length"].mean()),
        "mean_mean_stagnant_frac": float(df_alpha["mean_stagnant_frac"].mean()),

        "mean_density_var2_frac": float(df_alpha["density_frac_var2"].mean()),
        "mean_step_var2_frac": float(df_alpha["step_frac_var2"].mean()),
        "mean_initial_var2_frac": float(df_alpha["initial_frac_var2"].mean()),
        "mean_end_var2_frac": float(df_alpha["end_frac_var2"].mean()),

        "corr_alpha_vs_density_var1": _safe_corr(df_alpha["target_var1"], df_alpha["density_frac_var1"]),
        "corr_alpha_vs_step_var1": _safe_corr(df_alpha["target_var1"], df_alpha["step_frac_var1"]),
        "corr_alpha_vs_end_var1": _safe_corr(df_alpha["target_var1"], df_alpha["end_frac_var1"]),
        "corr_alpha_vs_initial_var1": _safe_corr(df_alpha["target_var1"], df_alpha["initial_frac_var1"]),

        "corr_prescribed_vs_estimated_alpha_end": _safe_corr(
            df_alpha["target_var1"], df_alpha["end_frac_var1"]
        ),
        "corr_prescribed_vs_estimated_alpha_density": _safe_corr(
            df_alpha["target_var1"], df_alpha["density_frac_var1"]
        ),

        "slope_prescribed_vs_estimated_alpha_end": _safe_slope(
            df_alpha["target_var1"], df_alpha["end_frac_var1"]
        ),
        "slope_prescribed_vs_estimated_alpha_density": _safe_slope(
            df_alpha["target_var1"], df_alpha["density_frac_var1"]
        ),

        "intercept_prescribed_vs_estimated_alpha_end": _safe_intercept(
            df_alpha["target_var1"], df_alpha["end_frac_var1"]
        ),
        "intercept_prescribed_vs_estimated_alpha_density": _safe_intercept(
            df_alpha["target_var1"], df_alpha["density_frac_var1"]
        ),
    }

    return {
        "trial_row": agg,
        "alpha_rows": rows_alpha,
        "member_rows": rows_members,
        "bundles": bundles,
    }


# -----------------------------------------------------------------------------
# Multiprocessing wrapper for one hyperparameter configuration
# -----------------------------------------------------------------------------
def evaluate_trial_worker(job, alpha_grid=ALPHA_MIX_GRID, M=30, seed=SEARCH_SEED):
    trial_id, trial = job
    t0 = time.perf_counter()

    try:
        out = evaluate_trial_across_alphas_full(
            trial_id=trial_id,
            trial=trial,
            alpha_grid=alpha_grid,
            M=M,
            seed=seed,
        )

        trial_row = out["trial_row"]
        trial_row["elapsed_sec"] = float(time.perf_counter() - t0)
        trial_row["ok"] = 1
        trial_row["error"] = ""

        return {
            "ok": True,
            "trial_row": trial_row,
            "alpha_rows": out["alpha_rows"],
            "member_rows": out["member_rows"],
            "bundles": out["bundles"],
        }

    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()

        trial_row = {
            "trial_id": int(trial_id),
            **copy.deepcopy(trial),

            "n_alpha": np.nan,
            "n_alpha_valid_physical": np.nan,
            "frac_alpha_valid_physical": np.nan,

            "mean_mae_initial": np.nan,
            "mean_mae_end": np.nan,
            "mean_mae_step": np.nan,
            "mean_mae_density": np.nan,

            "max_mae_initial": np.nan,
            "max_mae_end": np.nan,
            "max_mae_step": np.nan,
            "max_mae_density": np.nan,

            "mean_ok_frac": np.nan,
            "min_ok_frac": np.nan,
            "mean_mean_n_steps": np.nan,
            "min_q25_n_steps": np.nan,
            "mean_mean_net_displacement": np.nan,
            "mean_mean_path_length": np.nan,
            "mean_mean_stagnant_frac": np.nan,

            "mean_density_var2_frac": np.nan,
            "mean_step_var2_frac": np.nan,
            "mean_initial_var2_frac": np.nan,
            "mean_end_var2_frac": np.nan,

            "corr_alpha_vs_density_var1": np.nan,
            "corr_alpha_vs_step_var1": np.nan,
            "corr_alpha_vs_end_var1": np.nan,
            "corr_alpha_vs_initial_var1": np.nan,
            "corr_prescribed_vs_estimated_alpha_end": np.nan,
            "corr_prescribed_vs_estimated_alpha_density": np.nan,
            "slope_prescribed_vs_estimated_alpha_end": np.nan,
            "slope_prescribed_vs_estimated_alpha_density": np.nan,
            "intercept_prescribed_vs_estimated_alpha_end": np.nan,
            "intercept_prescribed_vs_estimated_alpha_density": np.nan,

            "elapsed_sec": float(time.perf_counter() - t0),
            "ok": 0,
            "error": err,
        }

        return {
            "ok": False,
            "trial_row": trial_row,
            "alpha_rows": [],
            "member_rows": [],
            "bundles": [],
        }


# -----------------------------------------------------------------------------
# Command-line orchestration and multiprocessing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Trace-ST multivariate synthetic hyperparameter search")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--n-trials", type=int, default=1000, help="Number of hyperparameter trials")
    parser.add_argument("--n-workers", type=int, default=120, help="Parallel workers")
    parser.add_argument("--ensemble-size", type=int, default=30, help="Ensemble members per alpha_mix")
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
    alphas_jsonl = outdir / "results_alphas.jsonl"
    members_jsonl = outdir / "results_members.jsonl"

    jobs_csv = outdir / "results_jobs_snapshot.csv"
    alphas_csv = outdir / "results_alphas_snapshot.csv"
    members_csv = outdir / "results_members_snapshot.csv"

    meta_path = outdir / "run_metadata.json"
    summary_path = outdir / "run_summary.json"

    all_jobs = build_multivar_jobs(n_trials=args.n_trials, seed=args.seed)

    completed_ids, existing_job_rows = load_completed_trial_ids(jobs_jsonl)
    pending_jobs = filter_pending_trials(all_jobs, completed_ids)

    existing_alpha_rows = load_jsonl_records(alphas_jsonl)
    existing_member_rows = load_jsonl_records(members_jsonl)

    meta = {
        "seed": int(args.seed),
        "n_trials": int(args.n_trials),
        "n_workers": int(args.n_workers),
        "ensemble_size": int(args.ensemble_size),
        "flush_every": int(args.flush_every),
        "alpha_mix_grid": [float(x) for x in ALPHA_MIX_GRID],
        "n_total_trials": len(all_jobs),
        "n_completed_trials_found_at_start": len(completed_ids),
        "n_pending_trials_at_start": len(pending_jobs),
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }
    atomic_write_json(meta_path, meta)

    if existing_job_rows:
        atomic_write_csv(jobs_csv, pd.DataFrame(existing_job_rows))
    if existing_alpha_rows:
        atomic_write_csv(alphas_csv, pd.DataFrame(existing_alpha_rows))
    if existing_member_rows:
        atomic_write_csv(members_csv, pd.DataFrame(existing_member_rows))

    print(f"Total trials: {len(all_jobs)}", flush=True)
    print(f"Completed:    {len(completed_ids)}", flush=True)
    print(f"Pending:      {len(pending_jobs)}", flush=True)

    if len(pending_jobs) == 0:
        print("Nothing to do. All trials already completed.", flush=True)

        final_jobs = pd.DataFrame(existing_job_rows)
        final_alphas = pd.DataFrame(existing_alpha_rows)
        final_members = pd.DataFrame(existing_member_rows)

        if len(final_jobs) > 0:
            sort_cols = [c for c in ["ok", "frac_alpha_valid_physical", "mean_mae_density", "mean_mae_step"] if c in final_jobs.columns]
            asc = [(False if c in ["ok", "frac_alpha_valid_physical"] else True) for c in sort_cols]
            final_jobs = final_jobs.sort_values(sort_cols, ascending=asc)
            atomic_write_csv(outdir / "results_jobs_final.csv", final_jobs)

        if len(final_alphas) > 0:
            atomic_write_csv(outdir / "results_alphas_final.csv", final_alphas)

        if len(final_members) > 0:
            atomic_write_csv(outdir / "results_members_final.csv", final_members)

        summary = {
            "finished_at": pd.Timestamp.utcnow().isoformat(),
            "elapsed_sec": 0.0,
            "n_trials_total": len(all_jobs),
            "n_trials_ok": int(final_jobs["ok"].sum()) if len(final_jobs) else 0,
            "n_trials_failed": int((1 - final_jobs["ok"]).sum()) if len(final_jobs) else 0,
            "resumed": True,
            "nothing_new_run": True,
        }
        atomic_write_json(summary_path, summary)
        return

    buffer_trial_rows = []
    buffer_alpha_rows = []
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
                ALPHA_MIX_GRID,
                args.ensemble_size,
                args.seed,
            )
            for job in pending_jobs
        ]

        for fut in as_completed(futures):
            result = fut.result()

            trial_row = result["trial_row"]
            alpha_rows = result["alpha_rows"]
            member_rows = result["member_rows"]
            bundles = result["bundles"]

            for bundle in bundles:
                npz_path, json_path = save_ensemble_bundle(bundle, bundles_dir=bundles_dir)

                trial_id = int(bundle["meta"]["trial_id"])
                alpha_mix = float(bundle["meta"]["alpha_mix"])

                for row in alpha_rows:
                    if int(row["trial_id"]) == trial_id and float(row["alpha_mix"]) == alpha_mix:
                        row["bundle_npz"] = str(npz_path)
                        row["bundle_json"] = str(json_path)

                for row in member_rows:
                    if int(row["trial_id"]) == trial_id and float(row["alpha_mix"]) == alpha_mix:
                        row["bundle_npz"] = str(npz_path)
                        row["bundle_json"] = str(json_path)

            buffer_trial_rows.append(trial_row)
            buffer_alpha_rows.extend(alpha_rows)
            buffer_member_rows.extend(member_rows)

            n_done_this_run += 1
            n_total_done_now += 1

            if (n_done_this_run % args.flush_every == 0) or (n_done_this_run == len(pending_jobs)):
                append_jsonl(jobs_jsonl, buffer_trial_rows)
                append_jsonl(alphas_jsonl, buffer_alpha_rows)
                append_jsonl(members_jsonl, buffer_member_rows)

                all_job_rows_now = load_jsonl_records(jobs_jsonl)
                all_alpha_rows_now = load_jsonl_records(alphas_jsonl)
                all_member_rows_now = load_jsonl_records(members_jsonl)

                df_jobs_now = pd.DataFrame(all_job_rows_now) if all_job_rows_now else pd.DataFrame()
                df_alphas_now = pd.DataFrame(all_alpha_rows_now) if all_alpha_rows_now else pd.DataFrame()
                df_members_now = pd.DataFrame(all_member_rows_now) if all_member_rows_now else pd.DataFrame()

                if len(df_jobs_now) > 0:
                    atomic_write_csv(jobs_csv, df_jobs_now)
                if len(df_alphas_now) > 0:
                    atomic_write_csv(alphas_csv, df_alphas_now)
                if len(df_members_now) > 0:
                    atomic_write_csv(members_csv, df_members_now)

                elapsed = time.perf_counter() - t0
                rate = n_done_this_run / elapsed if elapsed > 0 else np.nan

                n_ok_total = int(df_jobs_now["ok"].sum()) if len(df_jobs_now) > 0 and "ok" in df_jobs_now.columns else 0
                best_msg = "best=NA"

                if len(df_jobs_now) > 0 and "mean_mae_density" in df_jobs_now.columns:
                    good = df_jobs_now[df_jobs_now.get("ok", 0) == 1].copy()
                    good = good[good["mean_mae_density"].notna()]

                    if len(good) > 0:
                        sort_cols = [c for c in ["frac_alpha_valid_physical", "mean_mae_density", "mean_mae_step"] if c in good.columns]
                        asc = [(False if c == "frac_alpha_valid_physical" else True) for c in sort_cols]
                        good = good.sort_values(sort_cols, ascending=asc)
                        best = good.iloc[0]

                        valid_frac_txt = (
                            f"{best['frac_alpha_valid_physical']:.2f}"
                            if "frac_alpha_valid_physical" in good.columns and pd.notna(best["frac_alpha_valid_physical"])
                            else "NA"
                        )
                        best_msg = (
                            f"best_trial={int(best['trial_id'])} "
                            f"valid_frac={valid_frac_txt} "
                            f"mean_mae_density={best['mean_mae_density']:.4f}"
                        )

                print(
                    f"[run {n_done_this_run:4d}/{len(pending_jobs)} | total {n_total_done_now:4d}/{len(all_jobs)}] "
                    f"ok_total={n_ok_total:4d} fail_total={n_total_done_now - n_ok_total:4d} "
                    f"rate={rate:.2f} trials/s {best_msg}",
                    flush=True,
                )

                buffer_trial_rows = []
                buffer_alpha_rows = []
                buffer_member_rows = []

    all_job_rows_final = load_jsonl_records(jobs_jsonl)
    all_alpha_rows_final = load_jsonl_records(alphas_jsonl)
    all_member_rows_final = load_jsonl_records(members_jsonl)

    final_jobs = pd.DataFrame(all_job_rows_final)
    final_alphas = pd.DataFrame(all_alpha_rows_final)
    final_members = pd.DataFrame(all_member_rows_final)

    if len(final_jobs) > 0:
        sort_cols = [c for c in ["ok", "frac_alpha_valid_physical", "mean_mae_density", "mean_mae_step"] if c in final_jobs.columns]
        asc = [(False if c in ["ok", "frac_alpha_valid_physical"] else True) for c in sort_cols]
        final_jobs = final_jobs.sort_values(sort_cols, ascending=asc)
        atomic_write_csv(outdir / "results_jobs_final.csv", final_jobs)

    if len(final_alphas) > 0:
        atomic_write_csv(outdir / "results_alphas_final.csv", final_alphas)

    if len(final_members) > 0:
        atomic_write_csv(outdir / "results_members_final.csv", final_members)

    summary = {
        "finished_at": pd.Timestamp.utcnow().isoformat(),
        "elapsed_sec": float(time.perf_counter() - t0),
        "n_trials_total": len(all_jobs),
        "n_trials_ok": int(final_jobs["ok"].sum()) if len(final_jobs) and "ok" in final_jobs.columns else 0,
        "n_trials_failed": int((1 - final_jobs["ok"]).sum()) if len(final_jobs) and "ok" in final_jobs.columns else 0,
        "resumed": len(completed_ids) > 0,
        "n_trials_completed_before_this_run": len(completed_ids),
        "n_trials_completed_this_run": n_done_this_run,
    }
    atomic_write_json(summary_path, summary)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
