from __future__ import annotations

import os
import sys
import copy
import time
import json
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import trace_st as tst

# ============================================================
# Quick speed benchmark: Elastic Net vs PCMCI vs DYNOTEARS
# Multivariate synthetic case, 5-member ensemble
# ============================================================



import copy
import time
import json
import signal
from pathlib import Path

import numpy as np
import pandas as pd

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
OUTDIR_ELASTICNET = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/ResultsSynthetic/multivariate_v4")
OUTDIR_PCMCI = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/ResultsSynthetic/multivariate_v5")

SPEED_OUTDIR = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/ResultsSynthetic/speed_benchmark_v1")
SPEED_OUTDIR.mkdir(parents=True, exist_ok=True)

ALPHA_TEST = 0.6
M_TEST = 1
N_STEPS_TIME = 30
TIMEOUT_PER_METHOD_SEC = 2 * 3600  # 2 hours per method; adjust if desired
GLOBAL_SEED = 11
SEARCH_SEED = 11


# ------------------------------------------------------------
# Timeout helper
# ------------------------------------------------------------
class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("Timed out")


class time_limit:
    def __init__(self, seconds):
        self.seconds = int(seconds)
        self.old_handler = None

    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old_handler)
        return False


# ------------------------------------------------------------
# Read best rows
# ------------------------------------------------------------
def read_best_elasticnet_row(outdir):
    jobs_path = outdir / "results_jobs_final.csv"
    if not jobs_path.exists():
        jobs_path = outdir / "results_jobs_snapshot.csv"

    jobs = pd.read_csv(jobs_path)

    sort_cols = [
        c for c in [
            "frac_alpha_valid_physical",
            "mean_mae_density",
            "mean_mae_step",
            "mean_density_var2_frac",
        ]
        if c in jobs.columns
    ]
    asc = [False if c == "frac_alpha_valid_physical" else True for c in sort_cols]

    return jobs.sort_values(sort_cols, ascending=asc).iloc[0].copy()


def read_best_pcmci_row(outdir):
    jobs_path = outdir / "results_jobs_final.csv"
    if not jobs_path.exists():
        jobs_path = outdir / "results_jobs_snapshot.csv"

    jobs = pd.read_csv(jobs_path)

    if "method" in jobs.columns:
        jobs = jobs[jobs["method"].astype(str).str.lower() == "pcmci"].copy()

    sort_cols = [
        c for c in [
            "frac_alpha_valid_physical",
            "mean_mae_density",
            "mean_mae_step",
            "mean_density_var2_frac",
        ]
        if c in jobs.columns
    ]
    asc = [False if c == "frac_alpha_valid_physical" else True for c in sort_cols]

    return jobs.sort_values(sort_cols, ascending=asc).iloc[0].copy()


best_elasticnet_row = read_best_elasticnet_row(OUTDIR_ELASTICNET)
best_pcmci_row = read_best_pcmci_row(OUTDIR_PCMCI)

print("Best Elastic Net trial:", int(best_elasticnet_row["trial_id"]))
print("Best PCMCI trial:", int(best_pcmci_row["trial_id"]))


# ------------------------------------------------------------
# Parameter builders
# Requires build_base_params_multivar or equivalent helpers from earlier notebook/script.
# If not loaded, use these standalone builders.
# ------------------------------------------------------------
def build_base_params_elasticnet_multivar():
    return dict(
        timeres="1d",
        spaceres=1,
        box_size=20,
        radius=2,
        starting_lat=None,
        starting_lon=None,
        timewindow="4d",
        child_of_interest=2,
        n_steps_time=N_STEPS_TIME,
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
        verbose=True,
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


def build_base_params_pcmci_multivar():
    return dict(
        timeres="1d",
        spaceres=1,
        box_size=20,
        radius=2,
        starting_lat=None,
        starting_lon=None,
        timewindow="4d",
        child_of_interest=2,
        n_steps_time=N_STEPS_TIME,
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
        verbose=True,
        winner_summary="mean",
        prob_rule="softmax",
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


def build_base_params_dynotears_multivar():
    return dict(
        timeres="1d",
        spaceres=1,
        box_size=20,
        radius=2,
        starting_lat=None,
        starting_lon=None,
        timewindow="4d",
        child_of_interest=2,
        n_steps_time=N_STEPS_TIME,
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
        verbose=True,
        winner_summary="mean",
        prob_rule="softmax",
        cd_method="dynotears",
        cd_kwargs=dict(
            lambda_a=0.01,
            max_iter=100,
            dependence_threshold=1e-3,
            strength_threshold=None,
            verbose=0,
            allow_center_directed_links=True,
            dependencies_wrap=False,
        ),
    )


def build_cond_ind_test_from_name(name):
    name = str(name).lower()
    if name == "parcorr":
        return ParCorr()
    if name == "robustparcorr":
        return RobustParCorr()
    raise ValueError(f"Unknown cond_ind_test_name: {name}")


# ------------------------------------------------------------
# Build all methods using the SAME shared TraCE-ST hyperparameters
# Use best Elastic Net shared hyperparameters as reference.
# Only causal-discovery-specific kwargs differ.
# ------------------------------------------------------------

SHARED_REFERENCE_ROW = best_elasticnet_row.copy()

def apply_shared_hyperparams(p, row):
    shared_keys = [
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
    ]

    for k in shared_keys:
        if k in row.index and not pd.isna(row[k]):
            p[k] = row[k]

    # Force smaller/faster common TraCE-ST geometry for speed benchmark
    p["timewindow"] = "4d"
    p["box_size"] = 20
    p["radius"] = 3

    p["child_of_interest"] = 2
    p["n_steps_time"] = N_STEPS_TIME
    p["montecarlo"] = True
    p["snap_to_grid"] = False
    p["averaging_winners"] = True
    p["verbose"] = True

    return p


def params_elasticnet_same_shared(row_shared, row_method):
    p = build_base_params_elasticnet_multivar()
    p = apply_shared_hyperparams(p, row_shared)

    for k in [
        "lambda_a",
        "l1_ratio",
        "dependence_threshold",
        "max_iter",
        "fit_intercept",
        "refit_ridge",
        "ridge_alpha",
    ]:
        if k in row_method.index and not pd.isna(row_method[k]):
            p["cd_kwargs"][k] = row_method[k]

    p["cd_method"] = "granger"
    return p


def params_pcmci_same_shared(row_shared, row_method):
    p = build_base_params_pcmci_multivar()
    p = apply_shared_hyperparams(p, row_shared)

    if "pc_alpha" in row_method.index and not pd.isna(row_method["pc_alpha"]):
        p["cd_kwargs"]["pc_alpha"] = float(row_method["pc_alpha"])

    if "graph_p_threshold" in row_method.index and not pd.isna(row_method["graph_p_threshold"]):
        p["cd_kwargs"]["graph_p_threshold"] = float(row_method["graph_p_threshold"])

    if "fdr_method" in row_method.index:
        p["cd_kwargs"]["fdr_method"] = None if pd.isna(row_method["fdr_method"]) else row_method["fdr_method"]

    if "cond_ind_test_name" in row_method.index and not pd.isna(row_method["cond_ind_test_name"]):
        p["cd_kwargs"]["cond_ind_test"] = build_cond_ind_test_from_name(row_method["cond_ind_test_name"])

    p["cd_method"] = "pcmci"
    return p


def params_dynotears_same_shared(row_shared):
    p = build_base_params_dynotears_multivar()
    p = apply_shared_hyperparams(p, row_shared)

    # Conservative fast DYNOTEARS settings for timing benchmark
    p["cd_kwargs"]["lambda_a"] = 0.01
    p["cd_kwargs"]["max_iter"] = 100
    p["cd_kwargs"]["dependence_threshold"] = 1e-3

    p["cd_method"] = "dynotears"
    return p


params_elasticnet = params_elasticnet_same_shared(
    row_shared=SHARED_REFERENCE_ROW,
    row_method=best_elasticnet_row,
)

params_pcmci = params_pcmci_same_shared(
    row_shared=SHARED_REFERENCE_ROW,
    row_method=best_pcmci_row,
)

params_dynotears = params_dynotears_same_shared(
    row_shared=SHARED_REFERENCE_ROW,
)

print("\nShared hyperparameters used by all methods:")
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
    print(
        f"{k:22s}",
        params_elasticnet[k],
        params_pcmci[k],
        params_dynotears[k],
    )


# -----------------------------------------------------------------------------
# Synthetic helpers
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
            y2d,
            x2d,
            y0=centers_A[t, 0],
            x0=centers_A[t, 1],
            sig_y=sig_y,
            sig_x=sig_x,
            amp=amp_A,
        ).astype(np.float32)

        C_clean[t] = gaussian_blob(
            y2d,
            x2d,
            y0=centers_C[t, 0],
            x0=centers_C[t, 1],
            sig_y=sig_y,
            sig_x=sig_x,
            amp=amp_C,
        ).astype(np.float32)

    rng_A = np.random.default_rng(seed + 0)
    rng_C = np.random.default_rng(seed + 2)

    noise_A = make_spatiotemporal_ar1_noise(
        A_clean.shape,
        phi=phi_A,
        sigma=sigma_A,
        rng=rng_A,
    )
    noise_C = make_spatiotemporal_ar1_noise(
        C_clean.shape,
        phi=phi_C,
        sigma=sigma_C,
        rng=rng_C,
    )

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
    noise_B = make_spatiotemporal_ar1_noise(
        (T, NY, NX),
        phi=phi_B,
        sigma=sigma_B,
        rng=rng_B,
    )

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
# Run helpers
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


def run_ensemble(case, base_params, M=20, seed=123):
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
            "member_id": int(m),
            "ok": bool(ok),
            "error": err,
            "elapsed_sec": float(time.perf_counter() - t0),
            "centers": np.asarray(centers, dtype=float),
            "parents": np.asarray(parents, dtype=int),
            "out": out,
        })

    return {"members": members, "params": copy.deepcopy(base_params), "M": int(M)}



# ------------------------------------------------------------
# Benchmark runner
# Requires build_multivariate_case and run_ensemble from earlier notebook/script.
# ------------------------------------------------------------
def summarize_timing_members(ensemble_out):
    member_times = [
        float(m.get("elapsed_sec", np.nan))
        for m in ensemble_out["members"]
    ]
    ok = [
        bool(m.get("ok", False))
        for m in ensemble_out["members"]
    ]

    return dict(
        n_members=len(member_times),
        n_ok=int(np.sum(ok)),
        ok_frac=float(np.mean(ok)) if len(ok) else np.nan,
        member_time_mean_sec=float(np.nanmean(member_times)) if len(member_times) else np.nan,
        member_time_median_sec=float(np.nanmedian(member_times)) if len(member_times) else np.nan,
        member_time_min_sec=float(np.nanmin(member_times)) if len(member_times) else np.nan,
        member_time_max_sec=float(np.nanmax(member_times)) if len(member_times) else np.nan,
    )


def benchmark_one_method(name, params, case, timeout_sec=TIMEOUT_PER_METHOD_SEC):
    t0 = time.perf_counter()

    try:
        with time_limit(timeout_sec):
            ensemble_out = run_ensemble(
                case,
                params,
                M=M_TEST,
                seed=SEARCH_SEED + 999,
            )

        elapsed = time.perf_counter() - t0
        stats = summarize_timing_members(ensemble_out)

        row = dict(
            method=name,
            status="completed",
            alpha_mix=ALPHA_TEST,
            M=M_TEST,
            n_steps_time=N_STEPS_TIME,
            total_elapsed_sec=float(elapsed),
            total_elapsed_min=float(elapsed / 60.0),
            total_elapsed_hr=float(elapsed / 3600.0),
            timeout_sec=float(timeout_sec),
            error="",
            **stats,
        )

        return row, ensemble_out

    except TimeoutException as e:
        elapsed = time.perf_counter() - t0

        row = dict(
            method=name,
            status="timeout",
            alpha_mix=ALPHA_TEST,
            M=M_TEST,
            n_steps_time=N_STEPS_TIME,
            total_elapsed_sec=float(elapsed),
            total_elapsed_min=float(elapsed / 60.0),
            total_elapsed_hr=float(elapsed / 3600.0),
            timeout_sec=float(timeout_sec),
            error=str(e),
            n_members=M_TEST,
            n_ok=np.nan,
            ok_frac=np.nan,
            member_time_mean_sec=np.nan,
            member_time_median_sec=np.nan,
            member_time_min_sec=np.nan,
            member_time_max_sec=np.nan,
        )

        return row, None

    except Exception as e:
        elapsed = time.perf_counter() - t0

        row = dict(
            method=name,
            status="error",
            alpha_mix=ALPHA_TEST,
            M=M_TEST,
            n_steps_time=N_STEPS_TIME,
            total_elapsed_sec=float(elapsed),
            total_elapsed_min=float(elapsed / 60.0),
            total_elapsed_hr=float(elapsed / 3600.0),
            timeout_sec=float(timeout_sec),
            error=repr(e),
            n_members=M_TEST,
            n_ok=np.nan,
            ok_frac=np.nan,
            member_time_mean_sec=np.nan,
            member_time_median_sec=np.nan,
            member_time_min_sec=np.nan,
            member_time_max_sec=np.nan,
        )

        return row, None


case = build_multivariate_case(
    alpha_mix=ALPHA_TEST,
    T=40,
    N_inject=8,
    seed=GLOBAL_SEED,
)

benchmarks = [
    ("elasticnet_granger", params_elasticnet),
    ("pcmci", params_pcmci),
    ("dynotears_fast", params_dynotears),
]

rows = []
outs = {}

for name, params in benchmarks:
    print(f"\nRunning {name}...")
    print("cd_method:", params["cd_method"])
    print("params:", {k: params[k] for k in ["timewindow", "box_size", "radius", "prob_rule", "beta_softmax"]})
    print("cd_kwargs:", params["cd_kwargs"])

    row, out = benchmark_one_method(
        name,
        params,
        case,
        timeout_sec=TIMEOUT_PER_METHOD_SEC,
    )

    rows.append(row)
    outs[name] = out

    print(pd.DataFrame([row]).T)

df_speed = pd.DataFrame(rows)

print(df_speed)

speed_csv = SPEED_OUTDIR / "speed_benchmark_multivar_M5.csv"
df_speed.to_csv(speed_csv, index=False)

with open(SPEED_OUTDIR / "speed_benchmark_multivar_M5.json", "w") as f:
    json.dump(rows, f, indent=2, default=str)

print("Saved:")
print(speed_csv)
print(SPEED_OUTDIR / "speed_benchmark_multivar_M5.json")