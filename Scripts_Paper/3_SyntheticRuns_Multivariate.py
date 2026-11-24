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
import contextlib
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
# night-2 D4 (opt-in): stability-selection settings; read from the environment so spawned workers see it
STABILITY = json.loads(os.environ["TST_STABILITY"]) if os.environ.get("TST_STABILITY") else None

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


def _build_extra_vars(A, C, event, time_coord, ys, xs, x2d, y2d, T, t0, *, n_noise, n_decoy, n_moving,
                      sig_y, sig_x, seed, clip_to_positive=True, decoy_phi=0.8, decoy_beta=0.8, decoy_sigma=0.1,
                      moving_placement="event", moving_offset=15.0, moving_tc_lag=2.0, decoy_driver="parent", B=None,
                      decoy_lag=1, decoy_nonlinear=None, extra_mix=None, mix_beta=None, t_pad=0):
    """Revision-1 T7 non-causal variables (none of them enters B's equation).

    Night-5 T3 (opt-in; defaults = previous behaviour):
    decoy_lag L     : decoys driven at lag L: D[t] = phi D[t-1] + beta X[t-L] + noise (no drive for t < L).
    decoy_nonlinear : "rectified" drives with beta * max(X[t-L] - 0.5, 0), i.e. only the blob core
                      (A, C have blob amplitude 1 and are clipped at 0, so max(X, 0) would equal X).
    extra_mix       : list of types appended after the n_* variables, in the given order, each with its own
                      counter k and seed: "noise" (as noise, seed+100+k, N<k>), "response" (driven by B at
                      lag 1, seed+700+k, R<k>_B), "moving" (random placement, seed+400+k, M<k>), "delayed"
                      (driven by A (k even) or C (k odd) at lag 2, seed+500+k, L<k>_A/C), "nonlinear"
                      (rectified drive by A/C at lag 1, seed+600+k, Q<k>_A/C). Decoys use decoy_phi,
                      decoy_beta and decoy_sigma; noise and moving fields as above.

    Night-4 C (opt-in): decoy_driver="child" makes every decoy a response decoy driven by the child B:
    D[t] = phi D[t-1] + beta B[t-1] + AR(1) noise (same noise, seed and clipping), named "D<k>_B".
    Default "parent" = the original decoys (driven by A or C).

    noise k  : independent smoothed AR(1) field, same construction as the A/C noise
               (make_spatiotemporal_ar1_noise, phi=0.85, sigma=0.10), seed + 100 + k.
    decoy k  : driven by A (k even) or C (k odd) with the T1 (case-1) V2 equation
               D[t] = 0.8 D[t-1] + 0.8 X[t-1] + AR(1) noise (phi 0.8, sigma 0.1), seed + 200 + k
               (defaults; --decoy-phi / --decoy-beta / --decoy-sigma change 0.8 / 0.8 / 0.1).
    moving k : independent Gaussian blob (same size/amplitude as A, C) on a straight Bezier
               track whose closest approach (15 units from the event, perpendicular offset)
               is at t = t0 - 2, i.e. within ~20 units of the event during t0-5..t0;
               approach direction 45 deg + 360 k / n_moving; speed 2.5 units/day; plus A/C-type
               AR(1) noise, seed + 300 + k. Not coupled to B.
               Night-2 opt-ins (defaults = above): moving_offset (closest-approach distance, 15),
               moving_tc_lag (closest approach at t0 - lag, 2); moving_placement="random": each blob
               gets a random straight track (direction U(0, 360), passing a point uniform in
               [20, 180]^2 at a time uniform in [0, T-1], same speed), rng seed + 400 + k, not aimed
               at the event.
    """
    out, info = [], []
    ey, ex = float(event[0]), float(event[1])
    for k in range(int(n_noise)):
        arr = make_spatiotemporal_ar1_noise(A.shape, phi=0.85, sigma=0.10, rng=np.random.default_rng(seed + 100 + k))
        name = f"N{k + 1}"
        out.append(build_dataarray(name, arr, time_coord, ys, xs)); info.append((name, "noise"))
    for k in range(int(n_decoy)):
        X = A if k % 2 == 0 else C
        if decoy_driver == "child":
            if B is None:
                raise ValueError("decoy_driver=child needs B")
            X = B
        elif decoy_driver != "parent":
            raise ValueError(f"decoy_driver {decoy_driver!r}")
        nz = make_spatiotemporal_ar1_noise(A.shape, phi=0.8, sigma=float(decoy_sigma), rng=np.random.default_rng(seed + 200 + k))
        D = np.zeros_like(A, dtype=np.float32)
        D[0] = nz[0]
        if int(decoy_lag) == 1 and not decoy_nonlinear:   # original equation (default)
            for t in range(1, T):
                D[t] = float(decoy_phi) * D[t - 1] + float(decoy_beta) * X[t - 1] + nz[t]
        else:
            D = _drive_decoy(X, nz, T, decoy_phi, decoy_beta, int(decoy_lag), decoy_nonlinear)
        if clip_to_positive:
            D = np.clip(D, 0.0, None)
        name = f"D{k + 1}_{'A' if k % 2 == 0 else 'C'}" if decoy_driver == "parent" else f"D{k + 1}_B"
        out.append(build_dataarray(name, D, time_coord, ys, xs)); info.append((name, "decoy"))
    for k in range(int(n_moving)):
        if moving_placement == "random":
            rng_m = np.random.default_rng(seed + 400 + k)
            th = float(rng_m.uniform(0.0, 2.0 * np.pi))
            d = np.array([np.sin(th), np.cos(th)])
            q = rng_m.uniform(20.0, 180.0, size=2)
            tc = float(rng_m.uniform(0.0, T - 1))
        else:
            th = np.deg2rad(45.0 + 360.0 * k / max(int(n_moving), 1))
            d = np.array([np.sin(th), np.cos(th)])          # (dy, dx) direction of motion
            perp = np.array([-d[1], d[0]])
            tc = float(t0 - float(moving_tc_lag))
            q = np.array([ey, ex]) + float(moving_offset) * perp
        start = q - 2.5 * tc * d
        end = q + 2.5 * (T - 1 - tc) * d
        cen = bezier_centers(start, start + (end - start) / 3, start + 2 * (end - start) / 3, end, T)
        clean = np.stack([gaussian_blob(y2d, x2d, cen[t, 0], cen[t, 1], sig_y=sig_y, sig_x=sig_x, amp=1.0)
                          for t in range(T)]).astype(np.float32)
        nz = make_spatiotemporal_ar1_noise(A.shape, phi=0.85, sigma=0.10, rng=np.random.default_rng(seed + 300 + k))
        M_ = clean + nz
        if clip_to_positive:
            M_ = np.clip(M_, 0.0, None)
        name = f"M{k + 1}"
        out.append(build_dataarray(name, M_, time_coord, ys, xs)); info.append((name, "moving"))
    if extra_mix:
        o2, i2 = _build_mix_vars(A, C, B, time_coord, ys, xs, x2d, y2d, T, list(extra_mix), sig_y=sig_y, sig_x=sig_x,
                                 seed=seed, clip_to_positive=clip_to_positive, decoy_phi=decoy_phi,
                                 decoy_beta=decoy_beta, decoy_sigma=decoy_sigma,
                                 n_moving_prev=int(n_moving) if moving_placement == "random" else 0,
                                 **({"mix_beta": dict(mix_beta)} if mix_beta else {}),
                                 **({"t_pad": int(t_pad)} if t_pad else {}))
        out += o2; info += i2
    return out, info


def _drive_decoy(X, nz, T, phi, beta, lag, nonlinear):
    """Night 5: D[t] = phi D[t-1] + beta f(X[t-lag]) + nz[t]; f = identity or rectified max(X - 0.5, 0)."""
    D = np.zeros_like(nz, dtype=np.float32)
    D[0] = nz[0]
    for t in range(1, T):
        drive = 0.0
        if t - lag >= 0:
            x = X[t - lag]
            if nonlinear == "rectified":
                x = np.maximum(x - 0.5, 0.0)
            elif nonlinear:
                raise ValueError(f"decoy_nonlinear {nonlinear!r}")
            drive = float(beta) * x
        D[t] = float(phi) * D[t - 1] + drive + nz[t]
    return D


def _typical_mix_field(A, B, C, k, clip):
    """Night 8: typical field of the case (median statistics of V1-V3, independent, no lagged links), realization k
    (seed typical_seed(4, 100 + k), data_processing/surrogates/make_linkless_synthetic.py)."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "data_processing" / "surrogates" / "make_linkless_synthetic.py"
    spec = importlib.util.spec_from_file_location("make_linkless_synthetic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tg = mod.typical_targets([np.asarray(A), np.asarray(B), np.asarray(C)])
    # not clipped: clipping a field built from the clipped variables' own mean and sd removes about half of its variance
    return mod.typical_field(tg, mod.typical_seed(4, 100 + int(k)), clip=False)


def _build_mix_vars(A, C, B, time_coord, ys, xs, x2d, y2d, T, mix, *, sig_y, sig_x, seed, clip_to_positive,
                    decoy_phi, decoy_beta, decoy_sigma, n_moving_prev=0, mix_beta=None, t_pad=0):
    """Night-5 T3 composition (see _build_extra_vars, extra_mix). Returns (dataarrays, [(name, type)]).
    mix_beta: optional {type: beta} overriding decoy_beta for that decoy type (rule-4 fairness adjustment).
    t_pad (night 6): T includes t_pad prepended days; moving tracks are drawn for T - t_pad and held at
    their first position during the pad (see build_multivariate_case)."""
    mix_beta = dict(mix_beta or {})
    out, info, cnt = [], [], {}
    for typ in mix:
        k = cnt.get(typ, 0); cnt[typ] = k + 1
        if typ == "noise":
            arr = make_spatiotemporal_ar1_noise(A.shape, phi=0.85, sigma=0.10, rng=np.random.default_rng(seed + 100 + k))
            name = f"N{k + 1}"
        elif typ == "moving":
            kk = k + int(n_moving_prev)
            Tm = T - int(t_pad)
            rng_m = np.random.default_rng(seed + 400 + kk)
            th = float(rng_m.uniform(0.0, 2.0 * np.pi))
            d = np.array([np.sin(th), np.cos(th)])
            q = rng_m.uniform(20.0, 180.0, size=2)
            tc = float(rng_m.uniform(0.0, Tm - 1))
            start = q - 2.5 * tc * d
            end = q + 2.5 * (Tm - 1 - tc) * d
            cen = bezier_centers(start, start + (end - start) / 3, start + 2 * (end - start) / 3, end, Tm)
            if t_pad:
                cen = np.concatenate([np.repeat(cen[:1], int(t_pad), axis=0), cen])
            clean = np.stack([gaussian_blob(y2d, x2d, cen[t, 0], cen[t, 1], sig_y=sig_y, sig_x=sig_x, amp=1.0)
                              for t in range(T)]).astype(np.float32)
            nz = make_spatiotemporal_ar1_noise(A.shape, phi=0.85, sigma=0.10, rng=np.random.default_rng(seed + 300 + kk))
            arr = clean + nz
            if clip_to_positive:
                arr = np.clip(arr, 0.0, None)
            name = f"M{kk + 1}"
        elif typ == "typical":   # night 8 (opt-in): independent linkless field with the median statistics of A, B, C
            arr = _typical_mix_field(A, B, C, k, clip_to_positive)
            name = f"Z{k + 1}"
        elif typ in ("response", "delayed", "nonlinear"):
            off = {"response": 700, "delayed": 500, "nonlinear": 600}[typ]
            nz = make_spatiotemporal_ar1_noise(A.shape, phi=0.8, sigma=float(decoy_sigma), rng=np.random.default_rng(seed + off + k))
            if typ == "response":
                X, lag, nl, name = B, 1, None, f"R{k + 1}_B"
            else:
                drv = "A" if k % 2 == 0 else "C"
                X = A if drv == "A" else C
                lag, nl = (2, None) if typ == "delayed" else (1, "rectified")
                name = f"{'L' if typ == 'delayed' else 'Q'}{k + 1}_{drv}"
            arr = _drive_decoy(X, nz, T, decoy_phi, float(mix_beta.get(typ, decoy_beta)), lag, nl)
            if clip_to_positive:
                arr = np.clip(arr, 0.0, None)
        else:
            raise ValueError(f"extra_mix type {typ!r}")
        out.append(build_dataarray(name, arr, time_coord, ys, xs)); info.append((name, typ))
    return out, info


def summarize_all_vars(case, ensemble_out, params):
    """Revision-1 T7 metrics over all p variables (the original summary only counts vars 1-3)."""
    xs, ys = case["xs"], case["ys"]
    names = [str(v) for v in case["full_data_run"]["var"].values]
    nv = len(names)
    types = ["cause", "target", "cause"] + [t for _, t in case.get("extra_vars", [])]
    dens = np.zeros(nv)
    end_counts = np.zeros(nv)
    T = int(case["T"])
    truth = {1: case["centers_A"], 3: case["centers_C"]}
    errs, elapsed, full = [], [], []
    for m in ensemble_out["members"]:
        if not m["ok"] or len(m["parents"]) == 0:
            continue
        c = np.asarray(m["centers"], float); pa = np.asarray(m["parents"], int)
        elapsed.append(float(m["elapsed_sec"])); full.append(len(c) == int(params["n_steps_time"]) + 1)
        for v in range(1, nv + 1):
            mask = pa == v
            if mask.any():
                dens[v - 1] += _accumulate_density_box(np.zeros((len(ys), len(xs))), xs, ys, c[mask],
                                                       float(params["box_size"])).sum()
        end_counts[pa[-1] - 1] += 1
        for kstep in range(1, len(pa)):
            v = int(pa[kstep]); ti = T - 1 - kstep
            if v in truth and ti >= 0:
                errs.append(float(np.hypot(*(c[kstep] - truth[v][ti]))))
    tot = dens.sum()
    a = float(case["alpha_mix"])
    d13 = dens[0] + dens[2]
    e13 = end_counts[0] + end_counts[2]
    box = tst.get_data_mcastle(list(case["event"]), float(params["box_size"]),
                               [case["full_data_run"].sel(var=v).isel(time=slice(0, int(pd.to_timedelta(params["timewindow"]) / pd.to_timedelta(params["timeres"]))))
                                for v in case["full_data_run"]["var"].values])
    r = int(params["radius"]); _, ny, nx, tw = box.shape
    n_samples = max(ny - 2 * r, 0) * max(nx - 2 * r, 0) * (tw - 1)
    n_features = nv * (2 * r + 1) ** 2
    row = {
        "t7_p": nv,
        "t7_n_noise": types.count("noise"), "t7_n_decoy": types.count("decoy"), "t7_n_moving": types.count("moving"),
        "t7_share_density_v1v3": float(d13 / tot) if tot > 0 else np.nan,
        "t7_share_density_target": float(dens[1] / tot) if tot > 0 else np.nan,
        "t7_share_density_noise": float(sum(dens[i] for i, t in enumerate(types) if t == "noise") / tot) if tot > 0 else np.nan,
        "t7_share_density_decoy": float(sum(dens[i] for i, t in enumerate(types) if t == "decoy") / tot) if tot > 0 else np.nan,
        "t7_share_density_moving": float(sum(dens[i] for i, t in enumerate(types) if t == "moving") / tot) if tot > 0 else np.nan,
        "t7_alpha_rec_density_13": float(dens[0] / d13) if d13 > 0 else np.nan,
        "t7_alpha_abs_err_density_13": float(abs(dens[0] / d13 - a)) if d13 > 0 else np.nan,
        "t7_share_end_v1v3": float(e13 / end_counts.sum()) if end_counts.sum() > 0 else np.nan,
        "t7_alpha_rec_end_13": float(end_counts[0] / e13) if e13 > 0 else np.nan,
        "t7_alpha_abs_err_end_13": float(abs(end_counts[0] / e13 - a)) if e13 > 0 else np.nan,
        "t7_track_error_mean": float(np.mean(errs)) if errs else np.nan,
        "t7_track_error_median": float(np.median(errs)) if errs else np.nan,
        "t7_completion_rate": float(np.mean(full)) if full else 0.0,
        "t7_sec_per_traj": float(np.mean(elapsed)) if elapsed else np.nan,
        "t7_n_samples": int(n_samples), "t7_n_features": int(n_features),
        "t7_samples_per_feature": float(n_samples / n_features),
    }
    for i, nm in enumerate(names):
        row[f"t7_density_{nm}"] = float(dens[i])
    for typ in ("response", "delayed", "nonlinear", "typical"):   # night-5 extra_mix types (typical: night 8) (columns only when present)
        if typ in types:
            row[f"t7_n_{typ}"] = types.count(typ)
            row[f"t7_share_density_{typ}"] = (float(sum(dens[i] for i, t in enumerate(types) if t == typ) / tot)
                                              if tot > 0 else np.nan)
    return row


def _linkless_full_data(full, time_coord, ys, xs, clip, variant="A", draw=0, log_dir=None):
    """Night 7 T3b (opt-in): every variable replaced by a surrogate of a linkless system with the same per-cell mean and
    sd, AR(1) memory, spatial e-folding lengths and (variant "A") lag-0 correlation matrix, no lagged links
    (data_processing/surrogates/make_linkless_synthetic.py; variant "indep": R = identity)."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "data_processing" / "surrogates" / "make_linkless_synthetic.py"
    spec = importlib.util.spec_from_file_location("make_linkless_synthetic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = [str(v) for v in full["var"].values]
    arrs = [full.sel(var=n).values for n in names]
    out, _ = mod.linkless_system(arrs, names, int(draw), variant=variant, clip=bool(clip), log_dir=log_dir)
    return xr.concat([build_dataarray(n, a, time_coord, ys, xs) for n, a in zip(names, out)], dim="var")


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
    n_noise=0,
    n_decoy=0,
    n_moving=0,
    decoy_phi=0.8,
    decoy_beta=0.8,
    decoy_sigma=0.1,
    target_driver="clean",
    parent_sigma=None,
    moving_placement="event",
    moving_offset=15.0,
    moving_tc_lag=2.0,
    decoy_driver="parent",
    decoy_lag=1,
    decoy_nonlinear=None,
    extra_mix=None,
    mix_beta=None,
    t_pad=0,
    linkless=None,
):
    """n_noise / n_decoy / n_moving > 0 append non-causal variables after A, B, C
    (revision-1 T7; defaults 0 = original 3-variable case). See _build_extra_vars.
    Night-2 opt-ins (defaults = original): target_driver="observed" drives B with the observed
    (noisy, clipped) A and C instead of A_clean / C_clean; parent_sigma sets sigma_A = sigma_C.
    Night-6 T4 opt-in t_pad (default 0 = original): t_pad days are prepended so that longer windows fit
    (fixed-ratio curve). The blob tracks (A, C and the extra_mix moving blobs) are computed for the original T
    and held at their first position during the pad; all noise and decoy recursions run over T + t_pad;
    B is driven in the last N_inject days as before. case["T"] = T + t_pad."""
    if parent_sigma is not None:
        sigma_A = sigma_C = float(parent_sigma)
    time_coord = pd.date_range("2001-01-01", periods=T + int(t_pad), freq="D")

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
    if t_pad:   # night-6 T4 (opt-in): blobs held at their first position during the prepended days
        if int(n_moving) > 0:
            raise ValueError("t_pad is implemented for extra_mix moving blobs only (n_moving must be 0)")
        centers_A = np.concatenate([np.repeat(centers_A[:1], int(t_pad), axis=0), centers_A])
        centers_C = np.concatenate([np.repeat(centers_C[:1], int(t_pad), axis=0), centers_C])
        T = T + int(t_pad)

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

    if target_driver == "observed":
        drv_A, drv_C = A, C
    elif target_driver == "clean":
        drv_A, drv_C = A_clean, C_clean
    else:
        raise ValueError(f"target_driver {target_driver!r}")
    for t in range(t0, T):
        B[t] = (
            alpha_mix * drv_A[t - tau_mix]
            + (1.0 - alpha_mix) * drv_C[t - tau_mix]
            + noise_B[t]
        )
    if clip_to_positive:
        B = np.clip(B, 0.0, None)

    full_data_A = build_dataarray("A", A, time_coord, ys, xs)
    full_data_B = build_dataarray("B", B, time_coord, ys, xs)
    full_data_C = build_dataarray("C", C, time_coord, ys, xs)

    extra_list, extra_info = [], []
    if (n_noise + n_decoy + n_moving) > 0 or extra_mix:
        n5kw = {}   # night-5 opt-ins, passed only when not default
        if int(decoy_lag) != 1:
            n5kw["decoy_lag"] = int(decoy_lag)
        if decoy_nonlinear:
            n5kw["decoy_nonlinear"] = str(decoy_nonlinear)
        if extra_mix:
            n5kw["extra_mix"] = list(extra_mix)
            if mix_beta:
                n5kw["mix_beta"] = dict(mix_beta)
            if decoy_driver == "parent":   # (the child-decoy path already passes B)
                n5kw["B"] = B
            if t_pad:
                n5kw["t_pad"] = int(t_pad)
        extra_list, extra_info = _build_extra_vars(
            A, C, (event_y, event_x), time_coord, ys, xs, x2d, y2d, T, t0,
            n_noise=n_noise, n_decoy=n_decoy, n_moving=n_moving,
            sig_y=sig_y, sig_x=sig_x, seed=seed, clip_to_positive=clip_to_positive,
            decoy_phi=decoy_phi, decoy_beta=decoy_beta, decoy_sigma=decoy_sigma,
            moving_placement=moving_placement, moving_offset=moving_offset, moving_tc_lag=moving_tc_lag,
            **({} if decoy_driver == "parent" else dict(decoy_driver=decoy_driver, B=B)),
            **n5kw,
        )

    full_data_run = xr.concat([full_data_A, full_data_B, full_data_C] + extra_list, dim="var")
    if linkless:   # night-7 T3b (opt-in): the whole candidate set replaced by a linkless system (null of approach A)
        full_data_run = _linkless_full_data(full_data_run, time_coord, ys, xs, clip_to_positive, **dict(linkless))
    full_data_run = full_data_run.rename({"y": "lat", "x": "lon"})
    date_end = str(time_coord[-1].date())

    case = dict(
        extra_vars=extra_info,
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


def use_wide_search_space():
    """Revision-1 T10 (opt-in): one step below/above the published W, box and r ranges,
    lambda widened by 0.5 decade on each side. Mutates the module-level search spaces."""
    SEARCH_SPACE_SHARED.update(timewindow=["4d", "5d", "6d", "7d", "8d"],
                               box_size=[25, 30, 35, 40, 45], radius=[3, 4, 5, 6, 7])
    SEARCH_SPACE_GRANGER["lambda_a"] = lambda rng: float(10 ** rng.uniform(-3.0, -0.5))


def use_common_search_space():
    """Night 8 T1c (opt-in): common lambda_a / l1_ratio ranges (same in every case); draw order unchanged."""
    SEARCH_SPACE_GRANGER["lambda_a"] = lambda rng: float(10 ** rng.uniform(-3.0, 0.0))
    SEARCH_SPACE_GRANGER["l1_ratio"] = lambda rng: float(10 ** rng.uniform(-4.0, 0.0))


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

        member_seed = int(rng_local.integers(0, 2_000_000_000))
        np.random.seed(member_seed)  # legacy global seed; does not seed default_rng
        p["seed"] = member_seed  # seeds run_track cluster selection (revision-1)
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


# -----------------------------------------------------------------------------
# revision-1 night 6 (opt-in --parallel-members): members of every (trial, alpha) run in one
# process pool instead of one after another. Same member seeds and member body as run_ensemble
# (unchanged above), so the ensembles are identical; only elapsed_sec differs. Pool processes
# build the case per alpha_mix once (cache) from the same arguments as evaluate_trial_across_alphas_full.
# -----------------------------------------------------------------------------
_MEMBER_CASES = {}


def member_seeds(M, seed):
    rng_local = np.random.default_rng(seed)
    return [int(rng_local.integers(0, 2_000_000_000)) for _ in range(M)]


def run_member_task(tag, alpha_mix, extra_vars, base_params, m, member_seed):
    key = float(alpha_mix)
    if key not in _MEMBER_CASES:
        _MEMBER_CASES[key] = build_multivariate_case(alpha_mix=key, T=40, N_inject=8, seed=GLOBAL_SEED, **dict(extra_vars or {}))
    case = _MEMBER_CASES[key]
    p = copy.deepcopy(base_params)
    p["starting_lat"] = float(case["event"][0])
    p["starting_lon"] = float(case["event"][1])

    np.random.seed(member_seed)  # legacy global seed; does not seed default_rng
    p["seed"] = member_seed  # seeds run_track cluster selection (revision-1)
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

    return tag, {
        "member_id": m,
        "ok": ok,
        "error": err,
        "elapsed_sec": time.perf_counter() - t0,
        "centers": np.asarray(centers, dtype=float),
        "parents": np.asarray(parents, dtype=int),
        "out": out,
    }


def iter_trials_parallel_members(pending_jobs, n_workers, alpha_grid, M, seed, extra_vars, strength_mode, grouping, ctx):
    """Yield one evaluate_trial_worker result per trial once all members of all its alphas are back."""
    state = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = []
        for trial_id, trial in pending_jobs:
            params = build_params_from_trial_multivar(trial)   # as in evaluate_trial_across_alphas_full
            if strength_mode != "native":
                params["strength_mode"] = str(strength_mode)
            if grouping != "dbscan":
                params["grouping"] = str(grouping)
            if STABILITY is not None:
                params["cd_kwargs"]["stability"] = dict(STABILITY)
            state[int(trial_id)] = dict(trial=trial, params=params, members={}, t0=time.perf_counter())
            for j, alpha_mix in enumerate(alpha_grid):
                for m, s in enumerate(member_seeds(M, int(seed + 1000 * trial_id + j))):
                    futures.append(ex.submit(run_member_task, (int(trial_id), j), float(alpha_mix), extra_vars, params, m, s))
        for fut in as_completed(futures):
            (trial_id, j), member = fut.result()
            st = state[trial_id]
            st["members"][(j, member["member_id"])] = member
            if len(st["members"]) < M * len(alpha_grid):
                continue
            ensembles = {j: {"members": [st["members"][(j, m)] for m in range(M)],
                             "params": copy.deepcopy(st["params"]), "M": M} for j in range(len(alpha_grid))}
            result = evaluate_trial_worker((trial_id, st["trial"]), alpha_grid, M, seed, extra_vars, strength_mode,
                                           grouping, ensembles=ensembles)
            result["trial_row"]["elapsed_sec"] = float(time.perf_counter() - st["t0"])
            del state[trial_id]
            yield result


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
def evaluate_trial_across_alphas_full(trial_id, trial, alpha_grid=ALPHA_MIX_GRID, M=30, seed=SEARCH_SEED,
                                      extra_vars=None, strength_mode="native", grouping="dbscan", ensembles=None):
    # ensembles: night 6 --parallel-members passes {alpha index: pooled ensemble} (default None = original)
    params = build_params_from_trial_multivar(trial)
    if strength_mode != "native":
        params["strength_mode"] = str(strength_mode)  # revision-1 opt-in (T9)
    if grouping != "dbscan":
        params["grouping"] = str(grouping)  # revision-1 opt-in (E, R1 major 2)
    if STABILITY is not None:  # night-2 D4 opt-in (set from --stability-selection in main / worker)
        params["cd_kwargs"]["stability"] = dict(STABILITY)
    extra_vars = dict(extra_vars or {})

    rows_alpha = []
    rows_members = []
    bundles = []

    for j, alpha_mix in enumerate(alpha_grid):
        case = build_multivariate_case(
            alpha_mix=float(alpha_mix),
            T=40,
            N_inject=8,
            seed=GLOBAL_SEED,
            **extra_vars,
        )

        if ensembles is None:
            ensemble_out = run_ensemble(
                case,
                params,
                M=M,
                seed=int(seed + 1000 * trial_id + j),
            )
        else:
            ensemble_out = ensembles[j]

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
        if sum(extra_vars.get(k, 0) for k in ("n_noise", "n_decoy", "n_moving")) > 0 or extra_vars.get("extra_mix"):
            row_alpha.update(summarize_all_vars(case, ensemble_out, params))
        if strength_mode != "native":
            row_alpha["strength_mode"] = str(strength_mode)
        if grouping != "dbscan":
            row_alpha["grouping"] = str(grouping)
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
def evaluate_trial_worker(job, alpha_grid=ALPHA_MIX_GRID, M=30, seed=SEARCH_SEED,
                          extra_vars=None, strength_mode="native", grouping="dbscan", ensembles=None):
    trial_id, trial = job
    t0 = time.perf_counter()

    try:
        out = evaluate_trial_across_alphas_full(
            trial_id=trial_id,
            trial=trial,
            alpha_grid=alpha_grid,
            M=M,
            seed=seed,
            extra_vars=extra_vars,
            strength_mode=strength_mode,
            grouping=grouping,
            ensembles=ensembles,
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
    # revision-1 opt-in flags (defaults = original behaviour)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=None,
                        help="alpha_mix values (default: ALPHA_MIX_GRID)")
    parser.add_argument("--n-noise", type=int, default=0, help="T7: extra independent AR(1) noise variables")
    parser.add_argument("--n-decoy", type=int, default=0, help="T7: extra decoy variables driven by A or C")
    parser.add_argument("--n-moving", type=int, default=0, help="T7: extra independent moving-blob variables")
    parser.add_argument("--decoy-phi", type=float, default=0.8, help="T7: decoy memory D[t-1] coefficient")
    parser.add_argument("--decoy-beta", type=float, default=0.8, help="T7: decoy coupling to A or C at t-1")
    parser.add_argument("--decoy-sigma", type=float, default=0.1, help="T7: decoy AR(1) noise sigma")
    parser.add_argument("--decoy-driver", type=str, default="parent", choices=["parent", "child"],
                        help="night-4 C: decoys driven by A/C (parent, default = original) or by the child B (response decoy)")
    parser.add_argument("--decoy-lag", type=int, default=1, help="night-5 T3: decoys driven at lag L (default 1 = original)")
    parser.add_argument("--decoy-nonlinear", type=str, default=None, choices=["rectified"],
                        help="night-5 T3: rectified drive beta*max(X[t-L]-0.5, 0) (default none = linear, original)")
    parser.add_argument("--trial-ids", type=int, nargs="+", default=None,
                        help="night 8 S10: run only these trial ids of the drawn configs (default: all)")
    parser.add_argument("--extra-mix", type=str, nargs="+", default=None,
                        choices=["noise", "response", "moving", "delayed", "nonlinear", "typical"],
                        help="night-5 T3: extra variables of these types, in this order (see _build_extra_vars)")
    parser.add_argument("--mix-beta", type=str, nargs="+", default=None,
                        help="night-5 T3: per-type beta for --extra-mix decoys, e.g. delayed=0.2 (default: --decoy-beta)")
    parser.add_argument("--set-param", type=str, nargs="+", default=None,
                        help="night-5 T4 (opt-in): KEY=VALUE overrides applied to every drawn config after the draws")
    parser.add_argument("--force-prob-rule", type=str, default=None, choices=["linear", "softmax"],
                        help="night 5 (opt-in): set prob_rule on every drawn config after the draws (draws unchanged)")
    parser.add_argument("--strength-mode", type=str, default="native", choices=["native", "refit"],
                        help="T9: link-strength definition passed to run_track")
    parser.add_argument("--search-space", type=str, default="paper", choices=["paper", "wide", "common"],
                        help="T10: 'wide' widens W/box/r by one step and lambda by 0.5 decade each side; "
                             "night 8 T1c: 'common' draws lambda_a 10^U(-3,0), l1_ratio 10^U(-4,0), scale parameters unchanged")
    # night-2 opt-ins (C/D); defaults = original generator
    parser.add_argument("--target-driver", type=str, default="clean", choices=["clean", "observed"],
                        help="C1: B driven by A_clean/C_clean (default) or by the observed noisy A/C")
    parser.add_argument("--parent-sigma", type=float, default=None, help="C2: sigma_A = sigma_C (default 0.10)")
    parser.add_argument("--moving-placement", type=str, default="event", choices=["event", "random"],
                        help="D2: moving blobs aimed at the event (default) or random straight tracks")
    parser.add_argument("--moving-offset", type=float, default=15.0, help="D3(i): closest-approach distance")
    parser.add_argument("--moving-tc-lag", type=float, default=2.0, help="D3(ii): closest approach at t0 - lag")
    parser.add_argument("--timewindow-choices", type=str, nargs="+", default=None,
                        help="D3(iii): override each trial's W with a seeded draw from these (e.g. 10d 11d 12d)")
    parser.add_argument("--stability-selection", action="store_true",
                        help="D4: Elastic-Net stability selection, n_sub 20, frac 0.5, thr 0.6 (Meinshausen & Buhlmann 2010)")
    parser.add_argument("--grouping", type=str, default="dbscan",
                        choices=["dbscan", "strongest", "variable_sign", "sectors8", "kmeans"],
                        help="revision-1 E: winner grouping passed to run_track (default dbscan = original)")
    parser.add_argument("--timewindow-scale-to-p3", action="store_true",
                        help="night-6 T4: grow each trial's drawn W so that N_samples/N_features equals its p = 3 value "
                             "(W_p = 1 + round((W - 1) p / 3); see scale_timewindow_to_p3)")
    parser.add_argument("--t-pad", type=int, default=0,
                        help="night-6 T4: prepend this many days to the synthetic series (default 0 = original T = 40)")
    parser.add_argument("--parallel-members", action="store_true",
                        help="night 6 (opt-in): the --n-workers pool runs ensemble members of every (trial, alpha) "
                             "(same seeds, same ensembles) instead of whole trials")
    parser.add_argument("--linkless", type=str, default=None, choices=["A", "indep"],
                        help="night 7 T3b (opt-in): replace all variables by a linkless surrogate system "
                             "(A: with the lag-0 correlation matrix R of the variables; indep: R = identity)")
    parser.add_argument("--linkless-draw", type=int, default=0, help="night 7 T3b: surrogate draw (seed)")
    parser.add_argument("--linkless-log", type=str, default=None, help="night 7 T3b: directory for the realized-statistics JSON")
    return parser.parse_args()


def _parse_set_param(items):
    """Night 5: ["radius=3", "box_size=30", "timewindow=2d"] -> {"radius": 3, "box_size": 30.0, "timewindow": "2d"}."""
    out = {}
    for s in items:
        k, v = s.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def night2_case_kwargs(args):
    """Non-default night-2 generator options only (so default runs pass exactly the old kwargs)."""
    kw = {}
    if args.target_driver != "clean":
        kw["target_driver"] = args.target_driver
    if args.parent_sigma is not None:
        kw["parent_sigma"] = float(args.parent_sigma)
    if args.moving_placement != "event":
        kw["moving_placement"] = args.moving_placement
    if args.moving_offset != 15.0:
        kw["moving_offset"] = float(args.moving_offset)
    if args.moving_tc_lag != 2.0:
        kw["moving_tc_lag"] = float(args.moving_tc_lag)
    if getattr(args, "decoy_driver", "parent") != "parent":   # older callers build Namespaces without it
        kw["decoy_driver"] = args.decoy_driver
    if getattr(args, "decoy_lag", 1) != 1:   # night-5 T3 opt-ins
        kw["decoy_lag"] = int(args.decoy_lag)
    if getattr(args, "decoy_nonlinear", None):
        kw["decoy_nonlinear"] = args.decoy_nonlinear
    if getattr(args, "extra_mix", None):
        kw["extra_mix"] = list(args.extra_mix)
    if getattr(args, "mix_beta", None):
        kw["mix_beta"] = {k: float(v) for k, v in (s.split("=") for s in args.mix_beta)}
    if getattr(args, "t_pad", 0):   # night-6 T4
        kw["t_pad"] = int(args.t_pad)
    if getattr(args, "linkless", None):   # night-7 T3b
        kw["linkless"] = dict(variant=args.linkless, draw=int(args.linkless_draw),
                              **({"log_dir": args.linkless_log} if args.linkless_log else {}))
    return kw


def scale_timewindow_to_p3(jobs, p, p_ref=3):
    """Night-6 T4 (fixed-ratio curve). The EN regression of one step (castle_core.mv_CaStLe_GRANGER, lag 1) has
    N_samples = (ny - 2r)(nx - 2r)(W - 1) stencil pairs (reference cells of the box x consecutive time pairs in the
    window of W steps) and N_features = p (2r + 1)^2 (p variables x stencil), as in summarize_all_vars
    (t7_n_samples, t7_n_features). With box and r fixed per trial the ratio scales as (W - 1) / p, so
    W_p = 1 + round((W - 1) p / p_ref) keeps it at the p_ref value (exact when (W - 1) p / p_ref is an integer).
    Returns new jobs (other parameters unchanged)."""
    out = []
    for trial_id, trial in jobs:
        t = dict(trial)
        w = int(pd.to_timedelta(t["timewindow"]) / pd.to_timedelta("1d"))
        t["timewindow_drawn"] = str(trial["timewindow"])
        t["timewindow"] = f"{1 + int(np.floor((w - 1) * int(p) / float(p_ref) + 0.5))}d"
        out.append((trial_id, t))
    return out


def override_timewindow(jobs, choices, seed):
    """D3(iii): W per trial from a separate rng (seed + 7000 + trial_id); other params unchanged."""
    out = []
    for trial_id, trial in jobs:
        t = dict(trial)
        t["timewindow"] = str(np.random.default_rng(int(seed) + 7000 + int(trial_id)).choice(list(choices)))
        out.append((trial_id, t))
    return out


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

    if args.search_space == "wide":
        use_wide_search_space()
    elif args.search_space == "common":
        use_common_search_space()
    all_jobs = build_multivar_jobs(n_trials=args.n_trials, seed=args.seed)
    if args.timewindow_choices:
        all_jobs = override_timewindow(all_jobs, args.timewindow_choices, args.seed)
    if args.force_prob_rule:
        for _, trial in all_jobs:
            trial["prob_rule"] = args.force_prob_rule
    if getattr(args, "trial_ids", None):   # night 8 S10 (opt-in): run only these paper configs (same draws, same ids)
        keep_ids = {int(i) for i in args.trial_ids}
        all_jobs = [j for j in all_jobs if int(j[0]) in keep_ids]
        if not all_jobs:
            raise SystemExit("--trial-ids: none of the ids exists")
    if args.set_param:   # night-5 T4 out-of-sample check (opt-in): fixed values after the draws
        for _, trial in all_jobs:
            trial.update(_parse_set_param(args.set_param))
    p_run = 3 + args.n_noise + args.n_decoy + args.n_moving + len(args.extra_mix or [])
    if args.timewindow_scale_to_p3:   # night-6 T4 fixed-ratio curve (opt-in)
        all_jobs = scale_timewindow_to_p3(all_jobs, p_run)
        w_max = max(int(pd.to_timedelta(t["timewindow"]) / pd.to_timedelta("1d")) for _, t in all_jobs)
        if w_max + 30 - 1 > 40 + args.t_pad:   # 30 = n_steps_time, 40 = T (evaluate_trial_across_alphas_full)
            raise SystemExit(f"--timewindow-scale-to-p3: W up to {w_max}d needs --t-pad >= {w_max + 29 - 40}")
    global STABILITY
    if args.stability_selection:
        STABILITY = dict(n_sub=20, frac=0.5, thr=0.6, seed=0)
        os.environ["TST_STABILITY"] = json.dumps(STABILITY)
    n2kw = night2_case_kwargs(args)

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
        "alpha_mix_grid": [float(x) for x in (args.alpha_grid or ALPHA_MIX_GRID)],
        "extra_vars": dict(n_noise=args.n_noise, n_decoy=args.n_decoy, n_moving=args.n_moving,
                           decoy_phi=args.decoy_phi, decoy_beta=args.decoy_beta, decoy_sigma=args.decoy_sigma),
        "strength_mode": args.strength_mode,
        "search_space": args.search_space,
        **({"night2_case_kwargs": n2kw} if n2kw else {}),
        **({"timewindow_choices": list(args.timewindow_choices)} if args.timewindow_choices else {}),
        **({} if args.grouping == "dbscan" else {"grouping": args.grouping}),
        **({"stability": STABILITY} if STABILITY else {}),
        **({"timewindow_scale_to_p3": dict(p=int(p_run), timewindows={str(i): t["timewindow"] for i, t in all_jobs})}
           if args.timewindow_scale_to_p3 else {}),
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

    extra_vars_run = dict(n_noise=args.n_noise, n_decoy=args.n_decoy, n_moving=args.n_moving,
                          **({} if (args.n_decoy == 0 and not args.extra_mix) else dict(decoy_phi=args.decoy_phi, decoy_beta=args.decoy_beta,
                                                                             decoy_sigma=args.decoy_sigma)),
                          **n2kw)
    if args.parallel_members:  # night 6 opt-in: pool over members (iter_trials_parallel_members owns its pool)
        ex = contextlib.nullcontext()
        results_iter = iter_trials_parallel_members(
            pending_jobs, args.n_workers, (args.alpha_grid or ALPHA_MIX_GRID), args.ensemble_size, args.seed,
            extra_vars_run, args.strength_mode, args.grouping, ctx)
    else:
        ex = ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx)
        futures = [
            ex.submit(
                evaluate_trial_worker,
                job,
                (args.alpha_grid or ALPHA_MIX_GRID),
                args.ensemble_size,
                args.seed,
                extra_vars_run,
                args.strength_mode,
                args.grouping,
            )
            for job in pending_jobs
        ]
        results_iter = (fut.result() for fut in as_completed(futures))

    with ex:
        for result in results_iter:

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
