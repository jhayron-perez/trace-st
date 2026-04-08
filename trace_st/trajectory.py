"""Trajectory driver for TraCE-ST

Core loop:
1) Slice *last N steps* ending at `date_end` (inclusive).
2) Run multivariate CaStLe/PCMCI inside the chosen spatial box.
3) Collect candidate parent edges into a table.
4) Update child variable (winner variable) using the single strongest edge (your rule).
5) Update movement using directional clustering + (optional) Monte Carlo.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
from tigramite.independence_tests.parcorr import ParCorr

from . import castle_core as mcastle
from .data_io import get_data_mcastle
from .movement import (
    pick_direction_then_weighted_delta,
    snap_delta_to_stencil,
    summarize_clusters,
)


def get_mcastle_box(
    center,
    box_size,
    data_list,
    *,
    radius: int = 1,
    child_variable: Optional[int] = None,
    cd_method: str = "pcmci",
    cd_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Run M-CaStLe in a box centered at (lat,lon) for a list of variables.

    Parameters
    ----------
    cd_method : {"pcmci","dynotears"}
        Which causal discovery backend to use.
    cd_kwargs : dict
        Extra keyword args forwarded to the underlying method.
        - For pcmci: pc_alpha, graph_p_threshold, fdr_method, cd_function, ...
        - For dynotears: lambda_a, lambda_w, max_iter, dependence_threshold, ...
    """
    datacastlebox = get_data_mcastle(center, box_size, data_list)

    cd_kwargs = {} if cd_kwargs is None else dict(cd_kwargs)
    cd_method = str(cd_method).lower()

    if cd_method in ("pcmci", "pc", "mv_castle_pc"):
        # defaults (you can override via cd_kwargs)
        kwargs = dict(
            cond_ind_test=ParCorr(significance="analytic"),
            cd_function="run_pcmci",
            pc_alpha=0.005,
            graph_p_threshold=0.001,
            min_tau=1,
            y_ascending=True,
            fdr_method="bh",
            intervariable_link_assumptions=None,
            allow_center_directed_links=True,
            dependencies_wrap=False,
            verbose=0,
            radius=int(radius),
            child_variable=child_variable,
        )
        kwargs.update(cd_kwargs)

        results = mcastle.mv_CaStLe_PC(datacastlebox, **kwargs)

        learned_stencil_graph = results["graph"]
        learned_stencil_vals = results["val_matrix"]

    elif cd_method in ("dynotears", "dynt", "mv_castle_dynotears"):
        # defaults (you can override via cd_kwargs)
        kwargs = dict(
            # match mv_CaStLe_PC semantics where possible
            min_tau=1,
            y_ascending=True,
            radius=int(radius),
            allow_center_directed_links=True,
            dependencies_wrap=False,
            verbose=0,
            child_variable=child_variable,
            return_reduced_space=False,

            # DYNOTEARS knobs
            lambda_a=0.01,
            lambda_w=None,
            max_iter=300,
            dependence_threshold=0.01,
            strength_threshold=None,  # optional post-threshold on |weights|
        )
        kwargs.update(cd_kwargs)

        results = mcastle.mv_CaStLe_DYNOTEARS(datacastlebox, **kwargs)

        learned_stencil_graph = results["graph"]
        learned_stencil_vals = results["val_matrix"]

    elif cd_method in ("granger", "mv_castle_granger", "var", "lasso"):
        kwargs = dict(
            min_tau=1,
            y_ascending=True,
            radius=int(radius),
            dependencies_wrap=False,
            verbose=0,
            child_variable=child_variable,
            return_reduced_space=False,
    
            # granger knobs
            lambda_a=0.01,
            dependence_threshold=0.01,
            strength_threshold=None,
            fit_intercept=True,
            max_iter=5000,
        )
        kwargs.update(cd_kwargs)
    
        results = mcastle.mv_CaStLe_GRANGER(datacastlebox, **kwargs)
        learned_stencil_graph = results["graph"]
        learned_stencil_vals  = results["val_matrix"]

    else:
        raise ValueError(f"Unknown cd_method='{cd_method}'. Use 'pcmci' or 'dynotears'.")

    # --- split into per-(parent,child) blocks like before ---
    stencil_size = (2 * int(radius) + 1) ** 2
    n_vars = learned_stencil_graph.shape[0] // stencil_size

    results_separated = {}
    for parent in range(1, n_vars + 1):
        for child in range(1, n_vars + 1):
            p_start = (parent - 1) * stencil_size
            p_end = parent * stencil_size
            c_start = (child - 1) * stencil_size
            c_end = child * stencil_size

            results_separated[f"p{parent}_c{child}"] = (
                learned_stencil_graph[p_start:p_end, c_start:c_end, :],
                learned_stencil_vals[p_start:p_end, c_start:c_end, :],
            )

    return results_separated

def run_track(
    full_data: xr.DataArray,
    params: Dict[str, Any],
    *,
    date_end=None,
    return_debug: bool = True,
    return_cluster_summaries: bool = True,
):
    """Run TraCE-ST back-trajectory.

    Key rule (UPDATED):
    - Movement is chosen by directional clustering.
    - The *winner child variable* is the parent-variable of the chosen (sign,cluster),
      because clusters compete separately by (parent, sign).
    """

    # -------------------------
    # Unpack params
    # -------------------------
    timeres = params["timeres"]
    spaceres = float(params["spaceres"])
    box_size = float(params["box_size"])
    radius = int(params["radius"])
    starting_lat = float(params["starting_lat"])
    starting_lon = float(params["starting_lon"])
    timewindow = params["timewindow"]
    child_of_interest = int(params["child_of_interest"])
    n_steps_time = int(params["n_steps_time"])
    verbose = bool(params.get("verbose", False))

    # Cluster-level MC
    montecarlo = bool(params.get("montecarlo", False))
    beta_softmax = params.get("beta_softmax", 100.0)
    beta_softmax = 100.0 if beta_softmax is None else float(beta_softmax)

    # Directional clustering knobs
    eps_dbscan = float(params.get("eps_dbscan", 0.5))
    min_samples_dbscan = int(params.get("min_samples_dbscan", 3))

    # Movement toggles
    averaging_winners = bool(params.get("averaging_winners", True))
    snap_to_grid = bool(params.get("snap_to_grid", True))

    # Cluster selection fairness
    score_mode = str(params.get("score_mode", "sum_over_n"))
    gamma = float(params.get("gamma", 0.5))
    alpha = float(params.get("alpha", 2.0))
    prefer_sign = str(params.get("prefer_sign", "auto"))
    winner_summary = str(params.get("winner_summary", "mean"))

    # Causal discovery backend
    cd_method = str(params.get("cd_method", "pcmci")).lower()
    cd_kwargs = params.get("cd_kwargs", None)
    if cd_kwargs is not None and not isinstance(cd_kwargs, dict):
        raise ValueError("params['cd_kwargs'] must be a dict if provided")

    # -------------------------
    # Sanity on full_data
    # -------------------------
    if "var" not in full_data.dims:
        raise ValueError("full_data must have a 'var' dimension")

    var_values = full_data["var"].values
    var_names = [str(v) for v in var_values]
    n_vars = int(full_data.sizes["var"])
    var_list = list(var_values)  # fixed order

    # Precompute stencil geometry
    r = int(radius)
    s = 2 * r + 1
    stencil_size = s * s

    # Time-step and window length checks
    dt_step = pd.to_timedelta(timeres)
    ratio = pd.to_timedelta(timewindow) / dt_step
    expected_len = int(np.round(ratio))
    if not np.isclose(ratio, expected_len):
        raise ValueError(
            f"timewindow ({timewindow}) must be an integer multiple of timeres ({timeres}). Got ratio={ratio}."
        )

    # -------------------------
    # Init date_end
    # -------------------------
    if date_end is None:
        date_end = params.get("date_end", None)
    if date_end is None:
        raise ValueError("date_end must be provided (either as argument or params['date_end']).")

    date_end = pd.to_datetime(date_end)

    # -------------------------
    # Init trajectory state
    # -------------------------
    current_child = int(child_of_interest)  # 1-based
    current_center = [float(starting_lat), float(starting_lon) % 360.0]

    all_centers: List[List[float]] = [current_center]
    all_parents: List[int] = [current_child]
    all_winners: List[pd.Series] = []

    all_debug = [] if return_debug else None
    all_cluster_summaries = [] if return_cluster_summaries else None

    def _step_back_in_time(t: pd.Timestamp) -> pd.Timestamp:
        return t - dt_step.to_pytimedelta()

    # -------------------------
    # Main loop
    # -------------------------
    for ti in range(n_steps_time):
        if verbose:
            print("================================================")
            print(
                f"Step {ti+1}/{n_steps_time}\n"
                f"End date: {date_end} time window: {timewindow}\n"
                f"child of interest: {current_child} current center: {current_center}"
            )
            print("================================================", flush=True)

        # last N steps ending at date_end (inclusive)
        date_start = date_end - pd.to_timedelta(timewindow) + dt_step
        full_data_win = full_data.sel(time=slice(date_start, date_end))
        # print(full_data_win)
        if verbose:
            print(
                f"[full_data_win] sizes={dict(full_data_win.sizes)} | "
                f"dates={full_data_win.time.values[0]}-{full_data_win.time.values[-1]} | "
                f"min={float(full_data_win.min()):.3g} | "
                f"max={float(full_data_win.max()):.3g} | "
                f"nan_frac={float(np.isnan(full_data_win).mean()):.3g}"
            )

        if full_data_win.sizes.get("time", 0) < expected_len:
            if verbose:
                print(
                    f"[STOP] Not enough time points for causal discovery.\n"
                    f"Requested window: {date_start} -> {date_end}\n"
                    f"Have {full_data_win.sizes.get('time', 0)} points, expected {expected_len}"
                )
            break

        try:
            data_list = [full_data_win.sel(var=v).drop_vars("var") for v in var_list]
        except:
            data_list = [full_data_win.sel(var=v) for v in var_list]

        # --- run causal discovery in box
        try:
            output_mcastle = get_mcastle_box(
                current_center,
                box_size,
                data_list,
                radius=r,
                child_variable=current_child,
                cd_method=cd_method,
                cd_kwargs=cd_kwargs,
            )
        except ValueError as e:
            if "No valid reference point" in str(e):
                if verbose:
                    print(f"[STOP] CD backend: {e}. Window {date_start} .. {date_end}")
                break
            raise

        # --- build candidate list: all parents p -> current_child
        rows = []
        for p in range(1, len(data_list) + 1):
            g, vals = output_mcastle[f"p{p}_c{current_child}"]
            ii, jj, kk = np.where(g == "-->")
            if len(ii) == 0:
                continue
            vars_str = f"p{p}_c{current_child}"
            for a, b, c in zip(ii, jj, kk):
                rows.append((vars_str, float(vals[a, b, c]), (int(a), int(b), int(c))))

        if len(rows) == 0:
            if verbose:
                print("No parents found")
            break

        list_candidates = pd.DataFrame(rows, columns=["vars", "causal_value", "position"])
        list_candidates["causal_value_abs"] = np.abs(list_candidates["causal_value"].to_numpy(dtype=float))

        # ---- TIME UPDATE (we always move one step back each iteration)
        date_end = _step_back_in_time(date_end)

        # ---- MOVEMENT + CHILD UPDATE
        current_lat, current_lon = current_center

        if not averaging_winners:
            # fallback mode: use strongest edge for both movement + child
            winner = list_candidates.sort_values("causal_value_abs", ascending=False).iloc[0]
            pos_winner = int(winner.position[0]) % stencil_size  # localize just in case

            di = (pos_winner // s) - r
            dj = (pos_winner % s) - r
            delta_lat = -spaceres * di
            delta_lon = spaceres * dj

            current_child = int(str(winner.vars).split("_")[0].replace("p", ""))

            df_debug = None
            chosen_sign, chosen_cluster = None, None
            df_cluster_summary = None

        else:
            # 1) choose movement by clustering (already separated by parent+sign in movement.py)
            delta_lat, delta_lon, chosen_sign, chosen_cluster, df_debug = pick_direction_then_weighted_delta(
                list_candidates,
                radius=r,
                spaceres=spaceres,
                eps=eps_dbscan,
                min_samples=min_samples_dbscan,
                montecarlo=montecarlo,
                beta_softmax=beta_softmax,
                prefer_sign=prefer_sign,
                alpha=alpha,
                score_mode=score_mode,
                gamma=gamma,
                winner_summary=winner_summary
            )

            # 2) set child = parent of the chosen (sign, cluster)
            if df_debug is None or len(df_debug) == 0 or chosen_cluster is None:
                # hard fallback
                winner = list_candidates.sort_values("causal_value_abs", ascending=False).iloc[0]
                current_child = int(str(winner.vars).split("_")[0].replace("p", ""))
            else:
                cluster_col = "cluster_global" if "cluster_global" in df_debug.columns else "cluster"
                df_ch = df_debug.copy()

                # enforce chosen sign filter
                if chosen_sign == "pos":
                    df_ch = df_ch[df_ch["sign"] == 1.0]
                elif chosen_sign == "neg":
                    df_ch = df_ch[df_ch["sign"] == -1.0]

                df_ch = df_ch[df_ch[cluster_col] == chosen_cluster].copy()

                if len(df_ch) == 0:
                    winner = list_candidates.sort_values("causal_value_abs", ascending=False).iloc[0]
                    current_child = int(str(winner.vars).split("_")[0].replace("p", ""))
                else:
                    df_ch["parent"] = df_ch["vars"].apply(lambda s: int(str(s).split("_")[0].replace("p", "")))
                    parents = df_ch["parent"].unique()
                    if len(parents) != 1:
                        raise ValueError(
                            f"[BUG] chosen cluster is not unique in parent. "
                            f"parents={parents} chosen_sign={chosen_sign} chosen_cluster={chosen_cluster}"
                        )
                    current_child = int(parents[0])

                    # winner record: strongest edge within chosen cluster
                    winner = df_ch.sort_values("causal_value_abs", ascending=False).iloc[0]

            df_cluster_summary = summarize_clusters(df_debug) if return_cluster_summaries else None

            if snap_to_grid:
                delta_lat, delta_lon = snap_delta_to_stencil(delta_lat, delta_lon, spaceres=spaceres, radius=r)

            if verbose:
                child_name = var_names[current_child - 1] if (1 <= current_child <= n_vars) else f"p{current_child}"
                print(
                    f"[movement] chosen_parent={child_name} sign={chosen_sign} cluster={chosen_cluster} "
                    f"delta_lat={delta_lat:.3f} delta_lon={delta_lon:.3f}"
                )

        # ---- update center
        new_lat = float(current_lat) + float(delta_lat)
        new_lon = (float(current_lon) + float(delta_lon)) % 360.0
        current_center = [new_lat, new_lon]

        if verbose:
            print("Winner deltas:", delta_lat, delta_lon)

        all_centers.append(current_center)
        all_parents.append(current_child)
        all_winners.append(winner)

        if return_debug:
            all_debug.append(df_debug)
        if return_cluster_summaries:
            all_cluster_summaries.append(df_cluster_summary)

        # ---- stop if out of latitude domain
        da0 = data_list[0]
        if "lat" in da0.coords:
            lat_coord = da0.lat
        elif "latitude" in da0.coords:
            lat_coord = da0.latitude
        else:
            raise ValueError(f"run_track: cannot find latitude coord. Coords: {list(da0.coords)}")

        latmax = float(lat_coord.max())
        latmin = float(lat_coord.min())
        if (new_lat + box_size / 2 > latmax) or (new_lat - box_size / 2 < latmin):
            if verbose:
                print("Out of domain")
            break

    out = {
        "centers": all_centers,
        "parents": all_parents,
        "winners": all_winners,
        "final_date_end": date_end,
    }
    if return_debug:
        out["debug"] = all_debug
    if return_cluster_summaries:
        out["cluster_summaries"] = all_cluster_summaries
    return out

# Backwards-compatible alias (so older notebooks don't break)
run_backtrajectory = run_track
