"""Movement and direction-selection utilities for TraCE-ST.

This module takes the list of causal edge candidates produced at each step
and turns them into a *movement* (delta_lat, delta_lon) via:

1) Choosing a sign (pos/neg/auto).
2) Clustering candidate directions using DBSCAN in unit (u,v) space.
3) Choosing a winning cluster (deterministically or Monte Carlo softmax).
4) Computing a weighted average delta within the chosen cluster.

Design principle (per your original notebook):
- Directional clustering is done *across all variables* (parents), so the
  competition is global after the sign is chosen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


# -------------------------
# Geometry helpers
# -------------------------
def _add_delta_cols(df: pd.DataFrame, radius: int, spaceres: float) -> pd.DataFrame:
    df = df.copy()

    r = int(radius)
    s = 2 * r + 1
    stencil_size = s * s

    # Original (possibly global) index
    df["pos_raw"] = df["position"].apply(lambda t: int(t[0]))

    # --- FIX: fold global node index into local stencil index ---
    df["pos_flat"] = df["pos_raw"] % stencil_size

    # ----------------------------
    # SAFETY CHECK (put it HERE)
    # ----------------------------
    bad = (df["pos_flat"] < 0) | (df["pos_flat"] >= stencil_size)
    if bad.any():
        raise ValueError(
            f"pos_flat outside stencil bounds 0..{stencil_size-1}. "
            f"Example bad values: {df.loc[bad, 'pos_flat'].unique()[:5]}"
        )

    # Optional: track which variable block it came from (for debugging)
    df["parent_block"] = df["pos_raw"] // stencil_size

    # Convert flat index → (row, col)
    df["row"] = df["pos_flat"] // s
    df["col"] = df["pos_flat"] % s

    df["di"] = df["row"] - r
    df["dj"] = df["col"] - r

    df["delta_lat"] = -float(spaceres) * df["di"]
    df["delta_lon"] =  float(spaceres) * df["dj"]

    df["dist"] = np.sqrt(df["delta_lat"]**2 + df["delta_lon"]**2)
    df["u"] = np.where(df["dist"] > 0, df["delta_lon"] / df["dist"], np.nan)
    df["v"] = np.where(df["dist"] > 0, df["delta_lat"] / df["dist"], np.nan)

    # df["u"] = np.where(df["dist"] > 0, df["delta_lon"], np.nan)
    # df["v"] = np.where(df["dist"] > 0, df["delta_lat"], np.nan)

    return df


def _dbscan_on_uv(df_sub: pd.DataFrame, eps: float, min_samples: int) -> np.ndarray:
    X = df_sub[["u", "v"]].to_numpy(dtype=float)
    return DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(X)


def _split_noise_into_singletons(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """Turn DBSCAN noise points (-1) into singleton clusters so they can compete."""
    df = df.copy()
    noise_mask = df[cluster_col] == -1
    if noise_mask.sum() == 0:
        return df

    existing = df.loc[df[cluster_col] >= 0, cluster_col]
    next_id = int(existing.max() + 1) if len(existing) else 0

    idx_noise = df.index[noise_mask]
    df.loc[idx_noise, cluster_col] = np.arange(next_id, next_id + len(idx_noise), dtype=int)
    return df

def _compute_strength(strength_sum: pd.Series, n_members: pd.Series, *, score_mode: str, gamma: float) -> pd.Series:
    if score_mode == "sum":
        return strength_sum
    elif score_mode == "mean":
        return strength_sum / n_members
    elif score_mode == "sum_over_n":
        return strength_sum / (n_members ** float(gamma))
    else:
        raise ValueError("score_mode must be 'sum', 'mean', or 'sum_over_n'")


def _pick_cluster(
    strength: pd.Series,
    *,
    montecarlo: bool,
    prob_rule: str = "linear",   # "softmax" or "linear"
    beta_softmax: float = 100.0,
    rng: np.random.Generator | None = None,
):
    """
    Pick a cluster from a Series indexed by cluster_id with values=strength/score.

    Returns:
      chosen_cluster_id (int or None),
      probs_series (pd.Series or None)
    """
    if strength is None or len(strength) == 0:
        return None, None

    clusters = strength.index.to_numpy(dtype=int)
    scores   = strength.to_numpy(dtype=float)

    if not np.all(np.isfinite(scores)):
        # drop non-finite
        ok = np.isfinite(scores)
        clusters = clusters[ok]
        scores   = scores[ok]
        if len(scores) == 0:
            return None, None

    if not montecarlo:
        return int(clusters[np.argmax(scores)]), None

    rng = np.random.default_rng() if rng is None else rng

    if prob_rule == "softmax":
        # stable softmax
        centered = scores - np.nanmax(scores)
        w = np.exp(float(beta_softmax) * centered)
        probs = w / w.sum()

    elif prob_rule == "linear":
        # proportional to score (clip negatives to 0)
        w = np.maximum(scores, 0.0)
        if w.sum() == 0:
            # fallback: uniform if all zero/negative
            probs = np.ones_like(w) / len(w)
        else:
            probs = w / w.sum()

    else:
        raise ValueError("prob_rule must be 'softmax' or 'linear'")

    probs_series = pd.Series(probs, index=clusters, name="prob_cluster")
    chosen = int(rng.choice(clusters, p=probs))
    return chosen, probs_series


def _parse_parent_from_vars(s: str) -> int:
    # "p2_c1" -> 2
    return int(str(s).split("_")[0].replace("p", ""))

def _namespace_clusters(df: pd.DataFrame, cluster_col="cluster") -> pd.DataFrame:
    """
    Make cluster ids unique across (parent_p, sign) groups.
    Only touches non-noise clusters (>=0). Noise stays -1 for now.
    """
    df = df.copy()
    df["parent_p"] = df["vars"].apply(_parse_parent_from_vars).astype(int)
    df["sign_label"] = df["sign"].map({1.0: "pos", -1.0: "neg"})

    mask = df[cluster_col] >= 0
    if mask.sum() == 0:
        return df

    # Build a unique integer id per (parent, sign, local_cluster)
    # Use factorize to keep them compact 0..K-1
    keys = list(zip(df.loc[mask, "parent_p"], df.loc[mask, "sign_label"], df.loc[mask, cluster_col].astype(int)))
    new_ids, _ = pd.factorize(keys)
    df.loc[mask, cluster_col] = new_ids.astype(int)

    return df

# -------------------------
# Public API
# -------------------------
def pick_direction_then_weighted_delta(
    list_candidates: pd.DataFrame,
    *,
    radius: int,
    spaceres: float,
    eps: float = 0.5,
    min_samples: int = 3,
    montecarlo: bool = False,
    beta_softmax: float = 100.0,
    prefer_sign: str = "both",          # "auto" | "pos" | "neg" | "both"
    alpha: float = 2.0,
    score_mode: str = "sum_over_n",
    gamma: float = 0.5,
    prob_rule: str = "linear",          # "softmax" or "linear"
    rng: np.random.Generator | None = None,
    winner_summary: str = "mean",       # <-- NEW: "mean" or "median"
):
    """
    Pick a direction cluster among candidate stencil links, then return the
    summary (delta_lat, delta_lon) of members in the winning cluster.

    winner_summary:
      - "mean": weighted mean using |causal_value|^alpha
      - "median": unweighted median across members (as in your patch)
    """
    df = list_candidates.copy()
    if len(df) == 0:
        return 0.0, 0.0, None, None, df

    # Ensure abs magnitude exists
    if "causal_value_abs" not in df.columns:
        df["causal_value_abs"] = np.abs(df["causal_value"].to_numpy(dtype=float))

    # Add delta/u/v/dist columns
    df = _add_delta_cols(df, radius=int(radius), spaceres=float(spaceres))

    # Compute sign (+1/-1) and drop zeros (treat as unusable)
    df["sign"] = np.sign(df["causal_value"]).astype(float)
    df.loc[df["sign"] == 0, "sign"] = np.nan
    if df["sign"].notna().sum() == 0:
        df["cluster"] = -1
        df["cluster_global"] = -1
        return 0.0, 0.0, None, None, df

    # Parse parent variable id (expects strings like "p3_..." etc.)
    df["parent_p"] = df["vars"].apply(lambda s: int(str(s).split("_")[0].replace("p", "")))

    # Decide sign(s) to consider
    if prefer_sign == "pos":
        signs_to_consider = [1.0]
    elif prefer_sign == "neg":
        signs_to_consider = [-1.0]
    elif prefer_sign == "both":
        signs_to_consider = [1.0, -1.0]
    elif prefer_sign == "auto":
        pos_strength = df.loc[df["sign"] == 1.0, "causal_value_abs"].sum()
        neg_strength = df.loc[df["sign"] == -1.0, "causal_value_abs"].sum()
        signs_to_consider = [1.0] if pos_strength >= neg_strength else [-1.0]
    else:
        raise ValueError("prefer_sign must be 'auto', 'pos', 'neg', or 'both'")

    # Filter to candidates that are allowed by sign choice
    df_use = df[df["sign"].isin(signs_to_consider)].copy()
    if len(df_use) == 0:
        df_use["cluster"] = -1
        df_use["cluster_global"] = -1
        return 0.0, 0.0, None, None, df_use

    # Init cluster cols
    df_use["cluster"] = -1
    df_use["cluster_global"] = -1

    # For reporting chosen_sign:
    if prefer_sign in ("pos", "neg", "auto"):
        chosen_sign = "pos" if signs_to_consider[0] > 0 else "neg"
    else:
        chosen_sign = None

    # --- Cluster and score ---
    global_id = 0
    strength_rows = []

    if prefer_sign == "both":
        group_cols = ["parent_p", "sign"]
    else:
        group_cols = "parent_p"   # <-- string, not list
    
    for key, dfp in df_use.groupby(group_cols, sort=True):
        idx = dfp.index

        mask_dir = (
            (df_use.loc[idx, "dist"] > 0)
            & np.isfinite(df_use.loc[idx, "u"])
            & np.isfinite(df_use.loc[idx, "v"])
        )

        if mask_dir.sum() > 0:
            labels = _dbscan_on_uv(
                df_use.loc[idx][mask_dir],
                eps=float(eps),
                min_samples=int(min_samples),
            )
            df_use.loc[df_use.loc[idx][mask_dir].index, "cluster"] = labels.astype(int)

        tmp = _split_noise_into_singletons(df_use.loc[idx].copy(), cluster_col="cluster")
        df_use.loc[idx, "cluster"] = tmp["cluster"].astype(int).values

        for cl in sorted(df_use.loc[idx, "cluster"].unique().tolist()):
            members = df_use.loc[idx][df_use.loc[idx, "cluster"] == cl]
            if len(members) == 0:
                continue

            df_use.loc[members.index, "cluster_global"] = global_id

            s = members["causal_value_abs"].sum()
            n = pd.Series([len(members)], index=[global_id], dtype=float)
            ss = pd.Series([s], index=[global_id], dtype=float)

            strength = _compute_strength(ss, n, score_mode=str(score_mode), gamma=float(gamma))
            strength_rows.append(strength)

            global_id += 1

    if len(strength_rows) == 0:
        return 0.0, 0.0, chosen_sign, None, df_use

    strength = pd.concat(strength_rows).sort_index()

    chosen_cluster, probs_series = _pick_cluster(
        strength,
        montecarlo=bool(montecarlo),
        prob_rule=str(prob_rule),
        beta_softmax=float(beta_softmax),
        rng=rng,
    )
    if chosen_cluster is None:
        return 0.0, 0.0, chosen_sign, None, df_use

    # Attach cluster probabilities (optional)
    df_use["prob_cluster"] = np.nan
    if probs_series is not None:
        df_use["prob_cluster"] = df_use["cluster_global"].map(probs_series).astype(float)

    # Extract winning cluster members
    use = df_use[df_use["cluster_global"] == chosen_cluster].copy()
    if len(use) == 0:
        return 0.0, 0.0, chosen_sign, chosen_cluster, df_use

    # Determine winning sign (only needed when both signs compete)
    if chosen_sign is None:
        ssum = use["causal_value"].sum()
        chosen_sign = "pos" if ssum >= 0 else "neg"

    # ---- NEW: choose summary method ----
    winner_summary = str(winner_summary).lower().strip()
    if winner_summary not in ("mean", "median"):
        raise ValueError("winner_summary must be 'mean' or 'median'")

    if winner_summary == "median":
        delta_lat = float(np.median(use["delta_lat"].to_numpy(dtype=float)))
        delta_lon = float(np.median(use["delta_lon"].to_numpy(dtype=float)))
        return delta_lat, delta_lon, chosen_sign, chosen_cluster, df_use

    # "mean" = weighted mean using |causal_value|^alpha
    w = (use["causal_value_abs"] ** float(alpha)).to_numpy(dtype=float)
    if np.all(~np.isfinite(w)) or w.sum() == 0:
        return 0.0, 0.0, chosen_sign, chosen_cluster, df_use

    w = w / w.sum()
    delta_lat = float(np.sum(use["delta_lat"].to_numpy(dtype=float) * w))
    delta_lon = float(np.sum(use["delta_lon"].to_numpy(dtype=float) * w))

    return delta_lat, delta_lon, chosen_sign, chosen_cluster, df_use

def snap_delta_to_stencil(delta_lat: float, delta_lon: float, *, spaceres: float, radius: int):
    """Snap a floating delta to the nearest allowed stencil move."""
    if float(spaceres) <= 0:
        return float(delta_lat), float(delta_lon)
    di = int(np.round(-float(delta_lat) / float(spaceres)))
    dj = int(np.round( float(delta_lon) / float(spaceres)))
    di = int(np.clip(di, -int(radius), int(radius)))
    dj = int(np.clip(dj, -int(radius), int(radius)))
    return (-float(spaceres) * di, float(spaceres) * dj)


def summarize_clusters(df_debug: pd.DataFrame) -> pd.DataFrame:
    if df_debug is None or len(df_debug) == 0:
        return pd.DataFrame()

    df = df_debug.copy()

    df["parent_p"] = df["vars"].apply(lambda s: int(str(s).split("_")[0].replace("p", "")))
    df["child_c"]  = df["vars"].apply(lambda s: int(str(s).split("_")[1].replace("c", "")))
    df["sign_label"] = df["sign"].map({1.0: "pos", -1.0: "neg"})

    cluster_col = "cluster_global" if "cluster_global" in df.columns else "cluster"
    group_cols = ["parent_p", "child_c", "sign_label", cluster_col]

    agg = {
        "n_members":       ("causal_value_abs", "size"),
        "sum_abs_causal":  ("causal_value_abs", "sum"),
        "mean_abs_causal": ("causal_value_abs", "mean"),
        "mean_delta_lat":  ("delta_lat", "mean"),
        "mean_delta_lon":  ("delta_lon", "mean"),
        "mean_dist":       ("dist", "mean"),
    }
    if "prob_cluster" in df.columns:
        agg["prob_cluster"] = ("prob_cluster", "max")

    out = (
        df.groupby(group_cols)
          .agg(**agg)
          .reset_index()
          .sort_values(["sign_label", "sum_abs_causal"], ascending=[True, False])
    )
    return out