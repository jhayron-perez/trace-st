"""Retention (quality-control) rules used for the paper figures, as named functions.

Defaults reproduce exactly the rules implemented in the submission notebooks
(audit 2026-11-01), not the manuscript wording:

- Debby (7_Analysis_DebbyPinatubo_Fig3_v2, cell 11):
  n_members_failed < 10 & mean_traj_length > 288 (12 d of hourly steps)   -> 68 / 300
- Pinatubo (same notebook, cell 18):
  n_members_failed < 10 & mean_traj_length > 15                            -> 999 / 1000
- PNW 2021 (9_AnalysisHW_Figs45_v2, cell 7):
  success_rate >= 0.66 & median_steps >= 25                                -> 242 / 300
- T1 synthetic (4and2_Fig2, define_internal_valid):
  min over the 3 tracks of len(centers_raw) == 31                          -> 993 / 935 / 597
  (granger / dynotears / pcmci)

Each function returns a boolean pandas Series aligned with the input rows
(T1: indexed by (method, trial_id)).
"""

from __future__ import annotations

import pandas as pd


def debby_retained(jobs: pd.DataFrame, *, max_failed: int = 10, min_mean_len: float = 12 * 24) -> pd.Series:
    """Debby trials kept: fewer than `max_failed` failed members and mean length > `min_mean_len`."""
    return (jobs["n_members_failed"] < max_failed) & (jobs["mean_traj_length"] > min_mean_len)


def pinatubo_retained(jobs: pd.DataFrame, *, max_failed: int = 10, min_mean_len: float = 15) -> pd.Series:
    """Pinatubo trials kept: fewer than `max_failed` failed members and mean length > `min_mean_len`."""
    return (jobs["n_members_failed"] < max_failed) & (jobs["mean_traj_length"] > min_mean_len)


def pnw_retained(summary: pd.DataFrame, *, min_success_rate: float = 0.66, min_median_steps: float = 25) -> pd.Series:
    """PNW runs kept: success_rate >= `min_success_rate` and median_steps >= `min_median_steps`."""
    return (summary["success_rate"] >= min_success_rate) & (summary["median_steps"] >= min_median_steps)


def t1_retained(tracks: pd.DataFrame, *, full_len: int = 31) -> pd.Series:
    """T1 trials kept: every track of the trial has `full_len` centers (start + 30 steps).

    `tracks` has one row per (method, trial_id, track) with column `raw_len`.
    Returns a boolean Series indexed by (method, trial_id).
    """
    return tracks.groupby(["method", "trial_id"])["raw_len"].min() == full_len


RULES = {
    "debby": debby_retained,
    "pinatubo": pinatubo_retained,
    "pnw": pnw_retained,
    "t1": t1_retained,
}


def harmonized_retained(n_success, n_requested, median_len, n_full, *, min_member_frac: float = 2 / 3,
                        min_len_frac: float = 0.8) -> pd.Series:
    """Revision-1 harmonized rule (night 5, one rule for every case; opt-in, the case rules above are unchanged).

    Keep a config if at least `min_member_frac` of the requested members succeeded and the median
    trajectory length (in steps, start point excluded) is at least `min_len_frac` of the configured
    number of steps `n_full`. The 2/3 default is tested exactly (3 * n_success >= 2 * n_requested).
    Inputs are aligned array-likes / Series; returns a boolean Series.
    """
    ns = pd.Series(n_success, dtype=float).reset_index(drop=True)
    nr = pd.Series(n_requested, dtype=float).reset_index(drop=True)
    ml = pd.Series(median_len, dtype=float).reset_index(drop=True)
    if abs(min_member_frac - 2 / 3) < 1e-12:
        ok_m = 3 * ns >= 2 * nr
    else:
        ok_m = ns >= min_member_frac * nr - 1e-9
    ok_l = ml >= min_len_frac * float(n_full) - 1e-9
    return (ok_m & ok_l).fillna(False).astype(bool)


def harmonized_case(case: str, table: pd.DataFrame, **kw) -> pd.Series:
    """`harmonized_retained` on a case's run table.

    debby / pinatubo: results_jobs (n_members_success, n_members_requested, median_traj_length = len(parents),
    N = n_steps_time); pnw: hyperparam_search_summary (n_success, n_requested; median_steps counts the start
    point, so length = median_steps - 1; N = 30).
    """
    if case in ("debby", "pinatubo"):
        n_full = {"debby": 360, "pinatubo": 16}[case]
        if "n_steps_time" in table:
            n_full = float(table["n_steps_time"].iloc[0])
        r = harmonized_retained(table["n_members_success"], table["n_members_requested"],
                                table["median_traj_length"], n_full, **kw)
    elif case == "pnw":
        r = harmonized_retained(table["n_success"], table["n_requested"], table["median_steps"] - 1, 30, **kw)
    else:
        raise ValueError(case)
    r.index = table.index
    return r
