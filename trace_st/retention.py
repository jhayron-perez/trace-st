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
