"""Shared helpers for the revision-1 analysis tasks H, I, J (analysis only; reads run outputs).

Paths, retention (trace_st.retention, unchanged), plotting style (colour-blind safe: no green/yellow),
bootstrap / Wilcoxon-Holm helpers, and per-window integrated-density recomputation from bundles
that reproduces the run scripts' box-footprint density (5_DebbyRuns.py / 6_PinatuboRuns.py
_accumulate_density_box; 8_HW_PNW21_Runs_v2.py density_by_variable_for_time_window_boxes).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from trace_st import retention  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
RUNS = dict(
    t1=R1 / "T1r_univariate", t2=R1 / "T2r_multivariate_en", t2pc=R1 / "T2r_multivariate_pcmci",
    debby=R1 / "T4r_debby_vorebuilt", debby_orig=R1 / "T4r_debby", pinatubo=R1 / "T5r_pinatubo",
    pnw=R1 / "T6r_pnw", t2_wide=R1 / "T10_screening" / "T2_wide", pnw_wide=R1 / "T10_screening" / "pnw_wide",
)
OUT = R1 / "analysis_v2"
FIG = R1 / "figures_v2"

# ---------------------------------------------------------------- variables / labels / colours
# Variable order = parent id order in the run outputs (parent k -> VARS[case][k-1]);
# Debby: 5_DebbyRuns.py density_range::Tb = parent 1, vo = 2, precip = 3.
VARS = dict(
    debby=["Tb", "vo", "precip"],
    pinatubo=["TMSO201", "BURDENSO401", "TMH2SO401", "AEROD_v"],
    pnw=["Z500", "Z10", "MTNLWRF", "TCWV", "MSLHF", "MSSHF"],
)
VARS["debby_orig"] = VARS["debby"]
PLOT_ORDER = dict(debby=["precip", "Tb", "vo"], debby_orig=["precip", "Tb", "vo"],
                  pinatubo=VARS["pinatubo"], pnw=VARS["pnw"])  # paper figure order
LABELS = {"precip": "Precip.", "Tb": "BT", "vo": "Vorticity",
          "TMSO201": "SO$_2$", "BURDENSO401": "SO$_4$", "TMH2SO401": "H$_2$SO$_4$", "AEROD_v": "AOD",
          "Z500": "Z500", "Z10": "Z10", "MTNLWRF": "OLR", "TCWV": "TCWV", "MSLHF": "SLHF", "MSSHF": "SSHF"}
PLAIN = {"precip": "Precip", "Tb": "BT", "vo": "Vorticity", "TMSO201": "SO2", "BURDENSO401": "SO4",
         "TMH2SO401": "H2SO4", "AEROD_v": "AOD", "MTNLWRF": "OLR", "MSLHF": "SLHF", "MSSHF": "SSHF"}
# Okabe-Ito without green/yellow
OI = ["#000000", "#E69F00", "#56B4E9", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
VAR_COLORS = {"precip": "#E69F00", "Tb": "#CC79A7", "vo": "#0072B2",
              "TMSO201": "#D55E00", "BURDENSO401": "#0072B2", "TMH2SO401": "#CC79A7", "AEROD_v": "#999999",
              "Z500": "#000000", "Z10": "#999999", "MTNLWRF": "#CC79A7", "TCWV": "#0072B2",
              "MSLHF": "#E69F00", "MSSHF": "#D55E00"}
ENGINE_COLORS = {"granger": "#0072B2", "pcmci": "#CC79A7", "dynotears": "#E69F00"}
ENGINE_LABELS = {"granger": "Elastic-Net", "pcmci": "PCMCI", "dynotears": "DYNOTEARS"}
CASE_TITLES = {"debby": "Debby (rebuilt vorticity, primary)", "debby_orig": "Debby (original vorticity)",
               "pinatubo": "Pinatubo", "pnw": "PNW 2021"}


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17, "xtick.labelsize": 15,
        "ytick.labelsize": 15, "legend.fontsize": 14, "figure.titlesize": 22,
        "axes.prop_cycle": mpl.cycler(color=OI), "savefig.dpi": 150, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6, "pdf.fonttype": 42,
    })


def savefig(fig, task, name):
    d = FIG / task
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    return d / f"{name}.png"


def outdir(task):
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- loaders
def load_jobs(case):
    return pd.read_csv(RUNS[case] / "results_jobs_final.csv")


def retained_jobs(case):
    j = load_jobs(case)
    rule = retention.debby_retained if case.startswith("debby") else retention.pinatubo_retained
    return j[rule(j)].copy()


def load_pnw_summary(case="pnw"):
    return pd.read_csv(RUNS[case] / "hyperparam_search_summary.csv")


# ---------------------------------------------------------------- stats helpers
def bootstrap_mean_ci(X, n_boot=10000, seed=0, q=(2.5, 97.5)):
    """X: (n_configs, n_vars). Returns mean, lo, hi (per column) and the bootstrap index matrix
    (same resamples for all columns so paired differences can be formed)."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, X.shape[0], size=(n_boot, X.shape[0]))
    boots = np.stack([X[:, j][idx].mean(axis=1) for j in range(X.shape[1])], axis=1)  # (n_boot, n_vars)
    lo, hi = np.percentile(boots, q, axis=0)
    return X.mean(axis=0), lo, hi, boots


def holm(pvals):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def wilcoxon_pairs(S, names):
    """Paired two-sided Wilcoxon signed-rank for every pair of columns of S (configs x vars)."""
    from itertools import combinations
    from scipy.stats import wilcoxon
    rows = []
    for a, b in combinations(range(len(names)), 2):
        x, y = S[:, a], S[:, b]
        d = x - y
        if np.allclose(d, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        rows.append(dict(var_a=names[a], var_b=names[b], mean_a=x.mean(), mean_b=y.mean(),
                         mean_diff=d.mean(), median_diff=np.median(d), frac_a_gt_b=np.mean(d > 0),
                         frac_b_gt_a=np.mean(d < 0), W=stat, p_raw=p, n=len(d)))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_raw"].values)
    return df


# ---------------------------------------------------------------- density recomputation
def _box_cell_counts(centers, xs, ys, box_size):
    """Number of grid cells covered by each centre's box (same inclusive masks as the run scripts,
    no longitude wrap: Debby/Pinatubo)."""
    c = np.asarray(centers, float)
    half = float(box_size) / 2.0
    nx = ((xs[None, :] >= (c[:, 1:2] - half)) & (xs[None, :] <= (c[:, 1:2] + half))).sum(1)
    ny = ((ys[None, :] >= (c[:, 0:1] - half)) & (ys[None, :] <= (c[:, 0:1] + half))).sum(1)
    return (nx * ny).astype(float)


def _box_cell_counts_wrap(centers, lons, lats, box_size):
    """PNW version (longitude wrap, 8_HW_PNW21_Runs_v2.density_by_variable_for_time_window_boxes)."""
    c = np.asarray(centers, float)
    half = 0.5 * float(box_size)
    lat = c[:, 0:1]
    lon = c[:, 1:2] % 360.0
    ny = ((lats[None, :] >= lat - half) & (lats[None, :] <= lat + half)).sum(1)
    lo = (lon - half) % 360.0
    hi = (lon + half) % 360.0
    L = lons[None, :]
    inside = np.where(lo <= hi, (L >= lo) & (L <= hi), (L >= lo) | (L <= hi))
    nx = inside.sum(1)
    return (nx * ny).astype(float)


def step_density_bundle(npz_path, box_size, n_vars):
    """Per-step, per-parent integrated density for one Debby/Pinatubo trial bundle.
    Returns array D (max_len, n_vars): D[s, p-1] = sum over ok members of box-cell count at step s
    where the step's parent is p."""
    b = np.load(npz_path, allow_pickle=True)
    xs = np.asarray(b["xs"], float)
    ys = np.asarray(b["ys"], float)
    C, P, L, ok = b["centers"], b["parents"], b["lengths"], b["ok"]
    D = np.zeros((C.shape[1], n_vars))
    for i in range(len(L)):
        n = int(L[i])
        if int(ok[i]) != 1 or n <= 0:
            continue
        cnt = _box_cell_counts(C[i, :n], xs, ys, box_size)
        par = np.asarray(P[i, :n], int)
        for p in range(1, n_vars + 1):
            m = par == p
            if m.any():
                D[np.nonzero(m)[0], p - 1] += cnt[m]
    return D


PNW_LATS = np.arange(-30.0, 90.0 + 2, 2)  # 8_HW_PNW21_Runs_v2.regrid_to_2p5 (RES_PNW=2)
PNW_LONS = np.arange(0.0, 360.0, 2)


def step_density_pnw(run_dir, box_size, n_vars=6, n_steps=31):
    res = pickle.load(open(Path(run_dir) / "ensemble_results.pkl", "rb"))
    D = np.zeros((n_steps, n_vars))
    for r in res:
        if not r.get("success", True):
            continue
        c = np.asarray(r["centers"], float)
        p = np.asarray(r["parents"], int)
        n = min(len(c), len(p))
        if n == 0:
            continue
        cnt = _box_cell_counts_wrap(c[:n], PNW_LONS, PNW_LATS, box_size)
        for k in range(1, n_vars + 1):
            m = p[:n] == k
            if m.any():
                D[np.nonzero(m)[0], k - 1] += cnt[m]
    return D


def window_sum(D, a, b):
    """Sum rows [a, b) of a per-step density array."""
    return D[a:min(b, D.shape[0])].sum(axis=0)


def to_markdown(df, floatfmt="{:.4g}"):
    """Minimal markdown table (tabulate is not installed in castle)."""
    def f(x):
        if isinstance(x, (float, np.floating)):
            return "" if not np.isfinite(x) else floatfmt.format(x)
        return str(x)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"
