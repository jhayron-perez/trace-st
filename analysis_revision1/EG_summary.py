"""Night-2 E (grouping options) and G (longer N): tables and figures.

E. T1 (Elastic-Net, first 200 configs of the paper draw, deterministic), per grouping, over configs:
   - track error (mean_distance), origin error (origin_distance), wrong-variable fraction (parent2_frac);
   - completion = the config's 3 tracks all reach raw_len 31 (trace_st.retention.t1_retained).
   T2-EN (50 configs, M 30, alpha 0.2 and 0.6): |alpha error| = |estimated_alpha_density - alpha|;
   physical validity (is_valid_physical); valid_ok_frac.
G. T1 N = 40 and N = 49 on the 993 retained configs, using the centres and parents stored per step:
   - per-step displacement |c_k - c_{k-1}|;
   - spread of positions (std over configs and tracks of the distance to the true origin);
   - share of steps on V1 (true parent) vs V2 by step.
   Steps 1-30 are compared with steps 31-N. The true origin (day 0) is reached at step 39.
   The causal strength of the winning group is not stored in results_tracks; the parent mix and
   displacement are used instead [inferred proxy].

python EG_summary.py   (PBS only)
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = C.R1
GROUPINGS = ["dbscan", "strongest", "variable_sign", "sectors8", "kmeans"]
GCOL = dict(zip(GROUPINGS, ["#000000", "#E69F00", "#56B4E9", "#CC79A7", "#D55E00"]))


def load_tracks(d):
    for f in ("results_tracks_final.csv", "results_tracks_snapshot.csv"):
        if (d / f).exists():
            return pd.read_csv(d / f)
    return None


def main():
    C.set_style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = C.outdir("EG_summary")
    # ---------------- E, T1
    rows = []
    for g in GROUPINGS:
        t = load_tracks(R1 / "E_grouping" / f"T1_{g}")
        if t is None:
            continue
        t = t[t["method"] == "granger"]
        comp = retention.t1_retained(t)
        per = t.groupby("trial_id")[["mean_distance", "origin_distance", "parent2_frac"]].mean()
        rows.append(dict(grouping=g, n_cfg=int(per.shape[0]), completion=float(comp.mean()),
                         track_err_median=float(per["mean_distance"].median()),
                         origin_err_median=float(per["origin_distance"].median()),
                         wrong_var_frac_mean=float(per["parent2_frac"].mean()),
                         track_err_median_completed=float(per.loc[comp[comp].index.get_level_values(1), "mean_distance"].median())
                         if comp.any() else np.nan))
    E1 = pd.DataFrame(rows)
    # ---------------- E, T2
    rows = []
    for g in GROUPINGS:
        f = R1 / "E_grouping" / f"T2_{g}" / "results_alphas_final.csv"
        if not f.exists():
            f = f.with_name("results_alphas_snapshot.csv")
        if not f.exists():
            continue
        x = pd.read_csv(f)
        x["abs_alpha_err"] = (x["estimated_alpha_density"] - x["alpha_mix"]).abs()
        for a, gg in x.groupby("alpha_mix"):
            rows.append(dict(grouping=g, alpha=a, n_cfg=len(gg), abs_alpha_err_median=float(gg["abs_alpha_err"].median()),
                             valid_physical=float(gg["is_valid_physical"].mean()) if "is_valid_physical" in gg else np.nan,
                             ok_frac=float(gg["valid_ok_frac"].mean()) if "valid_ok_frac" in gg else np.nan,
                             mae_step_median=float(gg["mae_step"].median()) if "mae_step" in gg else np.nan))
    E2 = pd.DataFrame(rows)
    E1.to_csv(out / "E_T1_grouping.csv", index=False); E2.to_csv(out / "E_T2_grouping.csv", index=False)
    print(E1.round(3).to_string(), "\n", E2.round(3).to_string(), flush=True)
    if len(E1):
        fig, ax = plt.subplots(1, 4, figsize=(18, 5.5))
        for k, (col, lab) in enumerate([("track_err_median", "Track error (grid units, median)"),
                                        ("origin_err_median", "Origin error (grid units, median)"),
                                        ("wrong_var_frac_mean", "Wrong-variable fraction (V2)"),
                                        ("completion", "Completion (all 3 tracks reach 30 steps)")]):
            ax[k].bar(E1["grouping"], E1[col], color=[GCOL[g] for g in E1["grouping"]])
            ax[k].set_title(lab, fontsize=15); ax[k].tick_params(axis="x", rotation=40)
        fig.suptitle("T1 (Elastic-Net, 200 configs): winner-grouping options")
        fig.tight_layout(); C.savefig(fig, "E_grouping", "E_T1_grouping"); plt.close(fig)
    if len(E2):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        for k, a in enumerate(sorted(E2["alpha"].unique())):
            g = E2[E2["alpha"] == a]
            ax[k].bar(g["grouping"], g["abs_alpha_err_median"], color=[GCOL[x] for x in g["grouping"]])
            ax[k].set_title(f"T2-EN, true α = {a}: |α error| (median)"); ax[k].tick_params(axis="x", rotation=40)
        fig.tight_layout(); C.savefig(fig, "E_grouping", "E_T2_grouping"); plt.close(fig)

    # ---------------- G
    spec_rows, prof_rows = [], []
    import importlib.util
    spec = importlib.util.spec_from_file_location("t1s", C.REPO / "Scripts_Paper" / "1_SyntheticRuns_Univariate.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    _, specs, _ = m.build_synthetic_case()
    origins = {i + 1: np.asarray(s["centers"], float)[0] for i, s in enumerate(specs)}
    for N in (40, 49):
        t = load_tracks(R1 / f"G_T1_N{N}")
        if t is None:
            continue
        for r in t.itertuples():
            c = np.asarray(ast.literal_eval(r.centers_raw), float)
            p = np.asarray(ast.literal_eval(r.parents_raw), int)
            o = origins[int(r.track_idx)]
            for k in range(1, len(c)):
                prof_rows.append(dict(N=N, trial_id=r.trial_id, track=r.track_idx, step=k,
                                      displacement=float(np.hypot(*(c[k] - c[k - 1]))),
                                      dist_to_origin=float(np.hypot(*(c[k] - o))), parent=int(p[k]) if k < len(p) else -1))
    if prof_rows:
        P = pd.DataFrame(prof_rows)
        G = P.groupby(["N", "step"]).agg(disp_mean=("displacement", "mean"), disp_median=("displacement", "median"),
                                         dist_origin_median=("dist_to_origin", "median"),
                                         dist_origin_std=("dist_to_origin", "std"),
                                         frac_parent_v1=("parent", lambda x: float((x == 1).mean())),
                                         frac_parent_v2=("parent", lambda x: float((x == 2).mean())),
                                         n=("parent", "size")).reset_index()
        G.to_csv(out / "G_profiles.csv", index=False)
        P["block"] = np.where(P["step"] <= 30, "steps 1-30", np.where(P["step"] <= 39, "steps 31-39", "steps 40+"))
        B = P.groupby(["N", "block"]).agg(disp_mean=("displacement", "mean"), dist_origin_median=("dist_to_origin", "median"),
                                          dist_origin_std=("dist_to_origin", "std"),
                                          frac_v1=("parent", lambda x: float((x == 1).mean()))).reset_index()
        B.to_csv(out / "G_blocks.csv", index=False); print(B.round(3).to_string(), flush=True)
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        for N, g in G.groupby("N"):
            ls = "-" if N == 40 else "--"
            ax[0].plot(g["step"], g["disp_mean"], ls, lw=2, color="#0072B2", label=f"N = {N}")
            ax[1].plot(g["step"], g["dist_origin_median"], ls, lw=2, color="#D55E00", label=f"N = {N}")
            ax[1].fill_between(g["step"], g["dist_origin_median"] - g["dist_origin_std"],
                               g["dist_origin_median"] + g["dist_origin_std"], color="#D55E00", alpha=0.1, lw=0)
            ax[2].plot(g["step"], g["frac_parent_v1"], ls, lw=2, color="#000000", label=f"V1, N = {N}")
            ax[2].plot(g["step"], g["frac_parent_v2"], ls, lw=2, color="#999999", label=f"V2, N = {N}")
        for a_ in ax:
            a_.axvline(30, color="#999999", lw=1, ls=":"); a_.axvline(39, color="#CC79A7", lw=1.5, ls=":")
            a_.set_xlabel("Tracing step")
        ax[0].set_ylabel("Mean step displacement (grid units)"); ax[1].set_ylabel("Distance to true origin (median ± sd)")
        ax[2].set_ylabel("Fraction of steps by parent"); ax[0].legend(frameon=False); ax[2].legend(frameon=False, fontsize=12)
        ax[1].text(39.3, 0.95, "true origin", color="#CC79A7", fontsize=12, transform=ax[1].get_xaxis_transform(), va="top")
        fig.suptitle("T1, Elastic-Net, 993 configs: beyond the published 30 steps")
        fig.tight_layout(); C.savefig(fig, "G_longer_N", "G_profiles"); plt.close(fig)


if __name__ == "__main__":
    main()
