"""Night-2 D1: where and when trajectories enter the moving-blob variables (existing T7 runs).

For each run (50 configs x M members, alpha 0.6):
- Lag profile: the share of member-steps whose parent is a moving blob, per tracing step
  (step s <-> time index T-1-s, 1 day per step [inferred from the run_track backward stepping]).
- Entry: for each member, the first step whose parent is a moving blob. Record:
  - the entry step;
  - the distance from the trajectory centre to that blob's centre at that time;
  - the distance from the centre to the event;
  - the distance to the nearest true-parent (A or C) centre.
Blob centres are recomputed with the generator's formula (3_SyntheticRuns_Multivariate._build_extra_vars,
event placement, default offset 15 and closest approach at t0 - 2).

python D1_moving_postproc.py   (PBS only)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
sys.path.insert(0, str(C.REPO))
from tests.test_fraction_full_length import load_script  # noqa: E402

R1 = C.R1
RUNS = {  # name: (dir, n_noise, n_decoy, n_moving)
    "no decoys, p=8": (R1 / "T7_decoy_sweep" / "nodecoy_p08", 3, 0, 2),
    "no decoys, p=16": (R1 / "T7_decoy_sweep" / "nodecoy_p16", 7, 0, 6),
    "T7 p=8 (with decoys)": (R1 / "T7_distractors" / "p08", 2, 2, 1),
    "T7 p=16 (with decoys)": (R1 / "T7_distractors" / "p16", 5, 4, 4),
}
T, N_INJECT = 40, 8
t0 = T - N_INJECT
m = load_script("3_SyntheticRuns_Multivariate.py")
EVENT = np.array([100.0, 100.0])


def blob_centres(n_moving):
    out = []
    for k in range(n_moving):
        th = np.deg2rad(45.0 + 360.0 * k / max(n_moving, 1))
        d = np.array([np.sin(th), np.cos(th)])
        perp = np.array([-d[1], d[0]])
        tc = float(t0 - 2)
        q = EVENT + 15.0 * perp
        start = q - 2.5 * tc * d
        end = q + 2.5 * (T - 1 - tc) * d
        out.append(m.bezier_centers(start, start + (end - start) / 3, start + 2 * (end - start) / 3, end, T))
    return out


def main():
    C.set_style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = C.outdir("D1_moving")
    prof_rows, entry_rows = [], []
    for name, (d, nn, nd, nm) in RUNS.items():
        if not d.exists():
            continue
        moving_ids = set(range(3 + nn + nd + 1, 3 + nn + nd + nm + 1))     # parent ids (1-based)
        bc = blob_centres(nm)
        for f in sorted((d / "bundles").glob("*.npz")):
            b = np.load(f, allow_pickle=True)
            Cc, P, L, ok = b["centers"], b["parents"], b["lengths"], b["ok"]
            tA, tC = b["truth_centers_A"], b["truth_centers_C"]
            cfg = int(f.name.split("_")[1])
            nsteps = Cc.shape[1]
            cnt = np.zeros(nsteps); mov = np.zeros(nsteps)
            for i in range(len(L)):
                n = int(L[i])
                if int(ok[i]) != 1 or n <= 1:
                    continue
                par = np.asarray(P[i, :n], int)
                for s in range(1, n):
                    cnt[s] += 1; mov[s] += par[s] in moving_ids
                hit = [s for s in range(1, n) if par[s] in moving_ids]
                if hit:
                    s = hit[0]; t = T - 1 - s; k = par[s] - (3 + nn + nd) - 1
                    c = np.asarray(Cc[i, s], float)
                    da = min(np.hypot(*(c - tA[t])), np.hypot(*(c - tC[t]))) if 0 <= t < len(tA) else np.nan
                    entry_rows.append(dict(run=name, cfg=cfg, member=i, entry_step=s, time_index=t,
                                           dist_to_blob=float(np.hypot(*(c - bc[k][t]))),
                                           blob_dist_to_event=float(np.hypot(*(bc[k][t] - EVENT))),
                                           centre_dist_to_event=float(np.hypot(*(c - EVENT))),
                                           dist_to_true_parent=float(da)))
            for s in range(1, nsteps):
                if cnt[s] > 0:
                    prof_rows.append(dict(run=name, cfg=cfg, step=s, moving_share=mov[s] / cnt[s]))
    prof = pd.DataFrame(prof_rows); ent = pd.DataFrame(entry_rows)
    prof.to_csv(out / "D1_moving_share_by_step.csv", index=False)
    ent.to_csv(out / "D1_entry_points.csv", index=False)
    summ = ent.groupby("run").agg(n_members_entering=("member", "size"),
                                  entry_step_median=("entry_step", "median"),
                                  dist_to_blob_median=("dist_to_blob", "median"),
                                  blob_dist_to_event_median=("blob_dist_to_event", "median"),
                                  centre_dist_to_event_median=("centre_dist_to_event", "median"),
                                  dist_to_true_parent_median=("dist_to_true_parent", "median"))
    summ.to_csv(out / "D1_entry_summary.csv"); print(summ.round(2).to_string(), flush=True)
    pm = prof.groupby(["run", "step"])["moving_share"].agg(["mean", lambda x: x.quantile(.1),
                                                            lambda x: x.quantile(.9)]).reset_index()
    pm.columns = ["run", "step", "mean", "p10", "p90"]
    print(pm[pm.step.isin([1, 2, 3, 5, 8, 12, 20, 30])].round(3).to_string(), flush=True)
    cols = dict(zip(RUNS, ["#0072B2", "#D55E00", "#56B4E9", "#E69F00"]))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
    for name, g in pm.groupby("run"):
        ax[0].plot(g["step"], g["mean"], "-o", ms=3, lw=2, color=cols[name], label=name)
        ax[0].fill_between(g["step"], g["p10"], g["p90"], color=cols[name], alpha=0.15, lw=0)
    ax[0].axvline(N_INJECT, color="#999999", ls="--", lw=1)
    ax[0].text(N_INJECT + 0.3, 0.95, "B starts\ndepending on A, C", fontsize=12, va="top", transform=ax[0].get_xaxis_transform())
    ax[0].set_xlabel("Tracing step (days before the event)"); ax[0].set_ylabel("Share of members on a moving blob")
    ax[0].legend(frameon=False, fontsize=12)
    for name, g in ent.groupby("run"):
        ax[1].hist(g["entry_step"], bins=np.arange(0.5, 31.5), histtype="step", lw=2.5, color=cols[name],
                   density=True, label=name)
    ax[1].set_xlabel("Step of first entry into a moving blob"); ax[1].set_ylabel("Fraction of entering members")
    fig.suptitle("Moving blobs: when trajectories jump onto them (existing T7 runs)")
    fig.tight_layout(); C.savefig(fig, "D1_moving", "D1_lag_profile_and_entry"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, g in ent.groupby("run"):
        ax.scatter(g["blob_dist_to_event"], g["dist_to_blob"], s=6, alpha=0.4, color=cols[name], label=name)
    ax.set_xlabel("Blob centre distance to the event at entry (grid units)")
    ax.set_ylabel("Trajectory centre distance to the blob centre (grid units)")
    ax.legend(frameon=False, markerscale=3)
    ax.set_title("Entry points into moving blobs")
    C.savefig(fig, "D1_moving", "D1_entry_geometry"); plt.close(fig)


if __name__ == "__main__":
    main()
