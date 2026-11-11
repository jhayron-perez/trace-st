"""Task E (R1 major 2): why merging all directions of one (variable, sign) group stalls the tracker.

One radius-2 stencil, variable 1, positive links only: two collinear parent links to the
east (|c| = 0.50, 0.46) and two to the west (|c| = 0.44, 0.42), i.e. two DBSCAN direction
clusters from opposite directions. Movement is computed with the real
trace_st.movement.pick_direction_then_weighted_delta and the T1 defaults
(eps 0.15, min_samples 2, score 'mean', alpha 3, deterministic):

- grouping='dbscan' (paper): two direction clusters compete; the east one wins -> ~(0, +1.5).
- grouping='variable_sign': one group, weighted mean of opposite links -> ~(0, +0.2): stall.

python analysis_revision1/illustrate_grouping.py [OUTDIR]
(default OUTDIR /glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/figures_v2/E_grouping)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trace_st import movement  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/figures_v2/E_grouping")
R = 2
# Okabe-Ito without bluish green and yellow
ORANGE, SKY, BLUE, VERM, PURPLE, BLACK = "#E69F00", "#56B4E9", "#0072B2", "#D55E00", "#CC79A7", "#000000"
KW = dict(radius=R, spaceres=1.0, eps=0.15, min_samples=2, montecarlo=False, prefer_sign="both",
          alpha=3.0, score_mode="mean", prob_rule="linear", winner_summary="mean")


def stencil():
    s = 2 * R + 1
    links = [((0, 1), 0.50), ((0, 2), 0.46), ((0, -1), 0.44), ((0, -2), 0.42)]  # ((di, dj), c); di < 0 north
    rows = [("p1_c1", c, ((di + R) * s + dj + R, 0, 1)) for (di, dj), c in links]
    df = pd.DataFrame(rows, columns=["vars", "causal_value", "position"])
    df["causal_value_abs"] = df["causal_value"].abs()
    return df


def panel(ax, dbg, d_lat, d_lon, chosen, title, colors):
    for k in range(-R, R + 1):
        ax.axhline(k, color="0.85", lw=0.8, zorder=0)
        ax.axvline(k, color="0.85", lw=0.8, zorder=0)
    for cl, g in dbg.groupby("cluster_global"):
        c = colors[int(cl) % len(colors)]
        for _, r in g.iterrows():
            ax.scatter(r.delta_lon, r.delta_lat, s=3000 * r.causal_value_abs ** 2, color=c, alpha=0.85,
                       edgecolor=BLACK, lw=1, zorder=3)
            ax.text(r.delta_lon, r.delta_lat + (0.5 if abs(r.delta_lon) > 1.5 else -0.5), f"{r.causal_value_abs:.2f}",
                    ha="center", va="center", fontsize=16, color=c)
    ax.annotate("", xy=(d_lon, d_lat), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=4, mutation_scale=30), zorder=5)
    ax.plot(0, 0, "o", color=BLACK, ms=10, zorder=6)
    ax.text(0, 0.35, "centre", ha="center", fontsize=16)
    ax.text(0, -1.6, "parent links: circles (size ~ |c|),\ncolour = group; arrow = step",
            ha="center", va="center", fontsize=16, color="0.3")
    ax.set_xlim(-R - 0.8, R + 0.8)
    ax.set_ylim(-R - 0.8, R + 0.8)
    ax.set_aspect("equal")
    ax.set_xticks(range(-R, R + 1))
    ax.set_yticks(range(-R, R + 1))
    ax.tick_params(labelsize=16)
    ax.set_xlabel("stencil offset east (grid cells)", fontsize=16)
    ax.set_title(f"{title}\nstep = ({d_lat:+.2f} N, {d_lon:+.2f} E)", fontsize=18)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = stencil()
    res = {}
    for g in ("dbscan", "variable_sign", "sectors8", "kmeans"):
        d_lat, d_lon, sign, cl, dbg = movement.pick_direction_then_weighted_delta(df, grouping=g, **KW)
        res[g] = (d_lat, d_lon, cl, dbg)
        print(f"{g:14s} step=({d_lat:+.3f}, {d_lon:+.3f}) n_groups={dbg['cluster_global'].nunique()} chosen={cl}")
    print("strongest      step=(+0.000, +1.000) (single strongest link, east, |c|=0.50)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), constrained_layout=True)
    d = res["dbscan"]
    panel(axes[0], d[3], d[0], d[1], d[2], "grouping = dbscan (paper)", [ORANGE, BLUE, VERM, PURPLE])
    v = res["variable_sign"]
    panel(axes[1], v[3], v[0], v[1], v[2], "grouping = variable_sign", [PURPLE])
    axes[0].set_ylabel("stencil offset north (grid cells)", fontsize=16)
    fig.suptitle("Opposite parent clusters of one variable/sign: direction clusters move, merging stalls",
                 fontsize=18)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"illustrate_grouping.{ext}", dpi=200)
    print("wrote", OUT / "illustrate_grouping.png", OUT / "illustrate_grouping.pdf")


if __name__ == "__main__":
    main()
