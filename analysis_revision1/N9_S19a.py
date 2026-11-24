"""Night-9 N3 (fork F2): S19a redrawn from night9/analysis/T5_supp/G/G_profiles_linear_vs_all.csv (written by N9_supp.py --which S19)
with the panel-c legend below the axes (the night-5/6 legend sat over the curves). Same data and style as N5_T7_supp.run_G.
-> figures_v9/S19_longer_N_relaxed/S19a_longer_N.{pdf,png}   (PBS develop)"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
S.set_output(R1 / "figures_v9")
S.set_style()
G = pd.read_csv(R1 / "night9/analysis/T5_supp/G/G_profiles_linear_vs_all.csv")
GL = G[G.subset == "linear"]
fig, ax = plt.subplots(1, 3, figsize=(S.DOUBLE, 7.2 * S.CM))
for N, g in GL.groupby("N"):
    ls = "-" if N == 40 else "--"
    ax[0].plot(g["step"], g["disp_mean"], ls, color="#0072B2", label=f"N = {N}")
    ax[1].plot(g["step"], g["dist_origin_median"], ls, color="#D55E00", label=f"N = {N}")
    ax[1].fill_between(g["step"], g["dist_origin_median"] - g["dist_origin_std"], g["dist_origin_median"] + g["dist_origin_std"],
                       color="#D55E00", alpha=0.12, lw=0)
    ax[2].plot(g["step"], g["frac_parent_v1"], ls, color="#000000", label=f"$V_1$ (true parent), N = {N}")
    ax[2].plot(g["step"], g["frac_parent_v2"], ls, color="#999999", label=f"$V_2$, N = {N}")
for a_ in ax:
    a_.axvline(30, color="#999999", lw=0.7, ls=":")
    a_.axvline(39, color="#CC79A7", lw=0.9, ls=":")
    a_.set_xlabel("Backward tracing step (time steps)")
ax[1].text(39.5, 0.97, "true origin", color="#CC79A7", fontsize=6, transform=ax[1].get_xaxis_transform(), va="top")
ax[0].text(29.5, 0.80, "paper N = 30", color="#777777", fontsize=6, transform=ax[0].get_xaxis_transform(), va="top", ha="right", rotation=90)
ax[0].set_ylabel("Mean step displacement\n(grid cells)")
ax[1].set_ylabel("Distance to true origin,\nmedian ± sd (grid cells)")
ax[2].set_ylabel("Fraction of steps by parent")
ax[0].legend(loc="upper left")
ax[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=5.5, columnspacing=1.0)
S.panel_labels(ax, x=-0.30)
fig.tight_layout(w_pad=2.2)
print(S.save(fig, "S19_longer_N_relaxed", "S19a_longer_N"))
