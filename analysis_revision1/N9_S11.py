"""Night-9 S11 figure: performance vs number of typical fields K (rows: cases; columns: leader shares, Kendall tau vs K = 1,
fraction of runs in which an added field outranks the window's leaders, mean share per added field vs 1/p). Reads
night9/analysis/N2_<case>_summary.csv. Output figures_v9/S11_realcase_vs_K/s11_realcase_vs_K.{pdf,png}."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
S.set_style()
S.set_output(R1 / "figures_v9")
COL = dict(OLR="#CC79A7", SLHF="#E69F00", SSHF="#D55E00", T2m="#882255", Z10="#999999")
col = lambda v: COL.get(v, S.col(v))
cases = [c for c in ("debby", "pinatubo", "pnw") if (R1 / f"night9/analysis/N2_{c}_summary.csv").exists()]
LEAD = dict(pnw={"1-10": ["Z500", "TCWV"], "11-30": ["OLR", "SLHF", "SSHF"]}, pinatubo={"4-16": ["TMSO201"]},
            debby={"1-4": ["vo"], "5-15": ["vo"]})
TITLE = dict(debby="Debby", pinatubo="Pinatubo", pnw="PNW")
WS = {"1-10": "-", "11-30": "--", "4-16": "-", "1-4": "-", "5-15": "--"}
fig, axes = plt.subplots(len(cases), 4, figsize=(S.DOUBLE, 2.55 * len(cases) + 0.2), squeeze=False)
for r, case in enumerate(cases):
    d = pd.read_csv(R1 / f"night9/analysis/N2_{case}_summary.csv")
    ax = axes[r]
    for w, ls in [(w, WS[w]) for w in d.window.unique()]:
        g = d[d.window == w].dropna(subset=["tau"])
        for v in LEAD[case][w]:
            ax[0].plot(g.K, g[f"share_{v}"], ls, color=col(v), marker="o", ms=2.5, label=f"{S.lab(v)} {w} d")
            ax[0].fill_between(g.K, g[f"lo_{v}"], g[f"hi_{v}"], color=col(v), alpha=0.15, lw=0)
        ax[1].errorbar(g.K, g.tau, yerr=[g.tau - g.tau_lo, g.tau_hi - g.tau], ls=ls, color="#000000", marker="o", ms=2.5, lw=0.8, capsize=1.5,
                       label=f"{w} d")
        ax[2].plot(g.K, g.outrank_any, ls, color="#D55E00", marker="o", ms=2.5, label=f"any leader, {w} d")
        ax[2].plot(g.K, g.outrank_all, ls, color="#332288", marker="s", ms=2.5, label=f"all leaders, {w} d")
        ax[3].plot(g.K, g.added_mean, ls, color="#000000", marker="o", ms=2.5, label=f"mean per added field, {w} d")
        ax[3].plot(g.K, g.added_max, ls, color="#999999", marker="s", ms=2.5, label=f"largest added field, {w} d")
    k = np.array(sorted(d.K.unique()))
    ax[3].plot(k, 1.0 / (k + len(S.VAR_COLORS) * 0 + (d.p.iloc[0] - d.K.iloc[0])), color="#0072B2", lw=0.8, ls=":", label="1/p")
    ax[0].set_ylabel(f"{TITLE[case]}\nshare of all candidates\n(real + typical fields)")
    ax[1].set_ylabel(r"Kendall $\tau$ vs $K=1$"); ax[2].set_ylabel("fraction of runs"); ax[3].set_ylabel("share of all candidates")
    ax[1].set_ylim(-0.1, 1.05); ax[2].set_ylim(-0.03, 1.05)
    for a in ax:
        a.set_xticks([1, 3, 6, 9, 12, 15]); a.set_xlabel("K (typical fields, incl. decoy)", fontsize=6)
    n = d[d.K > 1].groupby("K").runs_retained.first()
    ax[1].set_title("retained runs/15: " + ", ".join(f"{int(v)}" for k_, v in n.items()), fontsize=5, pad=2)
    _unused = (
               )
    for a, loc in zip(ax, ("best", "lower left", "upper left", "upper right")):
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=4.8, handlelength=1.6, ncol=1)
S.panel_labels(axes.ravel(), x=-0.28, y=1.02)
fig.subplots_adjust(wspace=0.6, hspace=0.95, left=0.09, right=0.99, top=0.95, bottom=0.04)
print(S.save(fig, "S11_realcase_vs_K", "s11_realcase_vs_K"))
