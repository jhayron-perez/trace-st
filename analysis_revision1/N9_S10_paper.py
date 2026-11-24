"""S10 paper version: panels A (true-parent share) and B (|alpha error|) vs p; pooled line over the 9 configs with 95 % bootstrap band; no best/random lines.
Reads night8/analysis/S10_summary.csv (N8_S10.py). -> figures_v9/S10_synthetic/S10_synthetic_p_curve_paper.{pdf,png}"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
S.set_output(R1 / "figures_v9")
S.set_style()
sm = pd.read_csv(R1 / "night8/analysis/S10_summary.csv")
h = sm[sm.subset == "pooled"].sort_values("p")
fig, ax = plt.subplots(1, 2, figsize=(S.DOUBLE, 5.5 * S.CM))
for a, (col, ttl) in zip(ax, [("share_true", "True-parent share (A + C)"), ("alpha_err", "|α error|")]):
    a.plot(h.p, h[col], "-", color="#000000", lw=1.8, ms=3.5, label="Pooled (9 configs)")
    a.fill_between(h.p, h[col + "_lo"], h[col + "_hi"], color="#000000", alpha=0.12, lw=0, label="95 % bootstrap CI")
    a.set_ylabel(ttl); a.set_xlabel("Number of variables p"); a.set_xticks([3, 6, 9, 12, 15, 18, 21])
ax[0].legend(loc="lower left")
S.panel_labels(ax)
fig.tight_layout()
print(S.save(fig, "S10_synthetic", "S10_synthetic_p_curve_paper"))
print(h[["p", "n_rows", "n_configs", "share_true", "share_true_lo", "share_true_hi", "alpha_err", "alpha_err_lo", "alpha_err_hi"]].round(3).to_string())
