"""Night-9 N3 (fork F2): final S10 (T2-EN synthetic, variable count p = 3..21) from the night-8 S10 runs, family colours.
Reads night8/analysis/S10_summary.csv and S10_linkstats.csv (N8_S10.py; 9 pre-declared configs x alpha 0.2/0.6, M 20, linear rule).
Panels: a true-parent share, b |alpha error| (best 5 / random 5 / pooled, 95 % CI of pooled), c density share of the added families
vs 1/p, d link-level selection (solid) vs density share (dashed) per family.  -> figures_v9/S10_synthetic/S10_synthetic_p_curve.{pdf,png}
python N9_S10.py   (PBS develop)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
S.set_output(R1 / "figures_v9")
S.set_style()
sm = pd.read_csv(R1 / "night8/analysis/S10_summary.csv")
ls = pd.read_csv(R1 / "night8/analysis/S10_linkstats.csv")
FAMC = dict(typical="#0072B2", response="#D55E00", nonlinear="#CC79A7")
FAML = dict(typical="Typical field", response="Response (driven by B)", nonlinear="Sibling (nonlinear)")
fig, ax = plt.subplots(2, 2, figsize=(S.DOUBLE, 10.5 * S.CM))
sty = dict(best=("o-", "#555555", "Best 5 configs"), random=("s--", "#999999", "Random 5 configs"), pooled=("-", "#000000", "Pooled (9 configs)"))
for k, (col, ttl) in enumerate([("share_true", "True-parent share (A + C)"), ("alpha_err", "|α error|")]):
    a = ax[0, k]
    for lab in ("best", "random", "pooled"):
        h = sm[sm.subset == lab].sort_values("p")
        m, c, name = sty[lab]
        a.plot(h.p, h[col], m, color=c, lw=1.8 if lab == "pooled" else 1.0, ms=3.5, label=name)
        if lab == "pooled":
            a.fill_between(h.p, h[col + "_lo"], h[col + "_hi"], color="#000000", alpha=0.12, lw=0)
    a.set_ylabel(ttl); a.set_xlabel("Number of variables p"); a.set_xticks([3, 6, 9, 12, 15, 18, 21])
ax[0, 0].legend(loc="lower left")
h = sm[sm.subset == "pooled"].sort_values("p")
a = ax[1, 0]
for f in FAMC:
    hh = h.dropna(subset=[f"share_{f}"])
    a.plot(hh.p, hh[f"share_{f}"], "o-", color=FAMC[f], ms=3.5, label=FAML[f])
    a.fill_between(hh.p, hh[f"share_{f}_lo"], hh[f"share_{f}_hi"], color=FAMC[f], alpha=0.15, lw=0)
pp = np.arange(3, 22)
a.plot(pp, 1.0 / pp, ":", color="#555555", lw=1.0, label="1/p")
a.set_ylabel("Density share per family"); a.set_xlabel("Number of variables p"); a.set_xticks([3, 6, 9, 12, 15, 18, 21])
a.set_ylim(0, 0.10); a.legend(loc="upper left", ncol=1, fontsize=5.5)
a = ax[1, 1]
for f in FAMC:
    a.plot(ls.p, ls[f], "o-", color=FAMC[f], ms=3.5)
    hh = h.dropna(subset=[f"share_{f}"])
    a.plot(hh.p, hh[f"share_{f}"], "o--", color=FAMC[f], ms=3, mfc="white")
a.set_ylabel("Fraction / share"); a.set_xlabel("Number of variables p"); a.set_xticks([6, 9, 12, 15, 18, 21])
a.plot([], [], "k-", label="Link-level selection (B regressions with any link)"); a.plot([], [], "k--", label="Density share")
a.legend(loc="center right", fontsize=5.5)
S.panel_labels(ax.ravel())
fig.tight_layout()
print(S.save(fig, "S10_synthetic", "S10_synthetic_p_curve"))
