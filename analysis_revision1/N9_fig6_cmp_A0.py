"""Night-9 N4 follow-up: PNW shares by lead, typical-pool decoy (T1b, not rank 1) vs pooled decoy (N4, not rank 1), same axes.
Reads figures_v9/Fig6_pnw/fig6_pnw_final_A0_stats.csv and ..._A_pooled_stats.csv (+ decoy_share csv); writes
figures_v9/Fig6_pnw/fig6_pnw_A0_typical_vs_pooled.{pdf,png} and ..._compare.csv.  PBS/Casper only."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402

D = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/figures_v9/Fig6_pnw")
A = pd.read_csv(D / "fig6_pnw_final_A0_stats.csv"); P = pd.read_csv(D / "fig6_pnw_final_A0_pooled_stats.csv")
A = A[(A.set == "final") & (A.cfgset == "own")]; P = P[(P.set == "final") & (P.cfgset == "own")]
dA = pd.read_csv(D / "fig6_pnw_final_A0_decoy_share.csv"); dP = pd.read_csv(D / "fig6_pnw_final_A0_pooled_decoy_share.csv")
S.set_style()
PNWC = {"Z500": "#000000", "Z10": "#999999", "OLR": "#CC79A7", "TCWV": "#0072B2", "SLHF": "#E69F00", "SSHF": "#D55E00", "T2m": "#882255"}  # as in Fig. 6 (no green)
RAW = {"Z500": "Z500", "Z10": "Z10", "OLR": "MTNLWRF", "TCWV": "TCWV", "SLHF": "MSLHF", "SSHF": "MSSHF", "T2m": "T2M"}
vars_ = ["Z500", "Z10", "OLR", "TCWV", "SLHF", "SSHF", "T2m"]
fig, ax = plt.subplots(1, 2, figsize=(17.5 / 2.54, 6.2 / 2.54), sharey=True)
for a, T, dd, ttl in ((ax[0], A, dA, f"Typical-field decoy (T1b), no filter, {int(A.n_cfg.max())} configs"), (ax[1], P, dP, f"Pooled decoy (N4), no filter, {int(P.n_cfg.max())} configs")):
    for v in vars_:
        t = T[T["var"] == RAW[v]].sort_values("lead")
        c = PNWC[v]
        a.plot(t.lead, t.share, color=c, lw=1.0, label=v)
        a.fill_between(t.lead, t.lo, t.hi, color=c, alpha=0.18, lw=0)
    a.plot(dd.lead, dd.share, color="0.4", lw=0.8, ls="--", label="Decoy")
    a.fill_between(dd.lead, dd.lo, dd.hi, color="0.45", alpha=0.3, lw=0)
    a.set_title(f"{'AB'[list(ax).index(a)]}. {ttl}", loc="left", fontsize=7.5, fontweight="bold", pad=2); a.set_xlabel("Lead (days)")
ax[0].set_ylabel("Share of all candidates"); ax[1].legend(fontsize=5.5, ncol=2, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(D / "fig6_pnw_A0_typical_vs_pooled.pdf", bbox_inches="tight"); fig.savefig(D / "fig6_pnw_A0_typical_vs_pooled.png", dpi=300, bbox_inches="tight")
m = A.merge(P, on=["lead", "var"], suffixes=("_typical", "_pooled"))[["lead", "var", "n_cfg_typical", "share_typical", "lo_typical", "hi_typical", "n_cfg_pooled", "share_pooled", "lo_pooled", "hi_pooled"]]
m.to_csv(D / "fig6_pnw_A0_typical_vs_pooled_compare.csv", index=False)
