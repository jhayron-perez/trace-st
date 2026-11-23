"""Night-8 T1b figure: split-half decoy shares, random-removal null of rank-1 agreement, shares filtered vs unfiltered vs random.
python N8_T1b_fig.py -> revision1/figures_v8/T1b/T1b_filter_test.{png,pdf}"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
A = R1 / "night8" / "analysis"
OUT = R1 / "figures_v8" / "T1b"
COL = dict(R1="#0072B2", R2="#D55E00", unf="#000000", rnd="#999999")
WIN = dict(pnw="11-30", pinatubo="4-16")
HEADW = dict(pnw="1-10", pinatubo="4-16")
TITLE = dict(pnw="PNW heat wave 2021 (300 configs)", pinatubo="Pinatubo 1991 (1000 configs)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.4))
    for r, case in enumerate(["pnw", "pinatubo"]):
        Cf = pd.read_csv(A / f"T1b_{case}_configs.csv"); Cr = pd.read_csv(A / f"T1b_{case}_criteria.csv").set_index("rule")
        Sh = pd.read_csv(A / f"T1b_{case}_shares.csv")
        p = 8
        a = ax[r, 0]
        a.scatter(Cf.decoy_h1, Cf.decoy_h2, s=6, c="#555555", alpha=.6)
        lim = [0, max(Cf.decoy_h1.max(), Cf.decoy_h2.max()) * 1.05]
        a.plot(lim, lim, c="#bbbbbb", lw=.8); a.axvline(1 / p, c=COL["R2"], lw=1); a.axhline(1 / p, c=COL["R2"], lw=1)
        a.set_xlabel("decoy share, members 0-14"); a.set_ylabel("decoy share, members 15-29")
        a.set_title(f"{TITLE[case]}\nsplit-half: R1 agree {Cr.loc['R1','split_half_agreement']:.2f}, R2 agree {Cr.loc['R2','split_half_agreement']:.2f}", fontsize=8)
        b = ax[r, 1]
        for rule in ("R1", "R2"):
            f = A / f"T1b_{case}_{rule}_random.csv"
            if f.exists():
                b.hist(pd.read_csv(f).rank1_agree, bins=25, alpha=.5, color=COL[rule], label=f"random removal ({rule}, n removed {int(Cr.loc[rule,'n_removed'])})")
            b.axvline(Cr.loc[rule, "rank1_agree_filtered"], c=COL[rule], lw=2, ls="--")
        b.axvline(Cr.loc["R1", "rank1_agree_unfiltered"], c=COL["unf"], lw=1.5, label="unfiltered")
        b.set_xlabel(f"rank-1 agreement, {HEADW[case]} d (modal variable's fraction)"); b.set_ylabel("random removals"); b.legend(fontsize=6)
        b.set_title("dashed = filtered", fontsize=8)
        c = ax[r, 2]
        w = Sh[(Sh.window == WIN[case]) & (Sh.rule == "R2")]
        x = np.arange(len(w))
        c.scatter(x, w.share_unfiltered, c=COL["unf"], s=22, label="unfiltered", zorder=3)
        c.scatter(x, w.share_filtered, c=COL["R2"], marker="s", s=18, label="R2 filtered")
        c.scatter(x, Sh[(Sh.window == WIN[case]) & (Sh.rule == "R1")].share_filtered, c=COL["R1"], marker="^", s=18, label="R1 filtered")
        c.scatter(x, w.share_random_median, c=COL["rnd"], marker="x", s=22, label="random-removal median (R2)")
        c.set_xticks(x); c.set_xticklabels(w["var"], rotation=45, fontsize=7); c.set_ylabel(f"real-variable share, {WIN[case]} d"); c.legend(fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"T1b_filter_test.{ext}", dpi=300)


if __name__ == "__main__":
    main()
