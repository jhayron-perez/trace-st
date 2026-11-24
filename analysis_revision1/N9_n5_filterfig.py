"""Night-9 N5 / N3 supplement: filter figure (figures_v9/S_filter/filter_figure.{pdf,png}).
Per case (rows: PNW, Pinatubo, Debby when night9/analysis/N1_debby_configs.csv exists): (a) configs removed by each rule (of the
harmonized-retained configs of the decoy run), (b) distribution of the decoy share over configs with 1/p marked, (c) real-variable
shares in the headline window before / after each rule (95 % bootstrap CI over configs, 2,000 resamples, seed 0).
Headline windows: PNW 11-30 d, Pinatubo 4-16 d, Debby 5-15 d. Rules: not rank 1 (R1), equal share (R2).
python N9_n5_filterfig.py   (PBS only)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
A8, A9 = R1 / "night8/analysis", R1 / "night9/analysis"
FINAL = dict(pnw=["Z500", "Z10", "OLR", "TCWV", "SLHF", "SSHF", "T2m"],
             pinatubo=["TMSO201", "BURDENSO401", "TMH2SO401", "AEROD_v", "TMSO203", "BURDENSO403", "T050"],
             debby=["Tb", "vo", "precip", "CAPE", "DIV200", "MSLP"])
WIN = dict(pnw="11-30", pinatubo="4-16", debby="5-15")
TITLE = dict(pnw="PNW heat wave 2021", pinatubo="Pinatubo 1991", debby="TS Debby 2006")
# paper variable colours (N5_style keys); T2m/OLR etc. use the night-5 names
KEY = dict(pnw=dict(Z500="Z500", Z10="Z10", OLR="MTNLWRF", TCWV="TCWV", SLHF="MSLHF", SSHF="MSSHF", T2m="T2M"))
RCOL = dict(unf="#000000", R1="#0072B2", R2="#D55E00")


def boot(x, rng, n=2000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    S.set_style(); S.set_output(R1 / "figures_v9")
    plt = S.plt
    cases = [c for c in ("pnw", "pinatubo", "debby") if (A8 / f"T1b_{c}_configs.csv" if c != "debby" else A9 / "N1_debby_configs.csv").exists()]
    fig, ax = plt.subplots(len(cases), 3, figsize=(S.DOUBLE, 5.4 * len(cases) / 2 + 0.2),
                           gridspec_kw=dict(width_ratios=[0.8, 1.2, 1.8]), squeeze=False)
    rng = np.random.default_rng(0)
    for r, c in enumerate(cases):
        C = pd.read_csv(A8 / f"T1b_{c}_configs.csv" if c != "debby" else A9 / "N1_debby_configs.csv")
        C = C[C["decoy_share"].notna()]
        n, p = len(C), len(FINAL[c]) + 1
        a = ax[r, 0]
        rem = [int(C.R1_drop.sum()), int(C.R2_drop.sum())]
        a.bar([0, 1], rem, color=[RCOL["R1"], RCOL["R2"]], width=0.6)
        for i, v in enumerate(rem):
            a.text(i, v, f"{v}\n({100 * v / n:.0f} %)", ha="center", va="bottom", fontsize=6)
        a.set_xticks([0, 1]); a.set_xticklabels(["not\nrank 1", "equal\nshare"]); a.set_ylim(0, max(max(rem) * 1.35, 3)); a.yaxis.set_major_locator(S.mpl.ticker.MaxNLocator(integer=True))
        a.set_ylabel(f"configs removed (of {n})"); a.set_title(TITLE[c], loc="left", fontsize=7.5)
        b = ax[r, 1]
        hi = max(0.3, float(C.decoy_share.max()) * 1.05)
        b.hist(C.decoy_share, bins=np.linspace(0, hi, 31), color="#999999", edgecolor="white", linewidth=0.3)
        b.axvline(1 / p, color=RCOL["R2"], lw=1.0); b.text(1 / p, b.get_ylim()[1] * 0.97, " 1/p", color=RCOL["R2"], fontsize=6, va="top")
        b.set_xlabel("decoy share of the total density"); b.set_ylabel("configs")
        d = ax[r, 2]
        w = WIN[c]
        sets = [("unf", "unfiltered", C), ("R1", "not rank 1", C[~C.R1_drop.astype(bool)]), ("R2", "equal share", C[~C.R2_drop.astype(bool)])]
        for j, v in enumerate(FINAL[c]):
            for k, (key, lab, D) in enumerate(sets):
                m, lo, hi_ = boot(D[f"s_{w}_{v}"], rng)
                x = j + (k - 1) * 0.22
                d.errorbar(x, m, yerr=[[m - lo], [hi_ - m]], fmt="o" if key == "unf" else ("^" if key == "R1" else "s"), ms=3, lw=0.8,
                           color=RCOL[key], label=f"{lab} (n={len(D)})" if j == 0 else None)
        d.set_xticks(range(len(FINAL[c]))); d.set_xticklabels([S.lab(KEY.get(c, {}).get(v, v)) for v in FINAL[c]], rotation=40, ha="right")
        d.set_ylabel(f"real-variable share, {w} d"); d.legend(loc="upper right" if c == "pinatubo" else "upper left", fontsize=5.5)
    for r in range(len(cases)):
        S.panel_labels([ax[r, 0], ax[r, 1], ax[r, 2]], start=3 * r)
    fig.tight_layout(w_pad=1.2, h_pad=1.5)
    print(S.save(fig, "S_filter", "filter_figure"), flush=True)


if __name__ == "__main__":
    main()
