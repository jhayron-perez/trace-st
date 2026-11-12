"""Night-3 figures (figures_v3, no green): N2 lead-dependent diff vs nulls; N5 selection-year twin diffs per variant.

python N3_figures.py   (PBS only). Inputs: night3/analysis/{n2_lead_summary.csv, bed_twins.csv}.
Outputs: revision1/figures_v3/N2_null_diff_by_lead.png, N5_selection_twin_diff.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
A = R1 / "night3/analysis"
OUT = R1 / "figures_v3"
FAM = {"time": ("#1f4e9c", "time shift"), "space": ("#d1495b", "spatial shift"),
       "replace_year": ("#e09f3e", "replace (other years)"), "replace_time": ("#7d5ba6", "replace (±10 d)")}


def fig_n2():
    L = pd.read_csv(A / "n2_lead_summary.csv")
    panels = [("pnw", "Z500"), ("pnw", "TCWV"), ("pnw", "MTNLWRF"), ("pnw", "MSLHF"), ("debby", "vo")]
    fig, axs = plt.subplots(1, 5, figsize=(19, 3.6), sharey=False)
    for ax, (case, var) in zip(axs, panels):
        g = L[(L.case == case) & (L["var"] == var)]
        for fam, (c, lab) in FAM.items():
            h = g[g.family == fam].sort_values("lead")
            if h.empty:
                continue
            ax.plot(h.lead, h.diff_mean, color=c, lw=1.6, label=lab)
            ax.fill_between(h.lead, h.lo, h.hi, color=c, alpha=0.18, lw=0)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_title(f"{case.upper()} {var}", fontsize=10)
        ax.set_xlabel("lead (days)")
    axs[0].set_ylabel("share(real X) − share(null)")
    axs[-1].legend(fontsize=8, frameon=False)
    fig.suptitle("N2: real field vs fairer nulls, baseline method (mean over configs, 95 % bootstrap CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "N2_null_diff_by_lead.png", dpi=150)


def fig_n5():
    T = pd.read_csv(A / "bed_twins.csv")
    keys = [("pnw", "Z500", "w1-10"), ("pnw", "TCWV", "w1-10"), ("debby", "vo", "w1-4"), ("pinatubo", "TMSO201", "w1-3")]
    order = ["base", "bf05", "bf01", "univ1", "univ05", "hybv", "hybl", "stab", "pw", "tst", "tgt", "ko20", "ko0_10",
             "thin2", "cv"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, (case, var, w) in zip(axs, keys):
        g = T[(T.case == case) & (T["var"] == var) & (T.window == w)].set_index("variant")
        vs = [v for v in order if v in g.index]
        y = range(len(vs))
        col = ["0.35" if v == "base" else ("#1f4e9c" if v == "ko0_10" else "#e09f3e") for v in vs]
        ax.barh(list(y), g.loc[vs, "diff_mean"], color=col,
                xerr=[g.loc[vs, "diff_mean"] - g.loc[vs, "lo"], g.loc[vs, "hi"] - g.loc[vs, "diff_mean"]],
                error_kw=dict(lw=0.8, ecolor="0.2"))
        ax.axvline(float(g.loc["base", "diff_mean"]), color="0.35", ls="--", lw=0.8)
        ax.axvline(0, color="0.2", lw=0.8)
        ax.set_yticks(list(y)); ax.set_yticklabels(vs); ax.invert_yaxis()
        ax.set_title(f"{case} {var} {w}", fontsize=10); ax.set_xlabel("share(real) − share(twin)")
    fig.suptitle("N5: twin diff on the selection years (test bed; expensive variants at M 10 / M 5, half configs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "N5_selection_twin_diff.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_n2(); fig_n5()
    print("ok")
