"""Night-4 figures (figures_v4, no green).

A_<case>_decoys_by_lead.png  : share by lead of the main driver(s) and the decoys, baseline (solid) and ko0_10 (dashed),
                               95 % bootstrap CI; other paper variables in grey.
A_summary_3cases.png         : 3 panels (Debby, PNW, Pinatubo), baseline method, main driver vs decoys by lead.
C_taxonomy.png               : per decoy class: median density share of the extra variable and the variable-level
                               selection rate, for baseline, ko0_10 and knockoff_cond.
B1_eval_twin_diff.png        : twin diff (real - twin) on the evaluation years, baseline vs ko0_10, per case/window.
python N4_figures.py   (PBS only). Inputs: night4/analysis/{a_lead_shares,c_taxonomy,b1_twins}.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R1 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1")
A = R1 / "night4" / "analysis"
OUT = R1 / "figures_v4"
BLUE, RED, ORANGE, PURPLE, GREY = "#1f4e9c", "#d1495b", "#e09f3e", "#7d5ba6", "0.55"
ROLES = dict(debby=(["vo"], ["CAPE", "DIV200"]), pnw=(["Z500", "TCWV"], ["T2M", "TCC"]),
             pinatubo=(["TMSO201", "BURDENSO401"], ["FLUT"]))
COL = {"vo": BLUE, "Z500": BLUE, "TCWV": PURPLE, "TMSO201": BLUE, "BURDENSO401": PURPLE,
       "CAPE": RED, "DIV200": ORANGE, "T2M": RED, "TCC": ORANGE, "FLUT": RED}
XMAX = dict(debby=15, pnw=30, pinatubo=16)
TITLE = dict(debby="Debby 2006 (+ CAPE, DIV200)", pnw="PNW 2021 (+ T2M, TCC)", pinatubo="Pinatubo (+ SO2/SO4 tag-03 twins, FLUT)")


def panel(ax, L, case, methods=("base", "ko0_10")):
    mains, decoys = ROLES[case]
    g = L[(L.case == case) & (L.lead <= XMAX[case])]
    for meth, ls in zip(methods, ("-", "--")):
        h = g[g.method == meth]
        if h.empty:
            continue
        for v in sorted(h["var"].unique()):
            x = h[h["var"] == v].sort_values("lead")
            c = COL.get(v, GREY); lw = 2.0 if v in mains + decoys else 0.8
            ax.plot(x.lead, x.share, ls=ls, color=c, lw=lw, label=f"{v} ({meth})" if v in mains + decoys else None)
            if v in mains + decoys and meth == methods[0]:
                ax.fill_between(x.lead, x.lo, x.hi, color=c, alpha=0.15, lw=0)
    ax.set_title(TITLE[case], fontsize=10); ax.set_xlabel("lead (days)"); ax.set_ylabel("density share")
    ax.set_ylim(0, None)


def fig_a():
    f = A / "a_lead_shares.csv"
    if not f.exists():
        return
    L = pd.read_csv(f)
    for case in ROLES:
        if (L.case == case).any():
            fig, ax = plt.subplots(figsize=(7, 4))
            panel(ax, L, case); ax.legend(fontsize=7, frameon=False, ncol=2)
            fig.tight_layout(); fig.savefig(OUT / f"A_{case}_decoys_by_lead.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4))
    for ax, case in zip(axs, ROLES):
        if (L.case == case).any():
            panel(ax, L, case, methods=("base",)); ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Real-case decoys: main drivers (blue/purple) vs physically linked decoys (red/orange), baseline method; "
                 "grey = other paper variables", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "A_summary_3cases.png", dpi=150); plt.close(fig)


def fig_c():
    f = A / "c_taxonomy.csv"
    if not f.exists():
        return
    S = pd.read_csv(f)
    order = [r for r in ["c_noise", "c_resp_b04_phi05", "c_resp_b04_phi08", "c_resp_b08_phi05", "c_resp_b08_phi08",
                         "c_proxy_b04_phi05", "c_proxy_b04_phi08", "c_proxy_b08_phi05", "c_proxy_b08_phi08",
                         "c_mov_event", "c_mov_random"] if r in set(S.run)]
    meths = [m for m in ("base", "ko0_10", "kocond") if m in set(S.method)]
    cols = dict(base=GREY, ko0_10=BLUE, kocond=PURPLE)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.5), sharey=True)
    y = np.arange(len(order)); h = 0.8 / max(len(meths), 1)
    for k, m in enumerate(meths):
        s = S[S.method == m].set_index("run").reindex(order)
        axs[0].barh(y + k * h, s["share_extra_median"], height=h, color=cols[m], label=m)
        axs[1].barh(y + k * h, s.get("sel_rate_extra", np.nan), height=h, color=cols[m], label=m)
    for ax, t in zip(axs, ("median density share of the extra variable", "selection rate of the extra block (per regression)")):
        ax.set_yticks(y + h * (len(meths) - 1) / 2); ax.set_yticklabels([o.replace("c_", "") for o in order])
        ax.invert_yaxis(); ax.set_xlabel(t); ax.axvline(0.10, color="0.3", ls=":", lw=0.8)
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Synthetic decoy taxonomy (T2-EN, alpha 0.6, M 20, p4 + 1 extra variable)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "C_taxonomy.png", dpi=150); plt.close(fig)


def fig_b1():
    f = A / "b1_twins.csv"
    if not f.exists():
        return
    S = pd.read_csv(f)
    keys = [("pnw", "Z500"), ("pnw", "TCWV"), ("pnw", "MTNLWRF"), ("pnw", "MSLHF"), ("debby", "vo"), ("pinatubo", "TMSO201")]
    keys = [k for k in keys if ((S.case == k[0]) & (S["var"] == k[1])).any()]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.8), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (case, var) in zip(axs, keys):
        g = S[(S.case == case) & (S["var"] == var)]
        ws = sorted(g.window.unique(), key=lambda w: int(w[1:].split("-")[0]))
        for k, (m, c) in enumerate((("base", GREY), ("ko0_10", BLUE))):
            h = g[g.variant == m].set_index("window").reindex(ws)
            x = np.arange(len(ws)) + (k - 0.5) * 0.35
            ax.bar(x, h.diff_mean, width=0.35, color=c, label=m,
                   yerr=[h.diff_mean - h.lo, h.hi - h.diff_mean], error_kw=dict(lw=0.8))
        ax.axhline(0, color="0.2", lw=0.8); ax.set_xticks(range(len(ws))); ax.set_xticklabels(ws)
        ax.set_title(f"{case} {var}", fontsize=10)
    axs[0].set_ylabel("share(real) − share(twin)"); axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("B1 (pre-registered): ko0_10 vs baseline on the evaluation years (PNW 2006/2012/2020, Debby 2003/2005/2008; "
                 "Pinatubo combined set)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "B1_eval_twin_diff.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_a(); fig_c(); fig_b1()
    print("ok", sorted(p.name for p in OUT.iterdir()))
