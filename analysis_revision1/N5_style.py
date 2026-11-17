"""Night-5 figure style (paper-ready): one style for every figures_v5 figure.

- Widths: single column 8.5 cm, double column 17.5 cm (SINGLE, DOUBLE in inches).
- Fonts 7-8 pt (Arial/Helvetica fallback DejaVu Sans), line widths 1.0, panel labels a, b, c (bold, 9 pt).
- Colour-blind-safe, no green: Okabe-Ito subset without green/yellow plus greys; one colour per variable (same
  colours as figures_v2-v4 for the paper variables), one hue per role. Lead-window colours are the paper's
  (9_AnalysisHW_Figs45_v2 contour_color_by_window: 1-10 d #c98245, 11-20 d #5a284d, 21-30 d #5f93d1).
- save(): vector PDF + 300-dpi PNG in revision1/figures_v5/<folder>/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG5 = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/figures_v5")
CM = 1 / 2.54
SINGLE, DOUBLE = 8.5 * CM, 17.5 * CM
WINDOW_COLORS = ["#c98245", "#5a284d", "#5f93d1"]

# paper variables keep the figures_v2-v4 colours; new variables get the remaining no-green colours / greys
VAR_COLORS = {
    # Debby
    "vo": "#0072B2", "Tb": "#CC79A7", "precip": "#E69F00", "CAPE": "#D55E00", "DIV200": "#882255",
    "WS10": "#56B4E9", "MSLP": "#555555",
    # Pinatubo
    "TMSO201": "#D55E00", "BURDENSO401": "#0072B2", "TMH2SO401": "#CC79A7", "AEROD_v": "#999999",
    "TMSO203": "#E69F00", "BURDENSO403": "#56B4E9", "FLUT": "#882255", "T050": "#000000", "FSDS": "#AA4499",
    # PNW
    "Z500": "#000000", "Z10": "#999999", "MTNLWRF": "#CC79A7", "TCWV": "#0072B2", "MSLHF": "#E69F00",
    "MSSHF": "#D55E00", "T2M": "#882255", "BLH": "#56B4E9",
}
LABELS = {"precip": "Precip.", "Tb": "BT", "vo": "Vorticity", "CAPE": "CAPE", "DIV200": "DIV200", "WS10": "WS10",
          "MSLP": "MSLP", "TMSO201": "SO$_2$", "BURDENSO401": "SO$_4$", "TMH2SO401": "H$_2$SO$_4$",
          "AEROD_v": "AOD", "TMSO203": "SO$_2$ (tag 03)", "BURDENSO403": "SO$_4$ (tag 03)", "FLUT": "FLUT",
          "T050": "T050", "FSDS": "FSDS", "Z500": "Z500", "Z10": "Z10", "MTNLWRF": "OLR", "TCWV": "TCWV",
          "MSLHF": "SLHF", "MSSHF": "SSHF", "T2M": "T2m", "BLH": "BLH"}
ROLE_COLORS = {"parent": "#0072B2", "intermediate": "#CC79A7", "candidate": "#E69F00", "competitor": "#D55E00",
               "child": "#882255", "surrogate": "#999999"}
ROLE_LABELS = {"parent": "Parent / path", "intermediate": "Intermediate", "candidate": "Paper candidate",
               "competitor": "Competitor", "child": "Child", "surrogate": "Surrogate"}
CLASS_COLORS = {"true": "#0072B2", "noise": "#999999", "response": "#CC79A7", "moving": "#56B4E9",
                "delayed": "#E69F00", "nonlinear": "#D55E00"}
CLASS_LABELS = {"true": "True parents (A + C)", "noise": "Independent noise", "response": "Response (driven by B)",
                "moving": "Moving blob (random)", "delayed": "Delayed child (lag 2)", "nonlinear": "Nonlinear (rectified)"}


def lab(v):
    return LABELS.get(v, v)


def col(v):
    return VAR_COLORS.get(v, "#777777")


def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6, "legend.frameon": False, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
    })


def panel_labels(axes, x=-0.14, y=1.04, start=0):
    for i, ax in enumerate(axes):
        ax.text(x, y, "abcdefghijklmnop"[start + i], transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="bottom", ha="left")


def save(fig, folder, name):
    d = FIG5 / folder
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return d / f"{name}.png"
