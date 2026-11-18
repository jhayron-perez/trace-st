"""Night-6 T2: Figs 4-6 (Debby, Pinatubo, PNW) on the final variable sets, pre-revision layout (17.5 cm).

Layout as the paper figures (7_Analysis_DebbyPinatubo_Fig3_v2 make_fig3_*_split, 9_AnalysisHW_Figs45_v2
plot_trace_st_new_fig4_compact_right_cbar), no extra panel:
- TraCE-ST ensemble of the paper's illustrative config (Debby 246, Pinatubo 929, PNW 205; night5/t6/<case>_configs.json
  "illustrative"); marker colour = variable selected at that step (Debby one marker every 12 h).
- Debby / Pinatubo keep the paper's distance histogram (distance to the IBTrACS track / final distance to the volcano).
- Shares by lead day for the full final set: mean over retained configs with the 95 % bootstrap CI (N6_T2
  T2_lead_shares.csv, set "final", own retained configs; step 0 excluded, also for PNW). Optional surrogate null
  (night6/analysis/T3_nullband_<case>.csv: case, variable, lead_day, p05, p50, p95; variable = the real variable whose
  surrogate it is): grey 5-95 % band per variable (--null-style band) or a marker at p95 (--null-style p95),
  pre-declared rule 6. Without the file no null is drawn.
- Density maps for every variable of the final set (role tag right of the title), pooled over the retained configs,
  step 0 excluded; PNW with the 95th-percentile contour per lead window (1-10, 11-20, 21-30 d) as in the paper.

python N6_figs456.py [--cases ...] [--nproc 32] [--test-n5 DIR] [--null-dir DIR] [--null-style p95|band]   (PBS only)
Run N6_T2.py first (lead shares). Outputs figures_v6/Fig4_debby/fig4_debby_final.{pdf,png} (Fig5_pinatubo, Fig6_pnw),
plus <name>_stats.csv (panel C numbers) and <name>_caption.txt.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N5_style as S  # noqa: E402
import N5_T2 as N5T2  # noqa: E402
import N6_T2 as N6T2  # noqa: E402
from trace_st import retention  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "Scripts_Paper"
ILLUSTRATIVE = {c: json.load(open(N6T2.N5 / "t6" / f"{c}_configs.json"))["illustrative"] for c in ("debby", "pinatubo", "pnw")}
ROLE_ORDER = ["parent", "intermediate", "competitor", "child"]
LONG = {"Tb": "Brightness temperature", "vo": "Mid-level vorticity", "precip": "Precipitation", "CAPE": "CAPE",
        "DIV200": "200-hPa divergence", "MSLP": "Mean sea-level pressure", "WS10": "10-m wind speed",
        "T050": "50-hPa temperature"}
FIGS = dict(debby=("Fig4_debby", "fig4_debby_final"), pinatubo=("Fig5_pinatubo", "fig5_pinatubo_final"),
            pnw=("Fig6_pnw", "fig6_pnw_final"))
EXTENT = dict(debby=(-50, 40, 0, 40), pinatubo=(-180, 180, -50, 80), pnw=(90, -40, 10, 80))
EVENT = dict(debby=(28.0, -46.0), pinatubo=(15.0, -160.0), pnw=(52.5, 240.0))   # (lat, lon) target point
VOLCANO = (15.13, 120.35)
PNW_WINDOWS = ((1, 11), (11, 21), (21, 31))   # steps [a, b): 1-10, 11-20, 21-30 d (step 0 excluded)
PNW_LATS, PNW_LONS = np.arange(-30, 90.0 + 2.5, 2.5), np.arange(0, 360, 2.5)   # paper density grid (2.5 deg)


def long_lab(v):
    return LONG.get(v, S.lab(v))


def order_vars(vars_, roles):
    return sorted(vars_, key=lambda v: (ROLE_ORDER.index(roles[v]), vars_.index(v)))


# ------------------------------------------------------------------ density maps
def _box_masks(c, xs, ys, half, wrap):
    if wrap:   # PNW (0-360, longitude wrap)
        lon = c[:, 1:2] % 360.0
        lo, hi = (lon - half) % 360.0, (lon + half) % 360.0
        X = np.where(lo <= hi, (xs[None] >= lo) & (xs[None] <= hi), (xs[None] >= lo) | (xs[None] <= hi))
    else:      # Debby / Pinatubo: no wrap, same inclusive masks as the run scripts
        X = (xs[None] >= c[:, 1:2] - half) & (xs[None] <= c[:, 1:2] + half)
    Y = (ys[None] >= c[:, 0:1] - half) & (ys[None] <= c[:, 0:1] + half)
    return X.astype(float), Y.astype(float)


def map_bundle(args):
    """Debby / Pinatubo: per-variable box-footprint counts (ny, nx) of one config, steps >= 1, ok members."""
    npz, box, n_vars = args
    b = np.load(npz, allow_pickle=True)
    xs, ys = np.asarray(b["xs"], float), np.asarray(b["ys"], float)
    H = np.zeros((n_vars, len(ys), len(xs)))
    for i in range(len(b["lengths"])):
        n = int(b["lengths"][i])
        if int(b["ok"][i]) != 1 or n <= 1:
            continue
        c = np.asarray(b["centers"][i, 1:n], float); p = np.asarray(b["parents"][i, 1:n], int)
        X, Y = _box_masks(c, xs, ys, float(box) / 2, False)
        for k in range(n_vars):
            m = p == k + 1
            if m.any():
                H[k] += Y[m].T @ X[m]
    return H, xs, ys


def map_pnw(args):
    """PNW: per-window, per-variable counts (n_win, n_vars, ny, nx) of one run on the 2.5-deg paper grid."""
    run_dir, n_vars = args
    half = 0.5 * float(json.load(open(Path(run_dir) / "params.json"))["box_size"])
    H = np.zeros((len(PNW_WINDOWS), n_vars, len(PNW_LATS), len(PNW_LONS)))
    for r in pickle.load(open(Path(run_dir) / "ensemble_results.pkl", "rb")):
        if not r.get("success", True):
            continue
        c = np.asarray(r["centers"], float); p = np.asarray(r["parents"], int)
        n = min(len(c), len(p))
        for iw, (a, b) in enumerate(PNW_WINDOWS):
            a, b = max(0, a), min(n, b)
            if b <= a:
                continue
            X, Y = _box_masks(c[a:b], PNW_LONS, PNW_LATS, half, True)
            for k in range(n_vars):
                m = p[a:b] == k + 1
                if m.any():
                    H[iw, k] += Y[m].T @ X[m]
    return H


def to_m180(lons, H):
    lp = ((np.asarray(lons) + 180) % 360) - 180
    o = np.argsort(lp)
    return lp[o], H[..., o]


# ------------------------------------------------------------------ data
def case_data(case, cfg, nproc):
    rel, vars_ = cfg["final"]
    tab = N5T2.load_table(case, rel, False)
    idc = "run_id" if case == "pnw" else "trial_id"
    keep = tab[retention.harmonized_case(case, tab)]
    ids = sorted(keep[idc].astype(int))
    out = dict(tab=tab, ids=ids, vars=vars_)
    with Pool(nproc) as pool:
        if case == "pnw":
            dirs = [str(Path(d) / f"run_{i:04d}") for d, i in zip(keep["_dir"], keep[idc].astype(int))]
            Hs = pool.map(map_pnw, [(d, len(vars_)) for d in dirs])
            out["H"] = np.sum(Hs, axis=0)
            ex = Path(tab.loc[tab[idc] == ILLUSTRATIVE[case], "_dir"].iloc[0]) / f"run_{ILLUSTRATIVE[case]:04d}"
            out["example"] = [dict(centers=np.asarray(r["centers"], float), parents=np.asarray(r["parents"], int))
                              for r in pickle.load(open(ex / "ensemble_results.pkl", "rb")) if r.get("success", True)]
        else:
            res = pool.map(map_bundle, [(r.bundle_npz, r.box_size, len(vars_)) for r in keep.itertuples()])
            out["H"] = np.sum([h for h, _, _ in res], axis=0)
            out["xs"], out["ys"] = res[0][1], res[0][2]
            row = tab[tab[idc] == ILLUSTRATIVE[case]].iloc[0]
            b = np.load(row.bundle_npz, allow_pickle=True)
            out["example"] = [dict(centers=np.asarray(b["centers"][i, :int(b["lengths"][i])], float),
                                   parents=np.asarray(b["parents"][i, :int(b["lengths"][i])], int))
                              for i in range(len(b["lengths"])) if int(b["ok"][i]) == 1 and int(b["lengths"][i]) > 0]
            mem = pd.concat([pd.read_csv(Path(d) / "results_members_final.csv") for d in N5T2.run_dirs(rel)],
                            ignore_index=True)
            mcol = "track_dist_km_mean" if case == "debby" else "final_dist_to_volcano_km"
            out["hist"] = mem.loc[mem["trial_id"].isin(ids) & (mem["ok"] == 1), mcol].to_numpy(float) \
                if "ok" in mem else mem.loc[mem["trial_id"].isin(ids), mcol].to_numpy(float)
    out["example_retained"] = ILLUSTRATIVE[case] in ids
    return out


def debby_track():
    import xarray as xr
    ds = xr.open_dataset(PAPER / "IBTrACS.NA.v04r01.nc")
    k = np.where(ds.sid == b"2006234N12338")[0][0]
    t = pd.to_datetime(ds.time.values[k]); la = np.asarray(ds.lat.values[k], float); lo = np.asarray(ds.lon.values[k], float)
    ok = (~pd.isna(t)) & np.isfinite(la) & np.isfinite(lo)
    t, la, lo = pd.to_datetime(t[ok]), la[ok], ((lo[ok] + 180) % 360) - 180
    back = np.asarray((pd.Timestamp("2006-08-26 00:00:00") - t) / pd.Timedelta(hours=1), float) >= 0
    return lo[back], la[back]


def debby_dem():
    import xarray as xr
    dem = xr.open_dataset(PAPER / "GMTED2010_15n015_00625deg.nc")
    x0, x1, y0, y1 = EXTENT["debby"]
    lo, la = dem["longitude"].values.astype(float), dem["latitude"].values.astype(float)
    mx, my = (lo >= x0) & (lo <= x1), (la >= y0) & (la <= y1)
    e = dem["elevation"].values.astype(float)[np.ix_(my, mx)]
    e[e <= 0] = np.nan
    return lo[mx], la[my], e


# ------------------------------------------------------------------ panels
def geo_ax(ax, extent, crs):
    import cartopy.feature as cfeature
    ax.set_extent(extent, crs=crs)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.45, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.25, alpha=0.3)


def title(ax, letter, text, role=None):
    ax.set_title(f"{letter}. {text}", loc="left", fontsize=7.5, fontweight="bold", pad=2)
    if role:
        ax.text(0.985, 0.965, role, transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=S.ROLE_COLORS[role],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8), zorder=20)


def panel_ensemble(ax, case, ens, vars_, track=None, stride=1):
    import cartopy.crs as ccrs
    geo = ccrs.PlateCarree()
    for r in ens:
        c, p = r["centers"], r["parents"]
        n = min(len(c), len(p))
        lon = c[:n, 1] % 360.0 if case == "pnw" else ((c[:n, 1] + 180) % 360) - 180
        ax.plot(lon, c[:n, 0], "-", color="0.55", lw=0.4, alpha=0.6, transform=geo, zorder=2)
        idx = np.arange(n)[::stride]
        ax.scatter(lon[idx], c[idx, 0], c=[S.col(vars_[max(0, min(len(vars_), q) - 1)]) for q in p[idx]], s=2.6,
                   edgecolor="0.15", linewidth=0.1, alpha=0.9, transform=geo, zorder=4)
    deco(ax, case, track, observed_x=True)


def deco(ax, case, track=None, observed_x=False):
    import cartopy.crs as ccrs
    geo = ccrs.PlateCarree()
    if track is not None:
        lo, la = track
        ax.plot(lo[::2], la[::2], "-", color="0.1", lw=0.7, transform=geo, zorder=6)
        if observed_x:
            ax.scatter(lo[::2], la[::2], marker="X", s=9, color="white", edgecolor="0.1", linewidth=0.35, transform=geo, zorder=7)
        else:
            ax.scatter(lo[::2], la[::2], s=2.5, color="0.1", transform=geo, zorder=7)
    if case == "pinatubo":
        ax.scatter([VOLCANO[1]], [VOLCANO[0]], marker="^", s=22, color="red", edgecolor="k", linewidth=0.4, transform=geo, zorder=10)
    la, lo = EVENT[case]
    ax.scatter([((lo + 180) % 360) - 180], [la], marker="*", s=34, color="gold", edgecolor="k", linewidth=0.4,
               transform=geo, zorder=10)


def panel_shares(ax, case, L, vars_, roles, null, null_style):
    x = L[(L["case"] == case) & (L["set"] == "final") & (L["cfgset"] == "own")]
    for v in vars_:
        g = x[x["var"] == v].sort_values("lead")
        if not len(g):
            continue
        ax.fill_between(g["lead"], g["lo"], g["hi"], color=S.col(v), alpha=0.22, lw=0, zorder=3)
        ax.plot(g["lead"], g["share"], "-", color=S.col(v), lw=1.0, marker="o", ms=1.6, zorder=4)
        if null is not None:
            q = null[null["variable"] == v].sort_values("lead_day")
            if len(q) and null_style == "band":
                ax.fill_between(q["lead_day"], q["p05"], q["p95"], color="0.5", alpha=0.12, lw=0, zorder=1)
            elif len(q):
                ax.scatter(q["lead_day"], q["p95"], marker="_", s=14, color=S.col(v), linewidths=0.9, zorder=5)
    wins = N6T2.WINDOWS[case][:3]
    for (w0, w1), c in zip(wins, S.WINDOW_COLORS):
        ax.axvspan(w0 - 0.5, w1 + 0.5, color=c, alpha=0.07, lw=0, zorder=0)
    nl = int(x["lead"].max()) if len(x) else 1
    ax.set_xlim(0.5, nl + 0.5)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, None)
    ax.set_xlabel("Lead before the event (days)")
    ax.set_ylabel("Share of trajectory density")
    ax.grid(alpha=0.2, lw=0.4)
    return int(x["n_cfg"].max()) if len(x) else 0, x


def nice_vmax(H):
    v = H[H > 0]
    q = float(np.percentile(v, 97)) if v.size else 1.0
    e = 10 ** np.floor(np.log10(q))
    return float(np.ceil(q / e) * e)


# ------------------------------------------------------------------ figure
def make_fig(case, cfg, D, L, null, null_style, root):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch
    S.set_style()
    roles = cfg["roles"]
    run_vars = D["vars"]
    vars_ = order_vars(run_vars, roles)
    nv = len(vars_)
    geo = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=180) if case == "pnw" else geo
    ncol = 3 if case == "debby" else 4
    nrow = int(np.ceil(nv / ncol))
    free = nrow * ncol - nv
    row_h = dict(debby=1.0, pinatubo=0.62, pnw=0.95)[case]
    top_h = dict(debby=1.25, pinatubo=1.25, pnw=1.45)[case]
    fig = plt.figure(figsize=(S.DOUBLE, top_h + 0.6 + nrow * (row_h + 0.22) + (0.5 if not free else 0.1)))
    gs = fig.add_gridspec(2 + nrow + (0 if free else 1), 12,
                          height_ratios=[top_h, 0.58] + [row_h] * nrow + ([0.42] if not free else []),
                          hspace=0.28, wspace=0.6, left=0.02, right=0.98, top=0.97, bottom=0.03)
    if case == "pnw":
        top = gs[0, :].subgridspec(1, 2, width_ratios=[1.3, 1], wspace=0.18)
        axA = fig.add_subplot(top[0], projection=proj); axH = None; axC = fig.add_subplot(top[1])
    else:
        top = gs[0, :].subgridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.42)
        axA = fig.add_subplot(top[0], projection=proj)
        axH = fig.add_subplot(top[1]); axC = fig.add_subplot(top[2])
    track = debby_track() if case == "debby" else None
    # A: ensemble of the illustrative config
    geo_ax(axA, EXTENT[case], proj)
    panel_ensemble(axA, case, D["example"], run_vars, track, stride=12 if case == "debby" else 1)
    ex_note = "" if D["example_retained"] else " (not retained)"
    title(axA, "A", f"TraCE-ST ensemble, config {ILLUSTRATIVE[case]}{ex_note}")
    letters = iter("ABCDEFGHIJKLMNOP"[1:])
    # B: paper histogram (Debby, Pinatubo)
    if axH is not None:
        h = D["hist"][np.isfinite(D["hist"])]
        axH.hist(h, bins=28, histtype="stepfilled", alpha=0.55, lw=0.8, color=S.WINDOW_COLORS[2 if case == "debby" else 0],
                 edgecolor="0.15")
        axH.axvline(h.mean(), color="0.15", ls="--", lw=0.8, label=f"Mean = {h.mean():.0f} km")
        axH.legend(loc="upper right", fontsize=5.5)
        axH.set_ylabel("Count (members)")
        axH.set_xlabel("Member mean distance from observed track (km)" if case == "debby"
                       else "Final distance to Mount Pinatubo (km)")
        axH.grid(alpha=0.2, lw=0.4)
        title(axH, next(letters), "Distance from official track" if case == "debby" else "Final distance to volcano")
    # C: shares by lead
    n_cfg, xstats = panel_shares(axC, case, L, vars_, roles, null, null_style)
    lC = next(letters)
    title(axC, lC, "Relative contributions by lead" + (" (step 0 excl.)" if case == "pnw" else ""))
    # variable legend (A markers, C lines, map titles), between the top row and the maps
    hv = [Line2D([0], [0], color=S.col(v), lw=1.2, marker="o", ms=3, label=f"{S.lab(v)} ({roles[v]})") for v in vars_]
    hx = [Patch(facecolor="0.5", alpha=0.3, label="95 % CI over configs")]
    if null is not None:
        hx.append(Patch(facecolor="0.5", alpha=0.15, label="Surrogate 5–95 %") if null_style == "band" else
                  Line2D([0], [0], color="0.3", marker="_", ls="none", ms=6, mew=1.0, label="Surrogate p95"))
    lax = fig.add_subplot(gs[1, :]); lax.axis("off")
    lax.legend(handles=hv + hx, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=min(len(hv) + len(hx), 5 if nv > 6 else 6), frameon=False,
               fontsize=6, handlelength=1.6, columnspacing=1.0,
               title="Variable (role): marker colour in A, line in " + lC + ", maps below", title_fontsize=6)
    # density maps
    if case == "pnw":
        Hwin = D["H"]                                   # (n_win, n_vars, ny, nx), 0-360 lons
        bg = Hwin.sum(0)
        vmax = 15000.0
        lonp, latp = PNW_LONS, PNW_LATS
    else:
        lonp, bg = to_m180(D["xs"], D["H"]); latp = D["ys"]
        vmax = nice_vmax(bg)
    dem = debby_dem() if case == "debby" else None
    mesh = None
    for j, v in enumerate(vars_):
        k = run_vars.index(v)
        r, c = divmod(j, ncol)
        w = 12 // ncol
        ax = fig.add_subplot(gs[2 + r, c * w:(c + 1) * w], projection=proj)
        geo_ax(ax, EXTENT[case], proj)
        mesh = ax.pcolormesh(lonp, latp, bg[k], transform=geo, shading="auto", cmap="BuPu", vmin=0, vmax=vmax,
                             alpha=0.55, zorder=1, rasterized=True)
        if case == "pnw":
            for iw, col in enumerate(S.WINDOW_COLORS):
                Hk = Hwin[iw, k]; pos = Hk[Hk > 0]
                if pos.size and Hk.max() >= np.percentile(pos, 95):
                    ax.contour(lonp, latp, Hk, levels=[np.percentile(pos, 95)], colors=[col], linewidths=0.9,
                               transform=geo, zorder=4 + iw)
        deco(ax, case, track)
        if case == "debby" and v == "vo":   # paper panel: Guinea-highlands circle, 800-m terrain, Hoggar
            ax.add_patch(Circle((-15, 18), radius=8, facecolor="none", edgecolor="k", lw=0.9, ls="--", transform=geo, zorder=9))
            ax.contour(*np.meshgrid(dem[0], dem[1]), dem[2], levels=[800], colors=["#653700"], linewidths=0.8,
                       alpha=0.7, transform=geo, zorder=5)
            ax.scatter([6.8], [24], marker="^", s=22, color="red", edgecolor="k", linewidth=0.4, transform=geo, zorder=10)
        if case == "pinatubo":
            ax.add_patch(Circle((VOLCANO[1], VOLCANO[0]), radius=18, facecolor="none", edgecolor="0.1", lw=0.8, ls="--",
                                transform=geo, zorder=9))
        title(ax, next(letters), long_lab(v), roles[v])
    # colour bar (+ PNW window legend): in the free map slot, else below the maps
    if free:
        r, c = divmod(nv, ncol); w = 12 // ncol
        host = fig.add_subplot(gs[2 + r, c * w:12]); host.axis("off")
        cax = host.inset_axes([0.08, 0.74, 0.84, 0.08])
    else:
        host = fig.add_subplot(gs[2 + nrow, :]); host.axis("off")
        cax = host.inset_axes([0.25, 0.72, 0.5, 0.16])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label("Integrated trajectory density (steps ≥ 1)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    cb.ax.xaxis.set_label_position("top")
    hl = [Line2D([0], [0], marker="*", ls="none", markerfacecolor="gold", markeredgecolor="k", ms=6, label="Target point")]
    if case == "pnw":
        hl = [Line2D([0], [0], color=cc, lw=1.2, label=f"{lab} (p95 contour)") for cc, lab in
              zip(S.WINDOW_COLORS, ("1–10 d", "11–20 d", "21–30 d"))] + hl
    if case == "debby":
        hl += [Line2D([0], [0], color="0.1", marker="o", ms=2, lw=0.7, label="IBTrACS track")]
    if case == "pinatubo":
        hl += [Line2D([0], [0], marker="^", ls="none", markerfacecolor="red", markeredgecolor="k", ms=4, label="Mt Pinatubo")]
    if free:
        host.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.5, 0.42), ncol=2, fontsize=5.5, frameon=False)
    else:
        host.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.5, 0.3), ncol=len(hl), fontsize=5.5, frameon=False)
    folder, name = FIGS[case]
    d = root / folder; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    xstats.to_csv(d / f"{name}_stats.csv", index=False)
    nul = ("no surrogate null drawn (T3 null band not available yet)" if null is None else
           "grey band = 5–95 % range of the surrogate share of each variable per lead day (T3)" if null_style == "band" else
           "short coloured dashes = 95th percentile of the surrogate share of each variable per lead day (T3; "
           "markers chosen over bands for readability, rule 6)")
    cap = (f"{cfg['title']}, final variable set ({nv} variables), linear rule, harmonized retention: {D['n_ret']} of "
           f"{D['n_done']} configs retained. A: ensemble of the illustrative config {ILLUSTRATIVE[case]}{ex_note}; marker "
           f"colour = variable selected at that step{' (one marker every 12 h)' if case == 'debby' else ''}. "
           f"{lC}: share of trajectory density per variable and lead day, mean over the {n_cfg} retained configs with the "
           f"95 % bootstrap CI (10,000 resamples of configs, seed 0); step 0 (the start box) excluded"
           f"{' (PNW: its parent is always Z500)' if case == 'pnw' else ''}; {nul}. Maps: integrated trajectory density "
           f"of the steps where each variable was selected, pooled over the retained configs (step 0 excluded)"
           f"{'; contours = 95th percentile of each lead window' if case == 'pnw' else ''}; role in the panel title.")
    (d / f"{name}_caption.txt").write_text(cap + "\n")
    print(cap)
    return d / f"{name}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["debby", "pinatubo", "pnw"])
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--test-n5", default=None, help="night-5 extended sets; reads/writes this dir")
    ap.add_argument("--null-dir", default=None, help="dir with T3_nullband_<case>.csv (default: the analysis dir)")
    ap.add_argument("--null-style", choices=["p95", "band"], default="p95")
    a = ap.parse_args()
    out, root = N6T2.OUT, N6T2.FIG6
    if a.test_n5:
        out = root = Path(a.test_n5)
        for c, v in N6T2.N5_EXT.items():
            N6T2.CASES[c]["final"] = v
    L = pd.read_csv(out / "T2_lead_shares.csv")
    for case in a.cases:
        cfg = N6T2.CASES[case]
        if not (L["case"] == case).any():
            print("no lead shares yet (run N6_T2.py):", case); continue
        D = case_data(case, cfg, a.nproc)
        D["n_ret"], D["n_done"] = len(D["ids"]), len(D["tab"])
        nf = Path(a.null_dir or out) / f"T3_nullband_{case}.csv"
        null = pd.read_csv(nf) if nf.exists() else None
        if null is not None:
            null = null[null["case"] == case]
            print("null band", nf, len(null))
        print("fig", make_fig(case, cfg, D, L, null, a.null_style, root), flush=True)


if __name__ == "__main__":
    main()
