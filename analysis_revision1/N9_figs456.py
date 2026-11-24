"""Night-9 N3: final Figs 4-6 (Debby, Pinatubo, PNW), two framings, pre-revision layout (17.5 cm), no night-7 band.

Framing A (decoy runs, "not rank 1" applied): final set + typical-field decoy pool (night8/t1b for PNW and Pinatubo,
night9/n1 for Debby), harmonized retention, then drop a configuration when the decoy density exceeds every real variable's
(flags in night8/analysis/T1b_*_configs.csv; Debby night9/analysis/N1_configs.csv). Panel C: share per real variable and lead
day renormalized among the real variables (mean over kept configs, 95 % bootstrap CI over configs, seed 0, step 0 excluded)
plus the decoy share of all candidates as a thin grey line (diagnostic). Maps: real variables only, kept configs.
Framing B (no criterion): night-6 final-set runs (night6/t2/*_final), panel C with the 95 % CIs only (same as figures_v6
without the null markers).
Code: derived from N6_figs456.py (layout, maps, helpers imported from it); differences: no null/band, framing A decoy data and
diagnostic line, the PNW map legend moved off panel I, per-step markers in panel A for both framings.
python N9_figs456.py --cases pnw pinatubo debby --framings A B --nproc 32   (PBS only)
Outputs figures_v9/Fig{4_debby,5_pinatubo,6_pnw}/fig*_final_{A,B}.{pdf,png}, *_stats.csv, *_caption.txt
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
import N6_figs456 as F  # noqa: E402
import N8_T1b as B  # noqa: E402
from A4_twins import boot_ci  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = N6T2.R1
FIG9 = R1 / "figures_v9"
OUTA = R1 / "night9" / "analysis"
S.VAR_COLORS["DECOY"] = "#BBBBBB"
FIGS = dict(debby=("Fig4_debby", "fig4_debby_final"), pinatubo=("Fig5_pinatubo", "fig5_pinatubo_final"),
            pnw=("Fig6_pnw", "fig6_pnw_final"))
REAL = dict(debby=N6T2.CASES["debby"]["final"][1], pinatubo=N6T2.CASES["pinatubo"]["final"][1],
            pnw=N6T2.CASES["pnw"]["final"][1])
T1B = R1 / "night8" / "t1b"


# ------------------------------------------------------------------ framing A data
def _pnw_member_D(args):
    rd, nv = args
    i, D, ok, hp = B.member_dens_pnw((rd, nv))
    return i, D.sum(0)


def _pin_member_D(args):
    npz, box, nv, tid = args
    i, D, ok, hp = B.member_dens_pin((npz, box, nv, tid, {}))
    return i, D.sum(0)


def case_data_A(case, nproc):
    """Decoy-run data of framing A: kept configs (harmonized AND not rank 1), per-config density incl. the decoy."""
    real = REAL[case]
    nv = len(real) + 1
    idc = "run_id" if case == "pnw" else "trial_id"
    if case == "pnw":
        tabs = []
        for d in sorted(T1B.glob("pnw_p[0-9]")):
            tabs.append(pd.read_csv(d / "hyperparam_search_summary.csv").assign(_dir=str(d)))
        tab = pd.concat(tabs, ignore_index=True)
        cf = pd.read_csv(R1 / "night8/analysis/T1b_pnw_configs.csv").set_index("id")
    elif case == "pinatubo":
        d = T1B / "pin"
        tab = pd.read_csv(d / "results_jobs_final.csv").assign(_dir=str(d))
        cf = pd.read_csv(R1 / "night8/analysis/T1b_pinatubo_configs.csv").set_index("id")
    else:   # debby: night 9 N1 (filled in by the N1 analysis)
        parts = sorted((R1 / "night9" / "n1").glob("deb_q[0-9]"))
        tabs = [pd.read_csv(p / ("results_jobs_final.csv" if (p / "results_jobs_final.csv").exists() else "results_jobs_snapshot.csv")).assign(_dir=str(p)) for p in parts]
        tab = pd.concat(tabs, ignore_index=True)
        cf = pd.read_csv(OUTA / "N1_debby_all.csv").set_index("id")
    keep_h = tab[retention.harmonized_case(case, tab)]
    ids_h = sorted(keep_h[idc].astype(int))
    assert set(ids_h) <= set(cf.index), "retained ids missing in the flag table"
    ids = [i for i in ids_h if not bool(cf.loc[i, "R1_drop"])]
    kp = keep_h[keep_h[idc].astype(int).isin(ids)]
    out = dict(tab=tab, ids=ids, ids_h=ids_h, vars=real)
    with Pool(nproc) as pool:
        if case == "pnw":
            dirs = [str(Path(d) / f"run_{i:04d}") for d, i in zip(kp["_dir"], kp[idc].astype(int))]
            Dd = dict(pool.map(_pnw_member_D, [(d, nv) for d in dirs]))
            Hs = pool.map(F.map_pnw, [(d, len(real)) for d in dirs])
            out["H"] = np.sum(Hs, axis=0)
        else:
            Dd = dict(pool.map(_pin_member_D, [(r.bundle_npz, r.box_size, nv, int(getattr(r, idc))) for r in kp.itertuples()])) \
                if case == "pinatubo" else {}
            if case == "debby":
                import N6_T3T4 as _T
                Dd = {}
                for p in parts:
                    Dp_, _ = _T.run_densities("debby", p, nv, True, nproc)
                    Dd.update({int(k): np.asarray(v, float) for k, v in Dp_.items()})
            res = pool.map(F.map_bundle, [(r.bundle_npz, r.box_size, len(real)) for r in kp.itertuples()])
            out["H"] = np.sum([h for h, _, _ in res], axis=0)
            out["xs"], out["ys"] = res[0][1], res[0][2]
    out["D_all"] = {i: np.asarray(Dd[i], float) for i in ids}
    # illustrative config (paper ids 246/929/205): the config itself if it is kept, else the kept config with the nearest id
    ill = F.ILLUSTRATIVE[case]
    ex_id = ill if ill in ids else min(ids, key=lambda i: abs(i - ill))
    out["example_id"], out["example_retained"] = ex_id, ex_id == ill
    row = tab[tab[idc].astype(int) == ex_id].iloc[0]
    if case == "pnw":
        rd = Path(row["_dir"]) / f"run_{ex_id:04d}"
        out["example"] = [dict(centers=np.asarray(r["centers"], float), parents=np.asarray(r["parents"], int))
                          for r in pickle.load(open(rd / "ensemble_results.pkl", "rb")) if r.get("success", True)]
    else:
        b = np.load(row.bundle_npz, allow_pickle=True)
        out["example"] = [dict(centers=np.asarray(b["centers"][i, :int(b["lengths"][i])], float),
                               parents=np.asarray(b["parents"][i, :int(b["lengths"][i])], int))
                          for i in range(len(b["lengths"])) if int(b["ok"][i]) == 1 and int(b["lengths"][i]) > 0]
        mem = pd.concat([pd.read_csv(Path(d) / ("results_members_final.csv" if (Path(d) / "results_members_final.csv").exists() else "results_members_snapshot.csv")) for d in sorted(set(tab["_dir"]))], ignore_index=True)
        mcol = "track_dist_km_mean" if case == "debby" else "final_dist_to_volcano_km"
        sel = mem["trial_id"].isin(ids) & ((mem["ok"] == 1) if "ok" in mem else True)
        out["hist"] = mem.loc[sel, mcol].to_numpy(float)
    return out


def lead_rows_A(case, D, roles):
    """Panel-C table of framing A (real variables renormalized) and the decoy share per lead."""
    real = D["vars"]
    nv = len(real)
    ids = D["ids"]
    Dreal = {i: D["D_all"][i][:, :nv] for i in ids}
    N5T2.WINDOWS = N6T2.WINDOWS
    W, L, K = N5T2.stats_rows(case, "final", real, Dreal, ids, roles, "own")
    dec = []
    nl = max(d.shape[0] for d in D["D_all"].values())
    for lead in range(1, nl):
        x = np.array([d[lead, nv] / d[lead].sum() for d in D["D_all"].values() if d[lead].sum() > 0])
        m, lo, hi = boot_ci(x)
        dec.append(dict(lead=lead, share=m, lo=lo, hi=hi, n_cfg=len(x)))
    return pd.DataFrame(W), pd.DataFrame(L), pd.DataFrame(dec)


# ------------------------------------------------------------------ figure
def make_fig(case, framing, D, L, root, decoy=None):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch
    S.set_style()
    cfg = N6T2.CASES[case]
    roles = cfg["roles"]
    run_vars = D["vars"]
    vars_ = F.order_vars(run_vars, roles)
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
                          height_ratios=[top_h, 0.74] + [row_h] * nrow + ([0.42] if not free else []),
                          hspace=0.28, wspace=0.6, left=0.02, right=0.98, top=0.97, bottom=0.03)
    if case == "pnw":
        top = gs[0, :].subgridspec(1, 2, width_ratios=[1.3, 1], wspace=0.18)
        axA = fig.add_subplot(top[0], projection=proj); axH = None; axC = fig.add_subplot(top[1])
    else:
        top = gs[0, :].subgridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.42)
        axA = fig.add_subplot(top[0], projection=proj)
        axH = fig.add_subplot(top[1]); axC = fig.add_subplot(top[2])
    track = F.debby_track() if case == "debby" else None
    F.geo_ax(axA, F.EXTENT[case], proj)
    mark_vars = run_vars + (["DECOY"] if framing == "A" else [])
    F.panel_ensemble(axA, case, D["example"], mark_vars, track, stride=12 if case == "debby" else 1)
    eid = D.get("example_id", F.ILLUSTRATIVE[case])
    ex_note = "" if D["example_retained"] else (f" (config {F.ILLUSTRATIVE[case]} not kept; nearest kept shown)" if framing == "A" else " (not retained)")
    F.title(axA, "A", f"TraCE-ST ensemble, config {eid}" + ("" if D["example_retained"] else "*"))
    letters = iter("ABCDEFGHIJKLMNOP"[1:])
    if axH is not None:
        h = D["hist"][np.isfinite(D["hist"])]
        axH.hist(h, bins=28, histtype="stepfilled", alpha=0.55, lw=0.8, color=S.WINDOW_COLORS[2 if case == "debby" else 0], edgecolor="0.15")
        axH.axvline(h.mean(), color="0.15", ls="--", lw=0.8, label=f"Mean = {h.mean():.0f} km")
        axH.legend(loc="upper right", fontsize=5.5)
        axH.set_ylabel("Count (members)")
        axH.set_xlabel("Member mean distance from observed track (km)" if case == "debby" else "Final distance to Mount Pinatubo (km)")
        axH.grid(alpha=0.2, lw=0.4)
        F.title(axH, next(letters), "Distance from official track" if case == "debby" else "Final distance to volcano")
    n_cfg, xstats = F.panel_shares(axC, case, L, vars_, roles, None, "p95", None)
    if decoy is not None and len(decoy):
        axC.plot(decoy["lead"], decoy["share"], "--", color="0.35", lw=0.6, zorder=3.5)
        axC.set_ylim(0, max(axC.get_ylim()[1], float(decoy["share"].max()) * 1.08))
    lC = next(letters)
    F.title(axC, lC, "Relative contributions by lead" + (" (step 0 excl.)" if case == "pnw" else ""))
    hv = [Line2D([0], [0], color=S.col(v), lw=1.2, marker="o", ms=3, label=f"{S.lab(v)} ({roles[v]})") for v in vars_]
    hx = [Patch(facecolor="0.5", alpha=0.3, label="95 % CI over configs")]
    if decoy is not None:
        hx.append(Line2D([0], [0], color="0.35", lw=0.8, ls="--", label="Decoy share (all candidates)"))
    lax = fig.add_subplot(gs[1, :]); lax.axis("off")
    lax.legend(handles=hv + hx, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=min(len(hv) + len(hx), 5 if nv > 6 else 6),
               frameon=False, fontsize=6, handlelength=1.6, columnspacing=1.0,
               title="Variable (role): marker colour in A, line in " + lC + ", maps below"
               + ("; grey markers = decoy" if framing == "A" else ""), title_fontsize=6)
    if case == "pnw":
        Hwin = D["H"]; bg = Hwin.sum(0); vmax = 15000.0; lonp, latp = F.PNW_LONS, F.PNW_LATS
    else:
        lonp, bg = F.to_m180(D["xs"], D["H"]); latp = D["ys"]; vmax = F.nice_vmax(bg)
    dem = F.debby_dem() if case == "debby" else None
    mesh = None
    for j, v in enumerate(vars_):
        k = run_vars.index(v)
        r, c = divmod(j, ncol); w = 12 // ncol
        ax = fig.add_subplot(gs[2 + r, c * w:(c + 1) * w], projection=proj)
        F.geo_ax(ax, F.EXTENT[case], proj)
        mesh = ax.pcolormesh(lonp, latp, bg[k], transform=geo, shading="auto", cmap="BuPu", vmin=0, vmax=vmax, alpha=0.55, zorder=1, rasterized=True)
        if case == "pnw":
            for iw, col in enumerate(S.WINDOW_COLORS):
                Hk = Hwin[iw, k]; pos = Hk[Hk > 0]
                if pos.size and Hk.max() >= np.percentile(pos, 95):
                    ax.contour(lonp, latp, Hk, levels=[np.percentile(pos, 95)], colors=[col], linewidths=0.9, transform=geo, zorder=4 + iw)
        F.deco(ax, case, track)
        if case == "debby" and v == "vo":
            ax.add_patch(Circle((-15, 18), radius=8, facecolor="none", edgecolor="k", lw=0.9, ls="--", transform=geo, zorder=9))
            ax.contour(*np.meshgrid(dem[0], dem[1]), dem[2], levels=[800], colors=["#653700"], linewidths=0.8, alpha=0.7, transform=geo, zorder=5)
            ax.scatter([6.8], [24], marker="^", s=22, color="red", edgecolor="k", linewidth=0.4, transform=geo, zorder=10)
        if case == "pinatubo":
            ax.add_patch(Circle((F.VOLCANO[1], F.VOLCANO[0]), radius=18, facecolor="none", edgecolor="0.1", lw=0.8, ls="--", transform=geo, zorder=9))
        F.title(ax, next(letters), F.long_lab(v), roles[v])
    hl = [Line2D([0], [0], marker="*", ls="none", markerfacecolor="gold", markeredgecolor="k", ms=6, label="Target point")]
    if case == "pnw":
        hl = [Line2D([0], [0], color=cc, lw=1.2, label=f"{lab} (p95 contour)") for cc, lab in zip(S.WINDOW_COLORS, ("1–10 d", "11–20 d", "21–30 d"))] + hl
    if case == "debby":
        hl += [Line2D([0], [0], color="0.1", marker="o", ms=2, lw=0.7, label="IBTrACS track")]
    if case == "pinatubo":
        hl += [Line2D([0], [0], marker="^", ls="none", markerfacecolor="red", markeredgecolor="k", ms=4, label="Mt Pinatubo")]
    if free:
        r, c = divmod(nv, ncol); w = 12 // ncol
        host = fig.add_subplot(gs[2 + r, c * w:12]); host.axis("off")
        cax = host.inset_axes([0.08, 0.80, 0.84, 0.07])
        # legend left-aligned inside the free slot (one column): it no longer reaches the map to its left (panel I in Fig. 6)
        host.legend(handles=hl, loc="upper left", bbox_to_anchor=(0.0, 0.62 if case == "pnw" else 0.36), ncol=1 if case == "pnw" else 2, fontsize=5.5, frameon=False,
                    handlelength=1.6, labelspacing=0.35, borderaxespad=0.0)
    else:
        host = fig.add_subplot(gs[2 + nrow, :]); host.axis("off")
        cax = host.inset_axes([0.25, 0.72, 0.5, 0.16])
        host.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.5, 0.3), ncol=len(hl), fontsize=5.5, frameon=False)
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label("Integrated trajectory density (steps ≥ 1)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    cb.ax.xaxis.set_label_position("top")
    folder, name = FIGS[case]
    name = f"{name}_{framing}"
    d = root / folder; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    xstats.to_csv(d / f"{name}_stats.csv", index=False)
    if decoy is not None:
        decoy.to_csv(d / f"{name}_decoy_share.csv", index=False)
    if framing == "A":
        desc = (f"framing A: final set + typical-field decoy pool, harmonized retention ({len(D['ids_h'])} configs) then 'not rank 1' "
                f"({len(D['ids'])} kept, {len(D['ids_h']) - len(D['ids'])} removed); panel C real-variable shares renormalized among the real variables, "
                f"thin grey line = decoy share of all candidates")
    else:
        desc = f"framing B: night-6 final-set runs without decoy, harmonized retention ({D['n_ret']} of {D['n_done']} configs)"
    cap = (f"{cfg['title']}, final variable set ({nv} variables), linear rule; {desc}. A: ensemble of config {eid}{ex_note}; marker colour = "
           f"variable selected at that step{' (one marker every 12 h)' if case == 'debby' else ''}. {lC}: share of trajectory density per variable "
           f"and lead day, mean over {n_cfg} configs with the 95 % bootstrap CI (10,000 resamples of configs, seed 0); step 0 excluded. Maps: integrated "
           f"trajectory density of the steps where each variable was selected, pooled over the configs"
           f"{'; contours = 95th percentile of each lead window' if case == 'pnw' else ''}.")
    (d / f"{name}_caption.txt").write_text(cap + "\n")
    print(cap, flush=True)
    return d / f"{name}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["pnw", "pinatubo", "debby"])
    ap.add_argument("--framings", nargs="+", default=["A", "B"])
    ap.add_argument("--nproc", type=int, default=32)
    a = ap.parse_args()
    Lb = pd.read_csv(N6T2.OUT / "T2_lead_shares.csv")
    for case in a.cases:
        cfg = N6T2.CASES[case]
        for fr in a.framings:
            if fr == "B":
                D = F.case_data(case, cfg, a.nproc)
                D["n_ret"], D["n_done"] = len(D["ids"]), len(D["tab"])
                out = make_fig(case, "B", D, Lb, FIG9)
            else:
                try:
                    D = case_data_A(case, a.nproc)
                except (FileNotFoundError, ValueError) as e:
                    print("framing A not available yet for", case, ":", e, flush=True)
                    continue
                W, L, dec = lead_rows_A(case, D, cfg["roles"])
                folder, name = FIGS[case]
                (FIG9 / folder).mkdir(parents=True, exist_ok=True)
                W.to_csv(FIG9 / folder / f"{name}_A_window_shares.csv", index=False)
                out = make_fig(case, "A", D, L, FIG9, decoy=dec)
            print("fig", out, flush=True)


if __name__ == "__main__":
    main()
