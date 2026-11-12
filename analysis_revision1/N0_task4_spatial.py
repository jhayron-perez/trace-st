"""Night-3 N0 task 4 (H3): spatial reproducibility of variable-attributed trajectory density.

Density map of variable v in window w for one config = box-footprint counts (same footprint as
rev1_common.step_density_* / the run scripts: every grid cell inside the box of size box_size centred on the
trajectory point, inclusive edges; PNW with longitude wrap, Debby without), summed over ok members and
over the steps s in the window whose parent is v. Step 0 excluded. Windows: PNW steps 1-10/11-20/21-30;
Debby lead days 1-4 / 5-15 = hourly steps 1-96 / 97-360.
Grids: PNW 2 deg (rev1_common.PNW_LATS/LONS); Debby bundle xs/ys (0.25 deg).
Statistics (per role real/twin, per window):
 (a) across-config mean pairwise Pearson r of config maps, per run, then mean over runs;
 (b) within-config mean pairwise r across members (members with non-empty maps), mean over configs;
 (c) entropy of the pooled map (configs normalised to sum 1 then averaged): H (nats), exp(H), H/log(N_cells);
 (d) same-config r between runs (twin_a vs twin_b, real_a vs real_b, real vs twin same run), mean over configs
     and year pairs, plus r between pooled maps.
Debby also: fraction of density mass on terrain > 1000 m (GMTED2010 0.0625 deg, Scripts_Paper/
GMTED2010_15n015_00625deg.nc, variable 'elevation', nearest neighbour to the Debby grid).
python N0_task4_spatial.py {pnw,debby} [--nproc N]   (PBS only)
"""
import argparse
import itertools
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import A4_twins as A  # noqa: E402
import rev1_common as C  # noqa: E402

R1 = A.R1
OUT = R1 / "night3/N0/task4"
REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- box index ranges
def ranges_nowrap(c, coord_asc, half):
    i0 = np.searchsorted(coord_asc, c - half, "left")
    i1 = np.searchsorted(coord_asc, c + half, "right")
    return i0, i1


def add_boxes(M, y0, y1, xr):
    """M: (ny+1, nx+1) difference array; xr: list of (x0, x1) arrays (wrap pieces)."""
    for x0, x1 in xr:
        ok = (y1 > y0) & (x1 > x0)
        np.add.at(M, (y0[ok], x0[ok]), 1); np.add.at(M, (y0[ok], x1[ok]), -1)
        np.add.at(M, (y1[ok], x0[ok]), -1); np.add.at(M, (y1[ok], x1[ok]), 1)


def finish(M):
    return np.cumsum(np.cumsum(M, 0), 1)[:-1, :-1]


def member_maps(cen, par, n, roles, wins, grid, box, wrap):
    """Returns dict (role_name, win) -> map for one member."""
    lat_asc, lon_asc, flip_y = grid
    half = box / 2.0
    out = {}
    for wn, (s0, s1) in wins.items():
        ss = np.arange(s0, min(s1, n - 1) + 1)
        if ss.size == 0:
            continue
        for rn, idx in roles.items():
            sel = ss[par[ss] == idx]
            M = np.zeros((len(lat_asc) + 1, len(lon_asc) + 1))
            if sel.size:
                la = cen[sel, 0].astype(float); lo = cen[sel, 1].astype(float)
                y0, y1 = ranges_nowrap(la, lat_asc, half)
                if wrap:
                    a = (lo - half) % 360.0; b = (lo + half) % 360.0
                    i0 = np.searchsorted(lon_asc, a, "left"); i1 = np.searchsorted(lon_asc, b, "right")
                    nw = a <= b
                    pieces = [(np.where(nw, i0, i0), np.where(nw, i1, len(lon_asc))),
                              (np.where(nw, 0, 0), np.where(nw, 0, i1))]
                else:
                    x0, x1 = ranges_nowrap(lo, lon_asc, half); pieces = [(x0, x1)]
                add_boxes(M, y0, y1, pieces)
            m = finish(M)
            if flip_y:
                m = m[::-1]
            out[(rn, wn)] = m.astype(np.float32)
    return out


def mean_pair_r(X):
    X = np.asarray(X, float).reshape(len(X), -1)
    X = X[X.std(1) > 0]
    if len(X) < 2:
        return np.nan, len(X)
    R = np.corrcoef(X)
    iu = np.triu_indices(len(X), 1)
    return float(np.nanmean(R[iu])), len(X)


def r2(a, b):
    a = a.ravel(); b = b.ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- per-config workers
PNW_WINS = {"w1-10": (1, 10), "w11-20": (11, 20), "w21-30": (21, 30)}
DEB_WINS = {"w1-4": (1, 96), "w5-15": (97, 360)}


def pnw_one(args):
    run_dir, roles, box = args
    res = pickle.load(open(Path(run_dir) / "ensemble_results.pkl", "rb"))
    grid = (C.PNW_LATS, C.PNW_LONS, False)
    cfgmap, mems = {}, {}
    for r in res:
        if not r.get("success", True):
            continue
        c = np.asarray(r["centers"], float); p = np.asarray(r["parents"], int)
        n = min(len(c), len(p))
        if n < 2:
            continue
        mm = member_maps(c, p, n, roles, PNW_WINS, grid, box, wrap=True)
        for k, v in mm.items():
            cfgmap[k] = cfgmap.get(k, 0) + v
            mems.setdefault(k, []).append(v)
    mr = {k: mean_pair_r(v) for k, v in mems.items()}
    return cfgmap, mr


def debby_one(args):
    npz, roles, box = args
    b = np.load(npz, allow_pickle=True)
    xs = np.asarray(b["xs"], float); ys = np.asarray(b["ys"], float)
    flip = ys[0] > ys[-1]
    grid = (ys[::-1] if flip else ys, xs, flip)
    Cn, P, L, ok = b["centers"], b["parents"], b["lengths"], b["ok"]
    cfgmap, mems = {}, {}
    for i in range(len(L)):
        n = int(L[i])
        if int(ok[i]) != 1 or n < 2:
            continue
        mm = member_maps(np.asarray(Cn[i, :n], float), np.asarray(P[i, :n], int), n, roles, DEB_WINS, grid, box,
                         wrap=False)
        for k, v in mm.items():
            cfgmap[k] = cfgmap.get(k, 0) + v
            mems.setdefault(k, []).append(v)
    mr = {k: mean_pair_r(v) for k, v in mems.items()}
    return cfgmap, mr


# ---------------------------------------------------------------- driver
def build_runs(case):
    """list of dict(name, group, kind(base|twin), tasks=[(cfg, args)])."""
    runs = []
    if case == "pnw":
        ids = A.retained_ids("pnw")
        import json
        box = {i: float(json.load(open(R1 / f"T6r_pnw/run_{i:04d}/params.json"))["box_size"]) for i in ids}
        for stem, nm in A.PNW_STEM.items():
            k = A.PNW6.index(nm) + 1
            runs.append(dict(name=f"{nm}_base", group=nm, year="base", kind="base",
                             tasks=[(i, (str(R1 / f"T6r_pnw/run_{i:04d}"), {"real": k}, box[i])) for i in ids
                                    if (R1 / f"T6r_pnw/run_{i:04d}/ensemble_results.pkl").exists()]))
            for y in A.A1_YEARS:
                d = R1 / "A1_pnw_twin" / f"{stem}_{y}"
                runs.append(dict(name=f"{nm}_{y}", group=nm, year=y, kind="twin",
                                 tasks=[(i, (str(d / f"run_{i:04d}"), {"real": k, "twin": 7}, box[i])) for i in ids
                                        if (d / f"run_{i:04d}/ensemble_results.pkl").exists()]))
    else:
        ids = A.retained_ids("debby")
        def tasks_of(d, roles):
            j = pd.read_csv(d / "results_jobs_final.csv")
            j = j[j.trial_id.isin(ids) & j.bundle_npz.notna()]
            return [(int(r.trial_id), (r.bundle_npz, roles, float(r.box_size))) for r in j.itertuples()]
        runs.append(dict(name="vo_base", group="vo", year="base", kind="base",
                         tasks=tasks_of(R1 / "T4r_debby_vorebuilt", {"real": 2, "Tb": 1, "precip": 3})))
        for y in A.A2_YEARS:
            runs.append(dict(name=f"vo_{y}", group="vo", year=y, kind="twin",
                             tasks=tasks_of(R1 / "A2_debby_twin" / f"vo_{y}", {"real": 2, "twin": 4, "Tb": 1,
                                                                                "precip": 3})))
    return runs


def norm(m):
    s = m.sum()
    return m / s if s > 0 else None


def entropy(p):
    q = p[p > 0]
    H = float(-(q * np.log(q)).sum())
    return H, float(np.exp(H)), H / np.log(p.size)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("case"); ap.add_argument("--nproc", type=int, default=60)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = build_runs(a.case)
    fn = pnw_one if a.case == "pnw" else debby_one
    wins = list(PNW_WINS if a.case == "pnw" else DEB_WINS)
    alltasks = [(ri, cfg, args) for ri, r in enumerate(runs) for cfg, args in r["tasks"]]
    print("tasks", len(alltasks), flush=True)
    with Pool(a.nproc) as pool:
        res = pool.map(fn, [t[2] for t in alltasks], chunksize=2)
    maps = {}   # (run idx, role, win) -> {cfg: map}
    memr = []
    for (ri, cfg, _), (cm, mr) in zip(alltasks, res):
        for (role, w), m in cm.items():
            maps.setdefault((ri, role, w), {})[cfg] = m
        for (role, w), (rr, nm) in mr.items():
            memr.append(dict(run=runs[ri]["name"], group=runs[ri]["group"], year=runs[ri]["year"], cfg=cfg,
                             role=role, window=w, member_r=rr, n_members_nonempty=nm))
    MR = pd.DataFrame(memr)
    MR.to_csv(OUT / f"task4_{a.case}_member_r.csv.gz", index=False)
    elev = None
    if a.case == "debby":
        import xarray as xr
        g = xr.open_dataset(REPO / "Scripts_Paper/GMTED2010_15n015_00625deg.nc")
        glat = g["latitude"].values.astype(float); glon = g["longitude"].values.astype(float)
        b = np.load(runs[0]["tasks"][0][1][0], allow_pickle=True)
        xs = np.asarray(b["xs"], float); ys = np.asarray(b["ys"], float)
        xl = ((xs + 180) % 360) - 180
        def nn(src, tgt):
            o = np.argsort(src); s = src[o]
            i = np.clip(np.searchsorted(s, tgt), 1, len(s) - 1)
            i = np.where(np.abs(s[i - 1] - tgt) <= np.abs(s[i] - tgt), i - 1, i)
            return o[i]
        E = g["elevation"].values
        elev = E[np.ix_(nn(glat, ys), nn(glon, xl))].astype(float)
        print("Debby grid lat", ys.min(), ys.max(), "lon(0-360)", xs.min(), xs.max(), "n", len(ys), len(xs),
              "elev max", elev.max(), "area frac >1000 m", float((elev > 1000).mean()), flush=True)
        np.save(OUT / "task4_debby_elev_on_grid.npy", elev)
    rows, pair_rows, pooled = [], [], {}
    for ri, r in enumerate(runs):
        for (rj, role, w), d in maps.items():
            if rj != ri:
                continue
            cfgs = sorted(d)
            X = np.stack([d[c] for c in cfgs])
            ra, nn_ = mean_pair_r(X)
            P = [norm(d[c]) for c in cfgs]
            P = [p for p in P if p is not None]
            pm = np.mean(P, 0) if P else None
            pooled[(ri, role, w)] = pm
            H = entropy(pm) if pm is not None else (np.nan,) * 3
            mr = MR[(MR.run == r["name"]) & (MR.role == role) & (MR.window == w)]["member_r"]
            row = dict(case=a.case, run=r["name"], group=r["group"], year=r["year"], role=role, window=w,
                       n_cfg=len(cfgs), n_cfg_nonempty=nn_, across_cfg_r=ra, within_cfg_member_r=mr.mean(),
                       H=H[0], expH=H[1], H_norm=H[2],
                       mean_cfg_mass_share=np.nan)
            if elev is not None:
                fr = [float((p * (elev > 1000)).sum()) for p in P]
                row.update(frac_mass_gt1000m_cfgmean=float(np.mean(fr)) if fr else np.nan,
                           frac_mass_gt1000m_pooled=float((pm * (elev > 1000)).sum()) if pm is not None else np.nan)
            rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / f"task4_{a.case}_per_run.csv", index=False)
    # (d) cross-run same-config correlations within a group
    for grp in sorted({r["group"] for r in runs}):
        ris = [i for i, r in enumerate(runs) if r["group"] == grp]
        tw = [i for i in ris if runs[i]["kind"] == "twin"]
        bs = [i for i in ris if runs[i]["kind"] == "base"]
        for w in wins:
            def same_cfg(ka, kb):
                da, db = maps.get(ka, {}), maps.get(kb, {})
                cs = sorted(set(da) & set(db))
                v = [r2(da[c], db[c]) for c in cs]
                return np.nanmean(v) if len(v) else np.nan, len(cs)
            def pooled_r(ka, kb):
                pa, pb = pooled.get(ka), pooled.get(kb)
                return r2(pa, pb) if pa is not None and pb is not None else np.nan
            comps = []
            for i, j in itertools.combinations(tw, 2):
                comps.append(("twin_a_vs_twin_b", (i, "twin", w), (j, "twin", w)))
                comps.append(("real_a_vs_real_b", (i, "real", w), (j, "real", w)))
            for i in tw:
                comps.append(("real_vs_twin_same_run", (i, "real", w), (i, "twin", w)))
                for j in bs:
                    comps.append(("real_twinrun_vs_real_base", (i, "real", w), (j, "real", w)))
                    comps.append(("twin_vs_real_base", (i, "twin", w), (j, "real", w)))
            for name, ka, kb in comps:
                m, n = same_cfg(ka, kb)
                pair_rows.append(dict(case=a.case, group=grp, window=w, comparison=name, run_a=runs[ka[0]]["name"],
                                      run_b=runs[kb[0]]["name"], same_cfg_r=m, n_cfg=n, pooled_r=pooled_r(ka, kb)))
    PR = pd.DataFrame(pair_rows)
    PR.to_csv(OUT / f"task4_{a.case}_pairs.csv", index=False)
    comp = PR.groupby(["group", "window", "comparison"]).agg(same_cfg_r=("same_cfg_r", "mean"),
                                                             pooled_r=("pooled_r", "mean"),
                                                             n_pairs=("run_a", "size")).reset_index()
    comp.to_csv(OUT / f"task4_{a.case}_pairs_summary.csv", index=False)
    summ = S.assign(kind=np.where(S.year.astype(str) == "base", "base", "twin_runs")).groupby(
        ["group", "kind", "role", "window"]).mean(numeric_only=True).reset_index()
    summ.to_csv(OUT / f"task4_{a.case}_summary.csv", index=False)
    np.savez_compressed(OUT / f"task4_{a.case}_pooled_maps.npz",
                        **{f"{runs[k[0]]['name']}__{k[1]}__{k[2]}": v for k, v in pooled.items() if v is not None})
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    print(comp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
