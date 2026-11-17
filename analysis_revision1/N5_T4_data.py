"""Night-5 T4 (R1 Major 2, parameter guidelines), stage 1: one row per configuration for every case,
plus the objective scale diagnostics (feature displacement per step, feature size).

Cases and sources
- T1 univariate (T1r_univariate, method granger = Elastic-Net), T2-EN (T2r_multivariate_en): night-1 runs.
- Real cases, --source night1 (default): T4r_debby_vorebuilt, T5r_pinatubo, T6r_pnw (night-1 baselines).
  --source night5: night5/t2/{debby_orig/part*, pin_orig, pnw_orig} (partial outputs allowed: *_final.csv,
  else *_snapshot.csv; all parts concatenated).
- prob_rule is kept as a column; the analysis (N5_T4_analysis.py) uses prob_rule == "linear" only (night-5
  decision after T0 PASS); night-5 runs have it forced.

Per configuration (columns)
- parameters: timewindow (steps), box_size, radius, eps_dbscan, min_samples_dbscan, score_mode, alpha, prob_rule,
  beta_softmax, lambda_a, l1_ratio.
- retention inputs for trace_st.retention.harmonized_retained: n_success, n_requested, median_len (steps, start
  excluded), n_full; adm = harmonized rule (rule 3) = "admissible" (rule 6, internal diagnostic); old = paper rule.
  T1: the 3 tracks are the members (n_success = tracks not failed; median of raw_len - 1; N = 30).
  T2: per alpha_mix (0.2/0.4/0.6/0.8) from results_members (ok, n_steps - 1; N = 30); admissible = all 4 alphas pass.
- completion = fraction of members at full length (T1 3 tracks raw_len == 31; T2 members n_steps == 31 averaged
  over alphas; Debby n_steps >= 358 (J_sensitivity), Pinatubo n_steps >= 17; PNW fraction_full_length).
- N/p: n_samples = (ny - 2r)(nx - 2r)(tw - 1), ny, nx = grid points in the box [c - b/2, c + b/2] at the start
  point (J_sensitivity.npts, checked against stored T7 values), n_features = p (2r + 1)^2.
- headline (step 0 excluded, N3_bed_summary.window_shares): Debby vo 5-15 d, Pinatubo TMSO201 4-16 d,
  PNW Z500 + TCWV 1-10 d (head_share = summed share of the headline variable(s); head_rank1 = the rank-1 variable
  is a headline variable). All variables' window shares are stored (sh_<window>_<var>).
  T1: share = fraction of steps on the true parent A (1 - parent2_frac), truth error = mean track distance to the
  true path (results_tracks mean_distance, mean over the 3 tracks; J_sensitivity / N4_summary definition).
  T2: share = true-source density share (1 - density_frac_var2), truth error = |estimated_alpha_density - alpha_mix|
  averaged over the 4 alphas (and at alpha 0.6 alone, err_a06).

Scale diagnostics (scale_diagnostics.json; how each is measured is written into the file)
- displacement per step of the tracked feature, per axis, in units of the method's move unit (spaceres: the stencil
  offset di is converted to a move of di * spaceres, trace_st/movement._add_delta_cols), Chebyshev max of the two axes:
  T1/T2: the generator's blob centres (A, C) over the traced steps; Debby: IBTrACS 2006234N12338 interpolated to 1 h
  (check: hourly pattern-correlation tracking of the vo anomaly); Pinatubo: westward advance per day of the
  TMSO201 plume front along the 7-23 N band (standardized field as in the run, 4 deg nearest regrid; front = farthest
  cell 0-340 deg west of the volcano with z >= 2; check: pattern-correlation tracking); PNW: daily pattern-correlation
  (PIV-style) tracking of the 2 deg Z500 anomaly (check: max-following).
- feature size at the start time: connected region around the peak nearest the start point where the tracked field
  >= half its peak; L_eq = sqrt(area), L_ext = largest of the lat / lon extents.

python N5_T4_data.py [--source night1|night5] [--nproc 16] [--skip-scale]      (PBS only)
Outputs: revision1/night5/analysis/T4/<source>/configs_<case>.csv, revision1/night5/analysis/T4/scale_diagnostics.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C  # noqa: E402
from N3_bed_summary import densities, window_shares  # noqa: E402
from trace_st import retention  # noqa: E402

R1 = C.R1
N5 = R1 / "night5"
T4 = N5 / "analysis" / "T4"
SP = C.REPO / "Scripts_Paper"
HEAD = dict(debby=("w5-15", ["vo"]), pinatubo=("w4-16", ["TMSO201"]), pnw=("w1-10", ["Z500", "TCWV"]))
SOURCES = dict(
    night1=dict(debby=[R1 / "T4r_debby_vorebuilt"], pinatubo=[R1 / "T5r_pinatubo"], pnw=[R1 / "T6r_pnw"]),
    night5=dict(debby=sorted((N5 / "t2" / "debby_orig").glob("part*")), pinatubo=[N5 / "t2" / "pin_orig"],
                pnw=[N5 / "t2" / "pnw_orig"]),
)
START = dict(debby=(28.0, -46.0 % 360), pinatubo=(15.0, -160.0 % 360), pnw=(52.5, 240.0))
SYN_GRID = (np.linspace(0, 200, 121), np.linspace(0, 200, 241))
T1_EVENTS = [(100.0, 50.0), (110.0, 105.0), (70.0, 100.0)]
PCOLS = ["timewindow", "box_size", "radius", "eps_dbscan", "min_samples_dbscan", "score_mode", "alpha", "prob_rule",
         "beta_softmax", "lambda_a", "l1_ratio"]


def npts(grid, c, b):
    return int(((grid >= c - b / 2) & (grid <= c + b / 2)).sum())


def n_over_p(box, r, tw, p, grids):
    """grids: [(ys, xs, (cy, cx))]; mean over start points (T1 has 3 events)."""
    ns = np.mean([max(npts(ys, c[0], box) - 2 * r, 0) * max(npts(xs, c[1], box) - 2 * r, 0) * (tw - 1)
                  for ys, xs, c in grids])
    nf = p * (2 * r + 1) ** 2
    return ns, nf, ns / nf


def tw_steps(s):
    return int(str(s).rstrip("dh"))


def add_np(t, p, grids):
    v = [n_over_p(float(b), int(r), tw_steps(w), p, grids) for b, r, w in zip(t.box_size, t.radius, t.timewindow)]
    t["n_samples"], t["n_features"], t["np_ratio"] = zip(*v)
    t["timewindow"] = t["timewindow"].map(tw_steps)
    return t


# ------------------------------------------------------------------ synthetic
def table_t1():
    d = C.RUNS["t1"]
    tr = pd.read_csv(d / "results_tracks_final.csv", usecols=["method", "trial_id", "raw_len", "mean_distance",
                                                              "parent2_frac", "failed"])
    tr = tr[tr.method == "granger"]
    jobs = pd.read_csv(d / "results_jobs_final.csv")
    jobs = jobs[(jobs.method == "granger") & (jobs.ok == 1)]
    tr["failed"] = tr["failed"].astype(str).str.lower().isin(["true", "1", "1.0"])
    g = tr.groupby("trial_id").agg(completion=("raw_len", lambda x: np.mean(x == 31)), min_raw_len=("raw_len", "min"),
                                   n_success=("failed", lambda x: int((~x).sum())), n_requested=("raw_len", "size"),
                                   median_len=("raw_len", lambda x: float(np.median(x - 1))),
                                   err=("mean_distance", "mean"), p2=("parent2_frac", "mean")).reset_index()
    t = jobs.merge(g, on="trial_id")
    t["n_full"] = 30
    t["share"] = 1 - t["p2"]
    t["rank1"] = t["share"] > 0.5
    t["old"] = t["min_raw_len"] == 31
    t["id"] = t["trial_id"]
    t = add_np(t, 2, [(SYN_GRID[0], SYN_GRID[1], c) for c in T1_EVENTS])
    return t


def table_t2():
    d = C.RUNS["t2"]
    jobs = pd.read_csv(d / "results_jobs_final.csv")
    jobs = jobs[jobs.ok == 1]
    al = pd.read_csv(d / "results_alphas_final.csv", usecols=["trial_id", "alpha_mix", "estimated_alpha_density",
                                                              "density_frac_var2"])
    m = pd.read_csv(d / "results_members_final.csv", usecols=["trial_id", "alpha_mix", "ok", "n_steps"])
    rows = []
    for (tid, a), g in m.groupby(["trial_id", "alpha_mix"]):
        okm = g[g.ok == 1]
        rows.append(dict(trial_id=tid, alpha_mix=a, n_success=len(okm), n_requested=len(g),
                         median_len=float(np.median(okm.n_steps - 1)) if len(okm) else 0.0,
                         comp=float(np.mean((g.ok == 1) & (g.n_steps == 31)))))
    ma = pd.DataFrame(rows)
    ma["adm_a"] = retention.harmonized_retained(ma.n_success, ma.n_requested, ma.median_len, 30).values
    al["e"] = (al.estimated_alpha_density - al.alpha_mix).abs()
    al["sh"] = 1 - al.density_frac_var2
    A = al.groupby("trial_id").agg(err=("e", "mean"), share=("sh", "mean")).reset_index()
    A = A.merge(al[np.isclose(al.alpha_mix, 0.6)][["trial_id", "e"]].rename(columns={"e": "err_a06"}), on="trial_id")
    M = ma.groupby("trial_id").agg(completion=("comp", "mean"), adm_all=("adm_a", "all"), n_success=("n_success", "sum"),
                                   n_requested=("n_requested", "sum"), median_len=("median_len", "min")).reset_index()
    t = jobs.merge(A, on="trial_id").merge(M, on="trial_id")
    t["n_full"] = 30
    t["rank1"] = t["share"] > 0.5
    t["old"] = True   # paper: all 300 T2 configs kept
    t["id"] = t["trial_id"]
    return add_np(t, 3, [(SYN_GRID[0], SYN_GRID[1], (100.0, 100.0))])


# ------------------------------------------------------------------ real cases
def _read(d, stem):
    for s in ("_final", "_snapshot", ""):
        f = d / f"{stem}{s}.csv"
        if f.exists():
            return pd.read_csv(f)
    return None


def table_bundle_case(case, dirs, nproc):
    out = []
    for k, d in enumerate(dirs):
        j = _read(d, "results_jobs")
        if j is None or not len(j):
            print("no table in", d); continue
        vars_ = [c.split("::")[1] for c in j.columns if c.startswith("density_full::") and "frac" not in c]
        vars_ = [v for v in vars_ if not v.startswith("frac")]
        mem = _read(d, "results_members")
        full = 358 if case == "debby" else 17
        if mem is not None:
            comp = mem.groupby("trial_id").apply(lambda g: np.mean((g.ok == 1) & (g.n_steps >= full))).rename("completion")
            j = j.merge(comp.reset_index(), on="trial_id", how="left")
        else:
            j["completion"] = np.nan
        j = j[j.ok == 1].copy()
        j["part"] = k
        j["old"] = (retention.debby_retained(j) if case == "debby" else retention.pinatubo_retained(j)).values
        j["adm_ret"] = retention.harmonized_case(case, j).values
        j["n_success"], j["n_requested"] = j.n_members_success, j.n_members_requested
        j["median_len"], j["n_full"] = j.median_traj_length, j.n_steps_time
        j["lambda_a"], j["l1_ratio"] = j["cd_kwargs.lambda_a"], j["cd_kwargs.l1_ratio"]
        ids = sorted(j.trial_id.astype(int))
        Dd, _ = densities(case, d, len(vars_), ids, nproc, 0)
        add_shares(j, "trial_id", Dd, case, vars_)
        b = np.load(j.bundle_npz.dropna().iloc[0])
        grids = [(np.asarray(b["ys"], float), np.asarray(b["xs"], float), START[case])]
        j = add_np(j, len(vars_), grids)
        j["vars"] = ",".join(vars_)
        out.append(j)
    return pd.concat(out, ignore_index=True) if out else None


def table_pnw(dirs, nproc):
    out = []
    for k, d in enumerate(dirs):
        s = _read(d, "hyperparam_search_summary")
        if s is None or not len(s):
            print("no table in", d); continue
        vars_ = [c.replace("days_1_10_integrated_norm_", "") for c in s.columns if c.startswith("days_1_10_integrated_norm_")]
        rows = []
        for rid in s.run_id:
            f = d / f"run_{int(rid):04d}" / "params.json"
            if not f.exists():
                continue
            p = json.load(open(f))
            rows.append(dict(run_id=int(rid), **{c: p[c] for c in PCOLS if c in p},
                             lambda_a=p["cd_kwargs"]["lambda_a"], l1_ratio=p["cd_kwargs"]["l1_ratio"]))
        t = s.merge(pd.DataFrame(rows), on="run_id")
        t["part"] = k
        t["completion"] = t["fraction_full_length"]
        t["old"] = retention.pnw_retained(t).values
        t["adm_ret"] = retention.harmonized_case("pnw", t).values
        t["median_len"], t["n_full"] = t.median_steps - 1, 30
        Dd, _ = densities("pnw", d, len(vars_), sorted(t.run_id.astype(int)), nproc, 0)
        add_shares(t, "run_id", Dd, "pnw", vars_)
        t = add_np(t, len(vars_), [(C.PNW_LATS, C.PNW_LONS, START["pnw"])])
        t["vars"] = ",".join(vars_)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else None


def add_shares(t, idc, Dd, case, vars_):
    w, hv = HEAD[case]
    hi = [vars_.index(v) for v in hv if v in vars_]
    cols = {}
    hs, hr, top = [], [], []
    for i in t[idc].astype(int):
        sh = window_shares(Dd[i], case) if i in Dd else {}
        for ww, s in sh.items():
            for q, v in enumerate(vars_):
                cols.setdefault(f"sh_{ww}_{v}", {})[i] = s[q]
        if w in sh:
            s = sh[w]
            hs.append(float(s[hi].sum())); top.append(vars_[int(np.argmax(s))]); hr.append(int(np.argmax(s)) in hi)
        else:
            hs.append(np.nan); top.append(None); hr.append(np.nan)
    for c, dct in cols.items():
        t[c] = t[idc].astype(int).map(dct)
    t["share"], t["rank1"], t["top_var"] = hs, hr, top
    t["id"] = t[idc]


# ------------------------------------------------------------------ scale diagnostics
def _load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"), SP / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step_disp(centers, unit):
    """Per-step Chebyshev displacement in move units (|dy|, |dx| / unit, max of the two)."""
    c = np.asarray(centers, float)
    d = np.abs(np.diff(c, axis=0)) / unit
    return np.max(d, axis=1)


def half_max_size(F, ys, xs, c, search):
    """Connected half-max region around the peak within `search` (coordinate units) of point c."""
    from scipy import ndimage
    F = np.nan_to_num(np.asarray(F, float), nan=-np.inf)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    near = (np.abs(Y - c[0]) <= search) & (np.abs(X - c[1]) <= search)
    Fn = np.where(near, F, -np.inf)
    iy, ix = np.unravel_index(np.argmax(Fn), F.shape)
    pk = F[iy, ix]
    lab, _ = ndimage.label(F >= 0.5 * pk)
    reg = lab == lab[iy, ix]
    dy, dx = abs(ys[1] - ys[0]), abs(xs[1] - xs[0])
    ext_y = (reg.any(1).sum()) * dy
    ext_x = (reg.any(0).sum()) * dx
    return dict(peak=float(pk), peak_at=[float(ys[iy]), float(xs[ix])], n_cells=int(reg.sum()),
                L_eq=float(np.sqrt(reg.sum() * dy * dx)), L_ext=float(max(ext_y, ext_x)), ext_lat=float(ext_y),
                ext_lon=float(ext_x), cell=[float(dy), float(dx)])


def max_follow(F, ys, xs, c0, rs):
    """Backward max-following: F (T, ny, nx) in forward time; start at c0 at the last time; at each earlier time the
    location of the maximum within +-rs cells of the previous location. Returns centres (back in time) and cap hits."""
    iy = int(np.argmin(np.abs(ys - c0[0]))); ix = int(np.argmin(np.abs(xs - c0[1])))
    path, hits = [(iy, ix)], 0
    for t in range(F.shape[0] - 2, -1, -1):
        y0, y1 = max(iy - rs, 0), min(iy + rs + 1, len(ys))
        xi = np.arange(ix - rs, ix + rs + 1) % len(xs)
        sub = np.nan_to_num(F[t, y0:y1][:, xi], nan=-np.inf)
        a, b = np.unravel_index(np.argmax(sub), sub.shape)
        ny_, nx_ = y0 + a, int(xi[b])
        if abs(ny_ - iy) >= rs or abs(b - rs) >= rs:
            hits += 1
        iy, ix = ny_, nx_
        path.append((iy, ix))
    cy = np.array([ys[i] for i, _ in path]); cx = np.array([xs[j] for _, j in path])
    cx = np.degrees(np.unwrap(np.radians(cx)))
    return np.c_[cy, cx], hits


def piv_track(F, ys, xs, c0, hw, smax):
    """Backward pattern-correlation (PIV-style) tracking: template = field at time t+1 in a (2hw+1)^2 window around
    the current centre; the centre at time t is the shift (|dy|, |dx| <= smax cells) that maximizes the Pearson
    correlation of the shifted window at t with the template. Longitude wraps; latitude windows are clipped.
    Returns centres (back in time, coordinate units) and the number of steps whose shift hit smax."""
    ny, nx = F.shape[1:]
    iy = int(np.argmin(np.abs(ys - c0[0]))); ix = int(np.argmin(np.abs(xs - c0[1])))
    path, hits = [(iy, ix)], 0

    def win(t, cy, cx):
        yy = np.clip(np.arange(cy - hw, cy + hw + 1), 0, ny - 1)
        xx = np.arange(cx - hw, cx + hw + 1) % nx
        return np.nan_to_num(F[t][np.ix_(yy, xx)].ravel())

    for t in range(F.shape[0] - 2, -1, -1):
        tmpl = win(t + 1, iy, ix); tmpl = tmpl - tmpl.mean()
        best, bs = -np.inf, (0, 0)
        for dy in range(-smax, smax + 1):
            cy = iy + dy
            if cy < 0 or cy >= ny:
                continue
            for dx in range(-smax, smax + 1):
                w = win(t, cy, ix + dx); w = w - w.mean()
                den = np.sqrt((w * w).sum() * (tmpl * tmpl).sum())
                r = (w * tmpl).sum() / den if den > 0 else -np.inf
                if r > best or (r == best and abs(dy) + abs(dx) < abs(bs[0]) + abs(bs[1])):
                    best, bs = r, (dy, dx)
        if max(abs(bs[0]), abs(bs[1])) >= smax:
            hits += 1
        iy, ix = iy + bs[0], (ix + bs[1]) % nx
        path.append((iy, ix))
    cy = np.array([ys[i] for i, _ in path]); cx = np.array([xs[j] for _, j in path])
    cx = np.degrees(np.unwrap(np.radians(cx)))
    return np.c_[cy, cx], hits


def summ(d, label, how, unit, extra=None):
    d = np.asarray(d, float)
    r = dict(label=label, how=how, unit=unit, n_steps=int(len(d)), max=float(d.max()), p90=float(np.percentile(d, 90)),
             median=float(np.median(d)), mean=float(d.mean()), r_rule=int(max(1, np.ceil(d.max() - 1e-9))))
    if extra:
        r.update(extra)
    return r


def scale_diagnostics():
    import xarray as xr
    out = {}
    # T1: generator tracks (spaceres 1: a stencil offset of one cell moves the centre by 1 coordinate unit)
    m1 = _load_script("1_SyntheticRuns_Univariate.py")
    fd, specs, _ = m1.build_synthetic_case()
    A = fd.isel(var=0).values  # (time, lat, lon)
    ys, xs = fd.lat.values, fd.lon.values
    disp, size = [], []
    for s in specs:
        cen = np.asarray(s["centers"], float)[-31:]
        disp.append(step_disp(cen, 1.0))
        size.append(half_max_size(A[-1], ys, xs, s["event"], 10.0))
    out["t1"] = dict(displacement=summ(np.concatenate(disp), "T1", "generator blob centres of the 3 tracks, last 31 days "
                                       "(30 traced steps), Chebyshev per-step move in spaceres units (1)", "spaceres=1 unit"),
                     size=dict(per_event=size, L_eq=float(np.mean([z["L_eq"] for z in size])),
                               L_ext=float(np.mean([z["L_ext"] for z in size])),
                               how="observed A field at the event day; half-max region around the peak within 10 units"),
                     spaceres=1.0, p=2)
    m3 = _load_script("3_SyntheticRuns_Multivariate.py")
    cs = m3.build_multivariate_case(alpha_mix=0.6)
    d2 = np.concatenate([step_disp(np.asarray(cs["centers_A"])[-31:], 1.0), step_disp(np.asarray(cs["centers_C"])[-31:], 1.0)])
    A2 = cs["full_data_run"].isel(var=0).values[-1]
    out["t2"] = dict(displacement=summ(d2, "T2", "generator centres of A and C, last 31 days, Chebyshev per-step move "
                                       "in spaceres units (1)", "spaceres=1 unit"),
                     size=half_max_size(A2, cs["ys"], cs["xs"], cs["event"], 10.0) | dict(
                         how="observed A field at the event day; half-max region around the peak within 10 units"),
                     spaceres=1.0, p=3)
    # Debby: IBTrACS, interpolated to 1 h, over the traced 360 h
    m5 = _load_script("5_DebbyRuns.py")
    tr = m5.load_debby_observed_track(ibtracs_path=str(SP / "IBTrACS.NA.v04r01.nc"))
    hb = np.asarray(tr["hours_back"], float)
    keep = hb <= 360
    hh = np.arange(0, min(hb[keep].max(), 360) + 1)
    lat = np.interp(hh, hb[keep], np.asarray(tr["lat"])[keep])
    lon = np.interp(hh, hb[keep], np.degrees(np.unwrap(np.radians(np.asarray(tr["lon_0360"])[keep]))))
    dD = step_disp(np.c_[lat, lon], 0.25)
    vo = xr.open_dataset(C.R1.parent / "FilesDebby" / "Vo_anomalies_TC_500-700.nc4").vo
    vo = vo.rename({"latitude": "lat", "longitude": "lon"}).sortby("lat")
    vo = vo.assign_coords(lon=vo.lon % 360).sortby("lon")
    vo = vo.assign_coords(time=vo.time.dt.round("1h")).sel(time=slice(None, "2006-08-26 00:00"))
    V = vo.values[-361:]
    pth, hits = piv_track(V, vo.lat.values, vo.lon.values, START["debby"], 8, 4)
    dV = step_disp(pth, 0.25)
    out["debby"] = dict(displacement=summ(dD, "Debby", f"IBTrACS 2006234N12338 fixes within 360 h of the start, "
                                          f"linearly interpolated to 1 h ({int(hh.max())} h covered), Chebyshev hourly move "
                                          "/ 0.25 deg", "0.25 deg"),
                        displacement_check=summ(dV, "Debby vo PIV", "hourly pattern-correlation tracking of the vo "
                                                "500-700 hPa anomaly (17x17-cell template, shifts <= 4 cells/h), 360 h",
                                                "0.25 deg", dict(cap_hits=int(hits))),
                        size=half_max_size(vo.values[-1], vo.lat.values, vo.lon.values, START["debby"], 2.0) | dict(
                            how="vo 500-700 hPa anomaly at 2006-08-26 00 UTC; half-max region around the peak within 2 deg"),
                        spaceres=0.25, p=3)
    # Pinatubo: standardized TMSO201 on the run grid (4 deg nearest), plume-front advance along 7-23 N
    m6 = _load_script("6_PinatuboRuns.py")
    fdp = m6.load_pinatubo_data(var_names=["TMSO201"]).sel(var="TMSO201").transpose("time", "lat", "lon")
    fdp = fdp.sel(time=slice("1991-06-15", "1991-07-02"))
    Z = fdp.values; plat, plon = fdp.lat.values, fdp.lon.values
    band = (plat >= 7) & (plat <= 23)
    H = np.nanmax(Z[:, band, :], axis=1)              # (time, lon)
    dwest = (120.35 - plon) % 360.0                    # degrees west of the volcano
    west = dwest <= 340.0                              # cells within 20 deg east of the volcano are near-source spread
    front = []
    for thr in (2.0, 3.0):
        f = np.array([dwest[(H[t] >= thr) & west].max() if ((H[t] >= thr) & west).any() else 0.0 for t in range(len(H))])
        front.append((thr, f))
    f2 = front[0][1]
    adv = np.abs(np.diff(f2)) / 4.0
    pthp, hp = piv_track(Z[-17:], plat, plon, START["pinatubo"], 4, 10)
    out["pinatubo"] = dict(displacement=summ(adv, "Pinatubo", "daily westward advance of the TMSO201 plume front "
                                             "(farthest cell 0-340 deg west of the volcano with z >= 2 in the 7-23 N band; "
                                             "z = run's standardization, 4 deg nearest grid), 1991-06-15..07-02, / 4 deg",
                                             "4 deg",
                                             dict(front_deg_west_z2=[float(x) for x in f2],
                                                  front_deg_west_z3=[float(x) for x in front[1][1]],
                                                  adv_cells_z3=[float(x) for x in np.abs(np.diff(front[1][1])) / 4.0],
                                                  start_deg_west=float((120.35 - START["pinatubo"][1]) % 360),
                                                  mean_speed_cells=float(((120.35 - START["pinatubo"][1]) % 360) / 17 / 4))),
                           displacement_check=summ(step_disp(pthp, 4.0), "Pinatubo PIV", "daily pattern-correlation "
                                                   "tracking of standardized TMSO201 from the start point (9x9 template, "
                                                   "shifts <= 10 cells/day), 16 steps", "4 deg", dict(cap_hits=int(hp))),
                           size=half_max_size(Z[-1], plat, plon, START["pinatubo"], 8.0) | dict(
                               how="standardized TMSO201 on 1991-07-02; half-max region around the peak within 8 deg"),
                           spaceres=4.0, p=4)
    # PNW: Z500 anomaly regridded to 2 deg (as 8_HW_PNW21_Runs_v2.regrid_to_2p5), daily PIV tracking
    z = xr.open_dataset(C.R1.parent / "JointFiles" / "z500_daily_anoms.nc").Z
    z = z.rename({"latitude": "lat", "longitude": "lon"}).sel(time=slice("2021-05-31", "2021-06-30")).sortby("lat")
    z = z.assign_coords(lon=z.lon % 360).sortby("lon").transpose("time", "lat", "lon").load()
    z = z.interp(lat=C.PNW_LATS, lon=C.PNW_LONS, method="linear")
    Zp = z.values
    pthz, hz = piv_track(Zp, C.PNW_LATS, C.PNW_LONS, START["pnw"], 8, 8)
    pthm, hm = max_follow(Zp, C.PNW_LATS, C.PNW_LONS, START["pnw"], 6)
    out["pnw"] = dict(displacement=summ(step_disp(pthz, 2.0), "PNW", "daily pattern-correlation tracking of the Z500 "
                                        "anomaly (2 deg grid; 17x17-cell template, shifts <= 8 cells/day) from the start "
                                        "point, 30 days back", "2 deg",
                                        dict(cap_hits=int(hz), path=[[float(a), float(b)] for a, b in pthz])),
                      displacement_check=summ(step_disp(pthm, 2.0), "PNW maxfollow", "daily max-following of the Z500 "
                                              "anomaly, search +-6 cells", "2 deg", dict(cap_hits=int(hm))),
                      size=half_max_size(Zp[-1], C.PNW_LATS, C.PNW_LONS, START["pnw"], 6.0) | dict(
                          how="Z500 anomaly on 2021-06-30 (2 deg); half-max region around the peak within 6 deg"),
                      spaceres=2.0, p=6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="night1", choices=list(SOURCES))
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--skip-scale", action="store_true")
    ap.add_argument("--skip-tables", action="store_true")
    a = ap.parse_args()
    od = T4 / a.source
    od.mkdir(parents=True, exist_ok=True)
    if not a.skip_tables:
        tabs = dict(t1=table_t1(), t2=table_t2())
        src = SOURCES[a.source]
        for case in ("debby", "pinatubo"):
            tabs[case] = table_bundle_case(case, src[case], a.nproc)
        tabs["pnw"] = table_pnw(src["pnw"], a.nproc)
        for k, t in tabs.items():
            if t is None:
                print("MISSING", k); continue
            if "adm_ret" not in t:
                t["adm_ret"] = (t["adm_all"] if k == "t2" else
                                retention.harmonized_retained(t.n_success, t.n_requested, t.median_len, 30).values)
            t["case"] = k
            t.to_csv(od / f"configs_{k}.csv", index=False)
            lin = t[t.prob_rule == "linear"]
            print(f"{k}: n={len(t)} linear={len(lin)} adm(all)={int(t.adm_ret.sum())} adm(linear)={int(lin.adm_ret.sum())} "
                  f"old={int(t.old.sum())} N/p median={t.np_ratio.median():.3g} [{t.np_ratio.min():.3g}, {t.np_ratio.max():.3g}] "
                  f"share(adm lin)={lin.loc[lin.adm_ret.astype(bool), 'share'].mean():.3f}", flush=True)
    if not a.skip_scale:
        sd = scale_diagnostics()
        json.dump(sd, open(T4 / "scale_diagnostics.json", "w"), indent=1)
        for k, v in sd.items():
            print(k, {kk: v["displacement"][kk] for kk in ("max", "p90", "median", "r_rule")},
                  {kk: v["size"][kk] for kk in ("L_eq", "L_ext")} if "L_eq" in v["size"] else "",
                  v.get("displacement_check", {}).get("max"), flush=True)


if __name__ == "__main__":
    main()
