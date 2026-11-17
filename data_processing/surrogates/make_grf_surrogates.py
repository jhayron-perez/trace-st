"""Gaussian-random-field (GRF) surrogates for the TraCE-ST real cases (revision 1, night 5 T6).

Each surrogate is an unconnected field matched to one source variable of the case, as the case sees it
(the case loader's analysis grid, the case's analysis time window):
  - variance: per-cell variance over the analysis window (validated as the cell mean),
  - persistence: AR(1) in time, phi calibrated so the pooled lag-1 autocorrelation matches
    (e-folding time tau = -1 / ln r1),
  - spatial scale: e-folding lags of the pooled spatial autocorrelation along lat and lon
    (grid cells of the analysis grid).

Construction (on the analysis grid):
  e_t   = white noise filtered in Fourier space with an anisotropic Gaussian kernel (std s_lat, s_lon cells),
          scaled to unit variance (zero-padded domain for regional grids, periodic in lon for global grids);
  u_t   = phi u_{t-1} + sqrt(1 - phi^2) e_t,  u_0 = e_0 (stationary, unit variance);
  x_t   = sigma(cell, in/out of window) * u_t   (Pinatubo: sigma chosen so that after the loader's
          re-standardization over the whole year the window variance matches the source).
phi, s_lat, s_lon and a global variance factor are calibrated by fixed-point iteration on one calibration
realization (seed recorded) until the measured statistics match the source within 2 %.
The analysis-grid field is copied to the native grid of the input files by nearest analysis cell
(native cells the loader never reads are filled with the value of their nearest analysis cell), packed as
int16 (scale_factor, zlib) and written with the coordinates, time axis and naming the case loader expects.

All randomness: np.random.default_rng(seed) passed explicitly; seed = BASE + 100000*case_idx + 100*draw + slot
(calibration: draw 99). Slots round-robin over the case's variable types in the extended-set order.

Usage (compute node only):
  python make_grf_surrogates.py --case pnw --outdir OUT [--draws 20] [--k 16] [--nproc 16]
Outputs: OUT/<case>/calibration.json, draw<NN>/ files (+ symlinks to the real extra inputs where the
loader needs one directory), surrogate_stats.csv, seeds.csv, loader_check.json, sha256.txt,
OUT/validation_<case>.csv/.md.
"""
import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "Scripts_Paper"
SCRATCH = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs")
BASE_SEED = 20261116
CASE_IDX = {"pnw": 1, "pinatubo": 2, "debby": 3}
CALIB_DRAW = 99
TOL_CALIB = 0.02
TOL_PASS = 0.10
INV_E = float(np.exp(-1.0))


# -----------------------------------------------------------------------------
# Generator and statistics (pure numpy; unit-tested in tests/test_surrogates.py)
# -----------------------------------------------------------------------------
def unit_grf_ar1(nt, ny, nx, phi, s_lat, s_lon, rng, periodic_lon=False, chunk=48, pad=None):
    """Unit-variance AR(1)-in-time field (nt, ny, nx), Gaussian spatial kernel std (s_lat, s_lon) cells.
    pad (py, px): zero-padding cells of the noise grid (default 4 s + 1; fixed during calibration so that the
    noise realization does not change with s)."""
    py = int(np.ceil(4.0 * s_lat)) + 1 if pad is None else int(pad[0])
    px = 0 if periodic_lon else (int(np.ceil(4.0 * s_lon)) + 1 if pad is None else int(pad[1]))
    NY, NX = ny + 2 * py, nx + 2 * px
    fy = np.fft.fftfreq(NY)[:, None]
    G = np.exp(-2.0 * np.pi ** 2 * (s_lat ** 2 * fy ** 2 + s_lon ** 2 * np.fft.rfftfreq(NX)[None, :] ** 2))
    var = np.mean(np.exp(-4.0 * np.pi ** 2 * (s_lat ** 2 * fy ** 2 + s_lon ** 2 * np.fft.fftfreq(NX)[None, :] ** 2)))
    G = G / np.sqrt(var)
    a, b = float(phi), float(np.sqrt(max(1.0 - phi ** 2, 0.0)))
    out = np.empty((nt, ny, nx), dtype=np.float32)
    prev = None
    for t0 in range(0, nt, chunk):
        c = min(chunk, nt - t0)
        w = rng.standard_normal((c, NY, NX))
        e = np.fft.irfft2(np.fft.rfft2(w) * G, s=(NY, NX))[:, py:py + ny, px:px + nx]
        for i in range(c):
            prev = e[i] if prev is None else a * prev + b * e[i]
            out[t0 + i] = prev
    return out


def _efold(z, axis, max_lag):
    """e-folding lag (cells) of the pooled lag correlation of z along axis (pairs normalized by their own
    second moments); nan if not reached within max_lag."""
    r_prev = 1.0
    n = z.shape[axis]
    ok = np.isfinite(z).astype(np.float32)
    z0 = np.where(ok > 0, z, 0.0).astype(np.float32)
    for d in range(1, min(max_lag, n - 1) + 1):
        a, b = np.take(z0, np.arange(d, n), axis=axis), np.take(z0, np.arange(0, n - d), axis=axis)
        ma, mb = np.take(ok, np.arange(d, n), axis=axis), np.take(ok, np.arange(0, n - d), axis=axis)
        sab = np.sum(a * b, dtype=np.float64)
        saa, sbb = np.sum(a * a * mb, dtype=np.float64), np.sum(b * b * ma, dtype=np.float64)
        r = float(sab / np.sqrt(saa * sbb))
        if r < INV_E:
            return d - 1 + (r_prev - INV_E) / (r_prev - r)
        r_prev = r
    return np.nan


def field_stats(x, tmask=None, cellmask=None, max_lag=60, remove_common=True, max_frames=60):
    """Statistics of x (nt, ny, nx) over the time window tmask: variance (cell mean of per-cell variance over
    cellmask), pooled lag-1 autocorrelation r1 of the per-cell standardized anomalies, tau = -1/ln r1, and the
    e-folding lengths L_lat, L_lon (cells) of their spatial lag correlation after removing, at each time, the
    domain mean (remove_common: the window's common trend would otherwise correlate all cells). The spatial
    lengths use every k-th time of the window so that at most max_frames frames enter (Debby: 361 -> 61)."""
    w = np.asarray(x if tmask is None else x[np.asarray(tmask)], dtype=np.float64)
    mu = np.nanmean(w, axis=0)
    sd = np.nanstd(w, axis=0)
    var_map = sd ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ((w - mu) / np.where(sd > 0, sd, np.nan)).astype(np.float32)
    del w
    if cellmask is None:
        cellmask = np.isfinite(var_map)
    r1 = float(np.nanmean(z[1:] * z[:-1], dtype=np.float64))
    tau = float(-1.0 / np.log(r1)) if 0 < r1 < 1 else np.nan
    z = z[::int(np.ceil(z.shape[0] / max_frames))]
    if remove_common:
        z = z - np.nanmean(z, axis=(1, 2), keepdims=True)
    return dict(var=float(np.nanmean(var_map[cellmask])), r1=r1, tau=tau,
                L_lat=_efold(z, 1, max_lag), L_lon=_efold(z, 2, max_lag)), var_map


def standardize_time(x):
    """What the Pinatubo loader does: (x - mean_time) / std_time per cell."""
    return ((x - x.mean(0)) / x.std(0)).astype(np.float32)


def scale_field(u, tmask, var_map_w, sd_out, c, restandardize, apply_std=True):
    """x = sigma * u; sigma from the source window variance map (x c) and out-of-window sd.
    restandardize (Pinatubo loader): sigma_out = 1, sigma_w^2 = v(1-fw)/(1-v fw), then (x-mean)/std over time
    (apply_std=False returns the field before that step, i.e. what is written to the file)."""
    tm = np.asarray(tmask)
    v = np.where(np.isfinite(var_map_w), var_map_w, 1.0) * c
    if restandardize:
        fw = tm.mean()
        v = np.minimum(v, 0.95 / fw)
        sw = np.sqrt(v * (1.0 - fw) / (1.0 - v * fw))
        so = np.ones_like(sw)
    else:
        sw = np.sqrt(v)
        so = np.where(np.isfinite(sd_out), sd_out, np.sqrt(np.where(np.isfinite(var_map_w), var_map_w, 1.0)))
    x = (u * np.where(tm[:, None, None], sw[None], so[None])).astype(np.float32)
    if restandardize and apply_std:
        x = standardize_time(x)
    return x


def calibrate(target, shape, tmask, var_map_w, sd_out, cellmask, seed, periodic_lon, restandardize,
              max_iter=12, max_lag=60, n_real=1, damp=0.7, log=print):
    """Fixed-point calibration of (phi, s_lat, s_lon, c) so the measured stats (mean over n_real calibration
    realizations, seeds seed + 7919 j) match target within TOL_CALIB; returns the best iterate."""
    phi = float(np.clip(target["r1"], 0.0, 0.999))
    s_lat = max(0.5 * target["L_lat"], 0.05) if np.isfinite(target["L_lat"]) else 0.5 * max_lag
    s_lon = max(0.5 * target["L_lon"], 0.05) if np.isfinite(target["L_lon"]) else 0.5 * max_lag
    c = 1.0
    hist, best = [], None
    pad = [int(np.ceil(6.0 * s_lat)) + 1, 0 if periodic_lon else int(np.ceil(6.0 * s_lon)) + 1]
    for it in range(max_iter):
        if 4.0 * s_lat + 1 > pad[0]:
            pad[0] = int(np.ceil(6.0 * s_lat)) + 1
        if not periodic_lon and 4.0 * s_lon + 1 > pad[1]:
            pad[1] = int(np.ceil(6.0 * s_lon)) + 1
        ms = []
        for j in range(n_real):
            u = unit_grf_ar1(*shape, phi, s_lat, s_lon, np.random.default_rng(seed + 7919 * j),
                             periodic_lon=periodic_lon, pad=pad)
            ms.append(field_stats(scale_field(u, tmask, var_map_w, sd_out, c, restandardize), tmask, cellmask,
                                  max_lag)[0])
            del u
        m = {k: float(np.mean([d[k] for d in ms])) for k in ms[0]}
        err = {k: stat_err(k, m[k], target[k]) for k in ("var", "r1", "L_lat", "L_lon")}
        worst = max(abs(v) if np.isfinite(v) else np.inf for v in err.values())
        hist.append(dict(it=it, phi=phi, s_lat=s_lat, s_lon=s_lon, c=c, worst=worst,
                         **{f"m_{k}": v for k, v in m.items()}, **{f"e_{k}": v for k, v in err.items()}))
        log(f"    it {it}: phi {phi:.4f} s {s_lat:.3f},{s_lon:.3f} c {c:.3f} | "
            + " ".join(f"{k} {m[k]:.4g}/{target[k]:.4g}" for k in err))
        if best is None or worst < best[0]:
            best = (worst, dict(phi=phi, s_lat=s_lat, s_lon=s_lon, c=c, pad_lat=pad[0], pad_lon=pad[1]))
        if worst < TOL_CALIB:
            break
        phi = float(np.clip(phi + damp * (target["r1"] - m["r1"]), 0.0, 0.998))
        for key, s_key in (("L_lat", "s_lat"), ("L_lon", "s_lon")):
            s = s_lat if s_key == "s_lat" else s_lon
            if np.isfinite(target[key]) and np.isfinite(m[key]):
                s = s * (target[key] / m[key]) ** damp
            elif np.isfinite(target[key]):          # surrogate too smooth (no e-folding within max_lag)
                s = 0.7 * s
            elif np.isfinite(m[key]):               # source has no e-folding within max_lag: smoother
                s = 1.3 * s
            s = float(np.clip(s, 0.05, 80.0))
            if s_key == "s_lat":
                s_lat = s
            else:
                s_lon = s
        if m["var"] > 0:
            c = float(c * (target["var"] / m["var"]) ** damp)
    return best[1], hist


def stat_err(key, a, b):
    """Relative error; for L, both nan (no e-folding within max_lag) counts as a match (0)."""
    if key.startswith("L") and not np.isfinite(a) and not np.isfinite(b):
        return 0.0
    return rel_err(a, b)


def rel_err(a, b):
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return np.nan
    return float((a - b) / abs(b))


def type_for_slot(types, slot):
    """Round-robin: slot 1..K -> variable type, in the extended-set order."""
    return types[(slot - 1) % len(types)]


def surrogate_seed(case, draw, slot):
    return int(BASE_SEED + 100000 * CASE_IDX[case] + 100 * int(draw) + int(slot))


def quantize(x):
    """int16 packing used for the files (scale_factor = max|x|/32000, offset 0); returns (int16, scale)."""
    m = float(np.nanmax(np.abs(x)))
    scale = np.float32(m / 32000.0 if m > 0 else 1.0)
    return np.round(x / scale).astype(np.int16), scale


# -----------------------------------------------------------------------------
# Case adapters
# -----------------------------------------------------------------------------
def load_script(name):
    spec = importlib.util.spec_from_file_location("grf_" + name.replace(".py", "").lstrip("0123456789_"),
                                                  SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DEB_IN = SCRATCH / "revision1/inputs_debby"
DEB_IN5 = SCRATCH / "revision1/night5/inputs_debby"
DEBBY_VO = (str(DEB_IN / "vo500700_2006_rebuilt.nc"), "vo")
DEBBY_EXTRAS = [("CAPE", str(DEB_IN / "cape_2006.nc"), "cape"), ("DIV200", str(DEB_IN / "d200_2006.nc"), "d"),
                ("WS10", str(DEB_IN5 / "ws10_2006.nc"), "ws10"), ("MSLP", str(DEB_IN5 / "msl_2006.nc"), "msl")]
PIN_SANDIA = SCRATCH / "Sandia"
PIN_SPA = SCRATCH / "E3SM_SPA_full"
PIN_EXTRAS = ["TMSO203", "BURDENSO403", "FLUT", "T050", "FSDS"]
PNW_JOINT = SCRATCH / "JointFiles"
PNW_IN5 = SCRATCH / "revision1/night5/inputs_pnw"
PNW_EXTRAS = ["t2m", "blh"]

CASES = {
    "debby": dict(script="5_DebbyRuns.py", space="SEARCH_SPACE_SHARED_DEBBY",
                  types=["Tb", "vo", "precip", "CAPE", "DIV200", "WS10", "MSLP"], restandardize=False, n_real=2),
    "pinatubo": dict(script="6_PinatuboRuns.py", space="SEARCH_SPACE_SHARED_PINATUBO",
                     types=["TMSO201", "BURDENSO401", "TMH2SO401", "AEROD_v"] + PIN_EXTRAS, restandardize=True, n_real=20),
    "pnw": dict(script="8_HW_PNW21_Runs_v2.py", space="SEARCH_SPACE_SHARED_PNW",
                types=["z500", "z10", "olr", "tcwv", "slhf", "sshf", "t2m", "blh"], restandardize=False, n_real=20),
}


def surrogate_name(case, slot, vtype):
    """Variable name the case sees (and file stem). PNW: stem lower-case, loader upper-cases it."""
    clean = "".join(ch for ch in vtype if ch.isalnum())
    return f"s{slot:02d}{clean.lower()}" if case == "pnw" else f"S{slot:02d}{clean}"


def build_case(case, surrogates=None, drawdir=None):
    """Case dict from the run script's own builder (extended set [+ surrogates of one draw])."""
    m = load_script(CASES[case]["script"])
    names = [n for n, _ in (surrogates or [])]
    if case == "debby":
        extra = DEBBY_EXTRAS + [(n, str(Path(drawdir) / f"{n}.nc"), n) for n in names]
        c = m.build_case_debby(extra_inputs=extra, vo_input=DEBBY_VO)
    elif case == "pinatubo":
        os.environ["TST_PINATUBO_EXTRA"] = ",".join(PIN_EXTRAS + names)
        if drawdir is not None:
            os.environ["TST_PINATUBO_EXTRA_PATH"] = str(drawdir)
        try:
            c = m.build_case_pinatubo()
        finally:
            os.environ.pop("TST_PINATUBO_EXTRA", None)
            os.environ.pop("TST_PINATUBO_EXTRA_PATH", None)
    else:
        c = m.build_case_pnw(extra_vars=PNW_EXTRAS + names, extra_path=str(drawdir or PNW_IN5))
    return m, c


def analysis_var_name(m, case, vtype):
    if case == "pnw":
        return m.VAR_NAME_MAP.get(vtype) or m.EXTRA_NAME_MAP.get(vtype, vtype.upper())
    return vtype


def analysis_arrays(full_data):
    """(var, time, lat, lon) with lat ascending and lon contiguous (regional grids: -180..180 order)."""
    fd = full_data.transpose("var", "time", "lat", "lon").sortby("lat")
    lon = np.asarray(fd.lon.values, dtype=float)
    dlon = float(np.median(np.diff(np.sort(lon % 360))))
    periodic = bool(abs(len(lon) * dlon - 360.0) < 1e-6)
    if not periodic:
        fd = fd.assign_coords(lon_m=(("lon",), ((lon + 180.0) % 360.0) - 180.0)).sortby("lon_m")
    return fd, periodic


def analysis_window(m, case, c, times):
    pb = c["params_base"]
    dt = pd.to_timedelta(pb["timeres"])
    tw = max(pd.to_timedelta(x) for x in getattr(m, CASES[case]["space"])["timewindow"])
    end = pd.Timestamp(c["date_end"])
    start = end - (int(pb["n_steps_time"]) - 1) * dt - tw + dt
    t = pd.to_datetime(times)
    return np.asarray((t >= start) & (t <= end)), (str(start), str(end))


def template_for(case, vtype):
    """(path, varname) of the native file whose grid/time axis the surrogate copies."""
    if case == "debby":
        return DEBBY_EXTRAS[0][1], DEBBY_EXTRAS[0][2]          # cape_2006.nc (an --extra-inputs file)
    if case == "pinatubo":
        return str(PIN_SANDIA / "TMSO201_199101_199112.nc"), "TMSO201"
    return str(PNW_JOINT / "z500_daily_anoms.nc"), None        # PNW: grid of the target variable


def native_layout(case, fd_coords, periodic, time_slice=None):
    """Template coords and index maps native -> analysis (time, lat, lon)."""
    path, vname = template_for(case, None)
    ds = xr.open_dataset(path, decode_times=False)
    vname = vname or list(ds.data_vars)[0]
    da = ds[vname]
    dims = da.dims
    tdim = [d for d in dims if "time" in d][0]
    ydim = [d for d in dims if d.startswith("lat")][0]
    xdim = [d for d in dims if d.startswith("lon")][0]
    if case == "pinatubo":  # noleap calendar; the loader converts with to_datetimeindex(unsafe=True)
        tdec = xr.decode_cf(ds[[tdim]], use_cftime=True).indexes[tdim].to_datetimeindex(unsafe=True)
    else:
        tdec = pd.DatetimeIndex(xr.decode_cf(ds[[tdim]])[tdim].values)
    if case == "debby":
        tdec = tdec.round("1h")
    isel_t = np.arange(len(tdec))
    if time_slice is not None:
        isel_t = np.where((tdec >= pd.Timestamp(time_slice[0])) & (tdec <= pd.Timestamp(time_slice[1]) + pd.Timedelta("23h")))[0]
    tdec = tdec[isel_t]
    it = pd.DatetimeIndex(pd.to_datetime(fd_coords["time"])).get_indexer(tdec)
    if (it < 0).any():
        raise ValueError(f"{case}: template times not on the analysis time axis")
    lat_a = np.asarray(fd_coords["lat"], dtype=float)
    lon_a = np.asarray(fd_coords["lon"], dtype=float) % 360.0
    lat_n = np.asarray(ds[ydim].values, dtype=float)
    lon_n = np.asarray(ds[xdim].values, dtype=float) % 360.0
    iy = np.abs(lat_n[:, None] - lat_a[None, :]).argmin(1)
    dl = np.abs(lon_n[:, None] - lon_a[None, :])
    ix = np.minimum(dl, 360.0 - dl).argmin(1)
    coords = {k: v for k, v in da.isel({tdim: isel_t}).coords.items()}
    ds.close()
    return dict(dims=dims, tdim=tdim, ydim=ydim, xdim=xdim, it=it, iy=iy, ix=ix, coords=coords)


def write_surrogate(path, name, x_analysis, lay, attrs):
    """Analysis-grid field (time, lat, lon) -> native file in the template layout, int16 packed."""
    xn = x_analysis[lay["it"]][:, lay["iy"]][:, :, lay["ix"]]
    order = [{lay["tdim"]: 0, lay["ydim"]: 1, lay["xdim"]: 2}[d] for d in lay["dims"]]
    q, scale = quantize(np.transpose(xn, order))
    da = xr.DataArray(q.astype(np.float32) * scale, dims=lay["dims"], coords=lay["coords"], name=name, attrs=attrs)
    enc = {name: dict(dtype="int16", scale_factor=float(scale), add_offset=0.0, _FillValue=np.int16(-32768),
                      zlib=True, complevel=1, shuffle=True)}
    tmp = Path(str(path) + ".tmp")
    da.to_dataset().to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)
    return scale


def file_name(case, name):
    if case == "debby":
        return f"{name}.nc"
    if case == "pinatubo":
        return f"{name}_199101_199112.nc"
    return f"{name}_daily_anoms.nc"


def link_real_extras(case, drawdir):
    """Pinatubo/PNW read every extra variable from one directory: link the real extended-set extras."""
    if case == "pinatubo":
        srcs = [PIN_SPA / f"{v}_199101_199112.nc" for v in PIN_EXTRAS]
    elif case == "pnw":
        srcs = [PNW_IN5 / f"{v}_daily_anoms.nc" for v in PNW_EXTRAS]
    else:
        return
    for s in srcs:
        dst = Path(drawdir) / s.name
        if not dst.is_symlink():
            dst.symlink_to(s)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
_G = {}


def _gen_task(task):
    case, draw, slot, vtype = task
    g = _G
    p, tgt = g["calib"][vtype]["params"], g["calib"][vtype]
    name = surrogate_name(case, slot, vtype)
    seed = surrogate_seed(case, draw, slot)
    t0 = time.time()
    u = unit_grf_ar1(*g["shape"], p["phi"], p["s_lat"], p["s_lon"], np.random.default_rng(seed),
                     periodic_lon=g["periodic"], pad=(p["pad_lat"], p["pad_lon"]))
    x = scale_field(u, g["tmask"], g["var_w"][vtype], g["sd_out"][vtype], p["c"], g["restandardize"],
                    apply_std=False)                                        # values written to the file
    del u
    q, scale = quantize(x)
    xq = q.astype(np.float32) * scale                                       # values read back from the file
    xs = standardize_time(xq) if g["restandardize"] else xq                 # as the case sees them
    m, _ = field_stats(xs, g["tmask"], g["cellmask"][vtype], g["max_lag"])
    drawdir = Path(g["outcase"]) / f"draw{draw:02d}"
    path = drawdir / file_name(case, name)
    write_surrogate(path, name, xq, g["lay"],
                    dict(long_name=f"GRF surrogate of {vtype} (TraCE-ST revision-1 T6)", source_variable=vtype,
                         seed=seed, phi=p["phi"], s_lat_cells=p["s_lat"], s_lon_cells=p["s_lon"],
                         generator="data_processing/surrogates/make_grf_surrogates.py"))
    row = dict(case=case, draw=draw, slot=slot, name=name, source=vtype, seed=seed, file=str(path),
               bytes=path.stat().st_size, sha256=sha256(path), secs=round(time.time() - t0, 1),
               **{f"sur_{k}": v for k, v in m.items()},
               **{f"src_{k}": tgt["target"][k] for k in ("var", "r1", "tau", "L_lat", "L_lon")})
    if draw == g.get("keep_draw"):
        np.save(drawdir / f".{name}_analysis.npy", xs)
    return row


def run_case(case, outdir, draws, k, nproc, max_lag=60, log=print):
    cfg = CASES[case]
    out = Path(outdir) / case
    out.mkdir(parents=True, exist_ok=True)
    log(f"[{case}] loading sources via the run script's builder")
    m, c = build_case(case)
    fd, periodic = analysis_arrays(c["full_data"])
    tmask, win = analysis_window(m, case, c, fd.time.values)
    shape = (fd.sizes["time"], fd.sizes["lat"], fd.sizes["lon"])
    log(f"[{case}] analysis grid {shape}, periodic lon {periodic}, window {win} ({int(tmask.sum())} steps)")
    calib_path = out / "calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    var_w, sd_out, cellmask = {}, {}, {}
    for ti, vt in enumerate(cfg["types"]):
        src = np.asarray(fd.sel(var=analysis_var_name(m, case, vt)).values, dtype=np.float32)
        tgt, vmap = field_stats(src, tmask, None, max_lag)
        var_w[vt] = vmap
        cellmask[vt] = np.isfinite(vmap)
        sd_out[vt] = np.nanstd(src[~tmask], axis=0) if (~tmask).any() else np.full(vmap.shape, np.nan)
        if vt not in calib:
            seed = surrogate_seed(case, CALIB_DRAW, ti + 1)
            # large-scale fields have few independent samples: more calibration realizations (up to 8)
            ll = np.sqrt(np.nan_to_num(tgt["L_lat"], nan=max_lag) * np.nan_to_num(tgt["L_lon"], nan=max_lag))
            n_real = int(np.clip(np.ceil(cfg["n_real"] * ll / 10.0), cfg["n_real"], max(cfg["n_real"], 8)))
            log(f"[{case}] calibrate {vt} (seed {seed}, {n_real} realizations): target {tgt}")
            params, hist = calibrate(tgt, shape, tmask, vmap, sd_out[vt], cellmask[vt], seed, periodic,
                                     cfg["restandardize"], max_lag=max_lag, n_real=n_real, log=log)
            calib[vt] = dict(target=tgt, params=params, seed=seed, n_real=n_real, history=hist)
            calib_path.write_text(json.dumps(dict(calib, _meta=dict(window=win, shape=shape, periodic=periodic)),
                                             indent=1, default=float))
    calib.pop("_meta", None)
    lay = native_layout(case, dict(time=fd.time.values, lat=fd.lat.values, lon=fd.lon.values), periodic,
                        time_slice=getattr(m, "TIME_SLICE", None) if case == "pnw" else None)
    _G.update(calib=calib, shape=shape, periodic=periodic, tmask=tmask, var_w=var_w, sd_out=sd_out,
              cellmask=cellmask, max_lag=max_lag, restandardize=cfg["restandardize"], outcase=str(out), lay=lay,
              keep_draw=draws[0])
    for d in draws:
        (out / f"draw{d:02d}").mkdir(exist_ok=True)
        link_real_extras(case, out / f"draw{d:02d}")
    tasks = [(case, d, s, type_for_slot(cfg["types"], s)) for d in draws for s in range(1, k + 1)]
    log(f"[{case}] generating {len(tasks)} surrogates with {nproc} processes")
    del c, fd
    with mp.get_context("fork").Pool(nproc) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_gen_task, tasks)):
            rows.append(r)
            if (i + 1) % 16 == 0:
                log(f"  {i + 1}/{len(tasks)}")
    st = pd.DataFrame(rows).sort_values(["draw", "slot"])
    st.to_csv(out / "surrogate_stats.csv", index=False)
    st[["case", "draw", "slot", "name", "source", "seed"]].to_csv(out / "seeds.csv", index=False)
    with open(out / "sha256.txt", "w") as f:
        for _, r in st.iterrows():
            f.write(f"{r.sha256}  {Path(r.file).relative_to(out)}\n")
    loader_check(case, out, draws[0], k, cfg, log)
    validation_table(case, st, Path(outdir), calib)
    return st


def loader_check(case, out, draw, k, cfg, log=print):
    """The run script's own loader reads draw `draw` (extended set + K surrogates): same values as generated."""
    drawdir = out / f"draw{draw:02d}"
    sur = [(surrogate_name(case, s, type_for_slot(cfg["types"], s)), type_for_slot(cfg["types"], s))
           for s in range(1, k + 1)]
    m, c = build_case(case, sur, drawdir)
    fd, _ = analysis_arrays(c["full_data"])
    names = [n.upper() if case == "pnw" else n for n, _ in sur]   # PNW loader: EXTRA_NAME_MAP.get(s, s.upper())
    res = dict(case=case, draw=draw, var_names=list(map(str, c["var_names"])), n_vars=len(c["var_names"]),
               nan_frac={}, max_abs_diff={})
    for (n, _), an in zip(sur, names):
        got = np.asarray(fd.sel(var=an).values, dtype=np.float64)
        ref = np.load(drawdir / f".{n}_analysis.npy").astype(np.float64)
        res["nan_frac"][an] = float(np.isnan(got).mean())
        res["max_abs_diff"][an] = float(np.nanmax(np.abs(got - ref)))
    res["ok"] = bool(max(res["max_abs_diff"].values()) < 1e-3 and max(res["nan_frac"].values()) == 0.0)
    (out / "loader_check.json").write_text(json.dumps(res, indent=1))
    for n, _ in sur:
        (drawdir / f".{n}_analysis.npy").unlink()
    log(f"[{case}] loader check draw {draw}: ok={res['ok']} max|diff|={max(res['max_abs_diff'].values()):.2e} "
        f"vars={res['var_names']}")
    return res


def validation_table(case, st, outdir, calib):
    rows = []
    for vt, g in st.groupby("source", sort=False):
        r = dict(case=case, source=vt, n_surrogates=len(g), phi=calib[vt]["params"]["phi"],
                 s_lat=calib[vt]["params"]["s_lat"], s_lon=calib[vt]["params"]["s_lon"])
        ok = True
        for k in ("var", "r1", "tau", "L_lat", "L_lon"):
            src = float(g[f"src_{k}"].iloc[0])
            e_ind = np.array([stat_err(k, v, src) for v in g[f"sur_{k}"]], dtype=float)
            e_ind = np.where(np.isfinite(e_ind), e_ind, np.inf)
            sur = float(np.nanmean(g[f"sur_{k}"])) if np.isfinite(g[f"sur_{k}"]).any() else np.nan
            r.update({f"{k}_src": src, f"{k}_sur": sur, f"{k}_relerr": stat_err(k, sur, src),
                      f"{k}_maxabs_relerr": float(np.abs(e_ind).max())})
            if k != "tau":
                ok &= bool(np.isfinite(r[f"{k}_relerr"]) and abs(r[f"{k}_relerr"]) <= TOL_PASS)
                r[f"{k}_frac_within"] = float((np.abs(e_ind) <= TOL_PASS).mean())
        r["pass"] = ok
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / f"validation_{case}.csv", index=False)
    cols = ["source", "n_surrogates", "var_src", "var_sur", "r1_src", "r1_sur", "tau_src", "tau_sur",
            "L_lat_src", "L_lat_sur", "L_lon_src", "L_lon_sur"]
    lines = [f"### {case}", "", "| " + " | ".join(cols + ["max ind. rel. err (var/r1/Llat/Llon)", "pass"]) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for _, r in t.iterrows():
        vals = [str(r[c]) if c in ("source", "n_surrogates") else f"{r[c]:.4g}" for c in cols]
        mx = "/".join(f"{100 * r[f'{k}_maxabs_relerr']:.1f}%" for k in ("var", "r1", "L_lat", "L_lon"))
        lines.append("| " + " | ".join(vals + [mx, "yes" if r["pass"] else "NO"]) + " |")
    (outdir / f"validation_{case}.md").write_text("\n".join(lines) + "\n")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--max-lag", type=int, default=60)
    a = ap.parse_args()
    t0 = time.time()
    st = run_case(a.case, a.outdir, list(range(a.draws)), a.k, a.nproc, a.max_lag,
                  log=lambda s: print(s, flush=True))
    print(f"[{a.case}] done: {len(st)} files, {st.bytes.sum() / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
