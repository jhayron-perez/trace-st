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

Night 6 (opt-in, --spec night6 [--mode exp1|exp2|both] [--draws 20] [--draws2 5] [--k2 32]): final variable sets;
per-cell variance map, pooled lag-1 autocorrelation and e-folding lengths estimated over the anomaly-computation
period (Debby JAS 2006 for the ERA5 fields, the 432-h files for Tb and precip; PNW the full record; Pinatubo the
whole 1991 file); stationary x_t = sd(cell) * AR(1) GRF with the per-time domain mean removed, over the loader's
time axis; (phi, s_lat, s_lon, c) calibrated on realizations as long as the statistics period, validated on
independent ones. exp1: OUT/<case>/exp1/<type>/drawNN/ (one surrogate S01<type>); exp2: OUT/<case>/exp2/drawNN/
(slots 1..32, round-robin over CASES6[case]["exp2_types"]). Seeds: surrogate_seed6 (BASE_SEED6 = 20261117).
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


def field_stats(x, tmask=None, cellmask=None, max_lag=60, remove_common=True, max_frames=60, censor=False):
    """Statistics of x (nt, ny, nx) over the time window tmask: variance (cell mean of per-cell variance over
    cellmask), pooled lag-1 autocorrelation r1 of the per-cell standardized anomalies, tau = -1/ln r1, and the
    e-folding lengths L_lat, L_lon (cells) of their spatial lag correlation after removing, at each time, the
    domain mean (remove_common: the window's common trend would otherwise correlate all cells). The spatial
    lengths use every k-th time of the window so that at most max_frames frames enter (Debby: 361 -> 61).
    censor (night 6, opt-in): an e-folding not reached within max_lag is reported as max_lag (censored) instead of
    nan, so that means over realizations stay defined when some of them do not e-fold."""
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
    L = [_efold(z, 1, max_lag), _efold(z, 2, max_lag)]
    if censor:
        L = [float(max_lag) if not np.isfinite(v) else v for v in L]
    return dict(var=float(np.nanmean(var_map[cellmask])), r1=r1, tau=tau, L_lat=L[0], L_lon=L[1]), var_map


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
              max_iter=12, max_lag=60, n_real=1, damp=0.7, log=print, field_fn=None, censor=False):
    """Fixed-point calibration of (phi, s_lat, s_lon, c) so the measured stats (mean over n_real calibration
    realizations, seeds seed + 7919 j) match target within TOL_CALIB; returns the best iterate.
    field_fn(u, c) (night 6, opt-in): the field as the case sees it from the unit field u (default: scale_field).
    censor (night 6, opt-in): field_stats(censor=True) for the measured statistics."""
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
            x = scale_field(u, tmask, var_map_w, sd_out, c, restandardize) if field_fn is None else field_fn(u, c)
            ms.append(field_stats(x, tmask, cellmask, max_lag, censor=censor)[0])
            del x
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
# night 6 (opt-in, --spec night6): final variable sets (WS10, FLUT, FSDS, blh dropped)
DEBBY_EXTRAS6 = [e for e in DEBBY_EXTRAS if e[0] != "WS10"]
PIN_EXTRAS6 = ["TMSO203", "BURDENSO403", "T050"]
PNW_EXTRAS6 = ["t2m"]
EXTRAS = {"night5": (DEBBY_EXTRAS, PIN_EXTRAS, PNW_EXTRAS), "night6": (DEBBY_EXTRAS6, PIN_EXTRAS6, PNW_EXTRAS6)}

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


def build_case(case, surrogates=None, drawdir=None, spec="night5"):
    """Case dict from the run script's own builder (extended set [+ surrogates of one draw]).
    spec="night6" (opt-in): the final-set extras (DEBBY_EXTRAS6, PIN_EXTRAS6, PNW_EXTRAS6)."""
    deb_ex, pin_ex, pnw_ex = EXTRAS[spec]
    m = load_script(CASES[case]["script"])
    names = [n for n, _ in (surrogates or [])]
    if case == "debby":
        extra = deb_ex + [(n, str(Path(drawdir) / f"{n}.nc"), n) for n in names]
        c = m.build_case_debby(extra_inputs=extra, vo_input=DEBBY_VO)
    elif case == "pinatubo":
        os.environ["TST_PINATUBO_EXTRA"] = ",".join(pin_ex + names)
        if drawdir is not None:
            os.environ["TST_PINATUBO_EXTRA_PATH"] = str(drawdir)
        try:
            c = m.build_case_pinatubo()
        finally:
            os.environ.pop("TST_PINATUBO_EXTRA", None)
            os.environ.pop("TST_PINATUBO_EXTRA_PATH", None)
    else:
        c = m.build_case_pnw(extra_vars=pnw_ex + names, extra_path=str(drawdir or PNW_IN5))
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


def link_real_extras(case, drawdir, spec="night5"):
    """Pinatubo/PNW read every extra variable from one directory: link the real extended-set extras
    (spec="night6": the final-set extras)."""
    _, pin_ex, pnw_ex = EXTRAS[spec]
    if case == "pinatubo":
        srcs = [PIN_SPA / f"{v}_199101_199112.nc" for v in pin_ex]
    elif case == "pnw":
        srcs = [PNW_IN5 / f"{v}_daily_anoms.nc" for v in pnw_ex]
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


# -----------------------------------------------------------------------------
# Night 6 (opt-in: --spec night6). Stationary AR(1) GRF matched to the statistics of the anomaly-computation
# period (Debby JAS 2006 where the files allow it, PNW full record, Pinatubo the whole 1991 file), per-time
# domain mean removed; experiment 1 (one surrogate of v per draw) and experiment 2 (ordered 32-slot sequences).
# -----------------------------------------------------------------------------
BASE_SEED6 = 20261117
EXP_IDX6 = {"exp1": 1, "exp2": 2, "val": 8, "calib": 9}
K2_MAX = 32
CASES6 = {
    "debby": dict(types=["vo", "Tb", "precip", "CAPE", "DIV200", "MSLP"],
                  exp2_types=["vo", "Tb", "precip", "CAPE", "DIV200", "MSLP"], restandardize=False, n_real=2),
    "pinatubo": dict(types=["TMSO201", "BURDENSO401", "AEROD_v", "TMH2SO401"] + PIN_EXTRAS6,
                     exp2_types=list(PIN_EXTRAS6), restandardize=True, n_real=20),
    "pnw": dict(types=["z500", "tcwv", "olr", "slhf", "sshf", "z10", "t2m"],
                exp2_types=["z500", "tcwv", "olr", "slhf", "sshf", "z10", "t2m"], restandardize=False, n_real=2),
}
# Debby ERA5 fields rebuilt over the whole JAS 2006 with the input builder's own preprocessing
# (data_processing/era5_debby/make_debby_extra_inputs.py: per-cell, per-hour standardization over JAS 2006).
# Tb and precip exist only as the 432-h FilesDebby anomaly files (their preprocessing is not on Derecho).
DEBBY_JAS6 = {"vo": "vo500700", "CAPE": "cape", "DIV200": "d200", "MSLP": "msl"}
DEBBY_CACHE = DEB_IN / "cache"


def surrogate_seed6(case, exp, draw, slot):
    """Night-6 seed: BASE_SEED6 + 1e6 case_idx + 1e4 exp (exp1 1, exp2 2, validation 8, calibration 9)
    + 100 draw + slot. exp1: slot = 1-based type index; exp2: slot 1..32; validation: draw = realization."""
    return int(BASE_SEED6 + 1000000 * CASE_IDX[case] + 10000 * EXP_IDX6[exp] + 100 * int(draw) + int(slot))


def remove_domain_mean(x):
    """Per-time domain mean (over the analysis grid cells) removed."""
    return (x - np.nanmean(x, axis=(1, 2), keepdims=True, dtype=np.float64)).astype(np.float32)


def field6(u, sd_map, c, restandardize):
    """Night-6 field from the unit AR(1) GRF u (nt, ny, nx): x = sqrt(c) sd(cell) u, per-time domain mean
    removed (stationary: one sd map for every time). Returns (values written to the file, values the case sees:
    restandardize = Pinatubo loader (x - mean_time) / std_time per cell)."""
    x = remove_domain_mean(u * (np.sqrt(c) * sd_map).astype(np.float32)[None])
    return x, (standardize_time(x) if restandardize else x)


def windowed_stats(src, n, max_lag=60, max_windows=60):
    """Mean of field_stats over non-overlapping windows of n steps of src (at most max_windows, evenly spread):
    what one file of n steps is expected to measure. Returns (stats, number of windows)."""
    nw = src.shape[0] // n
    if nw <= 1:
        return field_stats(src, None, None, max_lag, censor=True)[0], 1
    starts = np.unique(np.round(np.linspace(0, nw - 1, min(nw, max_windows))).astype(int)) * n
    ss = [field_stats(src[s:s + n], None, None, max_lag, censor=True)[0] for s in starts]
    return {k: float(np.nanmean([d[k] for d in ss])) for k in ss[0]}, len(ss)


def load_module(path):
    """Import a module from a file path, registered in sys.modules (its functions must pickle for its Pool)."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _as_analysis(da, name):
    """One loader-preprocessed field (dims incl. lat, lon, time; optional var) -> analysis arrangement."""
    if "var" not in da.dims:
        da = da.expand_dims(var=[name])
    fd, _ = analysis_arrays(da)
    return fd.isel(var=0)


def stats_period_source(case, m, vt, fd, log=print, return_times=False):
    """Source field (time, lat, lon) over the anomaly-computation period, on the analysis grid, with the case
    loader's preprocessing; returns (array float32, period dict) [night 7, opt-in return_times: (array, period, times)]."""
    an = analysis_var_name(m, case, vt)
    if case == "pnw":
        ts0 = m.TIME_SLICE
        m.TIME_SLICE = None                     # whole record of the anomaly files (statistics only)
        try:
            if vt in m.VAR_NAME_MAP:
                da = m.load_full_data_for_vars([vt])
            else:
                da = m.tst.data_io.load_full_data(path_files=str(PNW_IN5), varstems=[vt], list_names_vars=[an],
                                                  time_slice=None, join="inner")
        finally:
            m.TIME_SLICE = ts0
        da = _as_analysis(m.regrid_to_2p5(da), an)
        how = "full record of the anomaly file (loader TIME_SLICE overridden to None; loader regrid)"
    elif case == "debby" and vt in DEBBY_JAS6:
        mk = load_module(REPO / "data_processing/era5_debby/make_debby_extra_inputs.py")
        field = DEBBY_JAS6[vt]
        s = mk.SPEC[field]
        probe = DEBBY_CACHE / (f"{s['code']}_20060701.nc" if s["kind"] == "pl" else f"{s['code']}_200607.nc")
        cache = DEBBY_CACHE if probe.exists() else None
        log(f"  [{vt}] JAS 2006 {field} from {'cache ' + str(cache) if cache else 'RDA ds633.0'}")
        da = mk.standardize(mk.load_jas(field, 2006, nproc=3, cache=cache), "hour")
        da = da.drop_vars([k for k in da.coords if k not in ("time", "latitude", "longitude")]).astype("float32")
        da = da.rename({"latitude": "lat", "longitude": "lon"})
        da = da.assign_coords(time=da.time.dt.round("1H"))
        da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
        da = _as_analysis(da, an)
        how = f"JAS 2006 rebuilt with make_debby_extra_inputs.build preprocessing (field {field}, stdz hour)"
    else:
        da = fd.sel(var=an)
        how = "whole period read by the case loader"
    t = pd.to_datetime(da.time.values)
    x = np.asarray(da.transpose("time", "lat", "lon").values, dtype=np.float32)
    if x.shape[1:] != (fd.sizes["lat"], fd.sizes["lon"]) or not (
            np.allclose(da.lat.values, fd.lat.values) and np.allclose(da.lon.values % 360, fd.lon.values % 360)):
        raise ValueError(f"{case} {vt}: statistics-period grid differs from the analysis grid")
    period = dict(start=str(t[0]), end=str(t[-1]), n_time=int(len(t)), how=how)
    if how.startswith("JAS") or case == "pnw":   # check: the loader's field equals the period field on its times
        idx = pd.DatetimeIndex(t).get_indexer(pd.to_datetime(fd.time.values))
        ok = idx >= 0
        ref = np.asarray(fd.sel(var=an).transpose("time", "lat", "lon").values, dtype=np.float32)[ok]
        period["loader_times_found"] = int(ok.sum())
        period["max_abs_diff_vs_loader"] = float(np.nanmax(np.abs(x[idx[ok]] - ref)))
    return (x, period, t) if return_times else (x, period)


def _same_stats(a, b):
    """Cached calibration still valid: same target statistics (nan == nan)."""
    return bool(a) and set(a) == set(b) and all(np.isclose(float(a[k]), float(b[k]), rtol=1e-9, equal_nan=True)
                                                 for k in b)


def _calib_task6(task):
    ti, vt, nt = task
    g = _G
    tgt = g["src"][vt]["target"]
    seed = surrogate_seed6(g["case"], "calib", 0, ti + 1)
    ll = np.sqrt(np.nan_to_num(tgt["L_lat"], nan=g["max_lag"]) * np.nan_to_num(tgt["L_lon"], nan=g["max_lag"]))
    n_real = int(np.clip(np.ceil(g["n_real"] * ll / 10.0), g["n_real"], max(g["n_real"], 6)))
    sd, rs = g["sd"][vt], g["restandardize"]
    log = lambda s: print(f"[{vt}] {s}", flush=True)
    log(f"calibrate (seed {seed} + 7919 j, {n_real} realizations of {nt} steps): target {tgt}")
    params, hist = calibrate(tgt, (nt,) + g["yx"], np.ones(nt, bool), None, None, g["cellmask"][vt], seed,
                             g["periodic"], rs, max_lag=g["max_lag"], n_real=n_real, log=log, censor=True,
                             field_fn=lambda u, c: field6(u, sd, c, rs)[1])
    return vt, dict(params=params, seed=seed, seed_formula="surrogate_seed6(case, 'calib', 0, type_idx) + 7919 j",
                    n_real=n_real, history=hist)


def _val_task6(task):
    ti, vt, nt, j = task
    g = _G
    p = g["calib"][vt]["params"]
    seed = surrogate_seed6(g["case"], "val", j, ti + 1)
    u = unit_grf_ar1(nt, *g["yx"], p["phi"], p["s_lat"], p["s_lon"], np.random.default_rng(seed),
                     periodic_lon=g["periodic"], pad=(p["pad_lat"], p["pad_lon"]))
    xs = field6(u, g["sd"][vt], p["c"], g["restandardize"])[1]
    m, _ = field_stats(xs, None, g["cellmask"][vt], g["max_lag"], censor=True)
    return dict(source=vt, realization=j, seed=seed, n_time=nt, **{f"sur_{k}": v for k, v in m.items()})


def _gen_task6(task):
    exp, draw, slot, vt, ti = task
    g = _G
    case = g["case"]
    p = g["calib"][vt]["params"]
    if exp == "exp1":
        name, seed = surrogate_name(case, 1, vt), surrogate_seed6(case, "exp1", draw, ti + 1)
        d = Path(g["outcase"]) / "exp1" / vt / f"draw{draw:02d}"
    else:
        name, seed = surrogate_name(case, slot, vt), surrogate_seed6(case, "exp2", draw, slot)
        d = Path(g["outcase"]) / "exp2" / f"draw{draw:02d}"
    t0 = time.time()
    u = unit_grf_ar1(*g["shape"], p["phi"], p["s_lat"], p["s_lon"], np.random.default_rng(seed),
                     periodic_lon=g["periodic"], pad=(p["pad_lat"], p["pad_lon"]))
    x, _ = field6(u, g["sd"][vt], p["c"], g["restandardize"])
    del u
    q, scale = quantize(x)
    xq = q.astype(np.float32) * scale                                       # values read back from the file
    xs = standardize_time(xq) if g["restandardize"] else xq                 # as the case sees them
    ma, _ = field_stats(xs, None, g["cellmask"][vt], g["max_lag"], censor=True)  # whole loader time axis
    mw, _ = field_stats(xs, g["tmask"], g["cellmask"][vt], g["max_lag"], censor=True)  # analysis window (N samples)
    dm = float(np.max(np.abs(np.mean(xs, axis=(1, 2), dtype=np.float64))))
    path = d / file_name(case, name)
    write_surrogate(path, name, xq, g["lay"],
                    dict(long_name=f"GRF surrogate of {vt} (TraCE-ST revision-1 night 6 {exp})", source_variable=vt,
                         seed=seed, phi=p["phi"], s_lat_cells=p["s_lat"], s_lon_cells=p["s_lon"],
                         variance_factor=p["c"], stationary=1, domain_mean_removed=1,
                         generator="data_processing/surrogates/make_grf_surrogates.py --spec night6"))
    row = dict(case=case, exp=exp, draw=draw, slot=slot, name=name, source=vt, seed=seed, file=str(path),
               bytes=path.stat().st_size, sha256=sha256(path), secs=round(time.time() - t0, 1),
               max_abs_domain_mean=dm, **{f"sur_{k}": v for k, v in ma.items()},
               **{f"win_{k}": v for k, v in mw.items()})
    if (exp, draw, vt if exp == "exp1" else None) in g["keep"]:
        np.save(d / f".{name}_analysis.npy", xs)
    return row


def run_args6(case, out, exp, draw, vtype=None, k=0):
    """Data arguments a run script needs: final set + the exp1 surrogate of vtype (draw) or exp2 slots 1..k."""
    out = Path(out)
    if exp == "exp1":
        d = out / "exp1" / vtype / f"draw{draw:02d}"
        names = [surrogate_name(case, 1, vtype)]
    else:
        d = out / "exp2" / f"draw{draw:02d}"
        names = [surrogate_name(case, s, type_for_slot(CASES6[case]["exp2_types"], s)) for s in range(1, k + 1)]
    if case == "debby":
        ex = DEBBY_EXTRAS6 + [(n, str(d / f"{n}.nc"), n) for n in names]
        return (f"--vo-input {DEBBY_VO[0]}:{DEBBY_VO[1]} --extra-inputs "
                + " ".join(f"{a}={p}:{v}" for a, p, v in ex))
    if case == "pinatubo":
        return "--extra-vars " + " ".join(PIN_EXTRAS6 + names) + (f" --extra-path {d}" if names else "")
    return "--extra-vars " + " ".join(PNW_EXTRAS6 + names) + f" --extra-path {d if names else PNW_IN5}"


def loader_check6(case, drawdir, sur, log=print):
    """The run script's own loader reads drawdir (final set + surrogates sur [(name, type)]): same values as
    generated (hidden .<name>_analysis.npy written by _gen_task6, removed afterwards)."""
    m, c = build_case(case, sur, drawdir, spec="night6")
    fd, _ = analysis_arrays(c["full_data"])
    res = dict(case=case, dir=str(drawdir), var_names=list(map(str, c["var_names"])), n_vars=len(c["var_names"]),
               nan_frac={}, max_abs_diff={})
    for n, _ in sur:
        an = n.upper() if case == "pnw" else n
        got = np.asarray(fd.sel(var=an).values, dtype=np.float64)
        ref = np.load(Path(drawdir) / f".{n}_analysis.npy").astype(np.float64)
        res["nan_frac"][an] = float(np.isnan(got).mean())
        res["max_abs_diff"][an] = float(np.nanmax(np.abs(got - ref)))
        (Path(drawdir) / f".{n}_analysis.npy").unlink()
    res["ok"] = bool(max(res["max_abs_diff"].values()) < 1e-3 and max(res["nan_frac"].values()) == 0.0)
    log(f"[{case}] loader check {drawdir}: ok={res['ok']} max|diff|={max(res['max_abs_diff'].values()):.2e} "
        f"n_vars={res['n_vars']}")
    return res


def validation_table6(case, src, val, st, outdir, calib):
    """S20 rows per type: source (anomaly period) vs independent validation realizations of the same length
    (pass: mean within TOL_PASS for var, r1, L_lat, L_lon); per-file check vs the source's expected per-file value
    (windowed_stats over the loader time axis); matched-window source statistics (analysis window, information)."""
    keys = ("var", "r1", "tau", "L_lat", "L_lon")
    rows = []
    for vt, info in src.items():
        g = val[val.source == vt]
        r = dict(case=case, source=vt, period=f"{info['period']['start']}..{info['period']['end']}",
                 n_time=info["period"]["n_time"], n_val=len(g), **{k: calib[vt]["params"][k] for k in
                                                                     ("phi", "s_lat", "s_lon", "c")})
        ok = True
        for k in keys:
            s = float(info["target"][k])
            e_ind = np.array([stat_err(k, v, s) for v in g[f"sur_{k}"]], dtype=float)
            e_ind = np.where(np.isfinite(e_ind), e_ind, np.inf)
            sur = float(np.nanmean(g[f"sur_{k}"])) if np.isfinite(g[f"sur_{k}"]).any() else np.nan
            r.update({f"{k}_src": s, f"{k}_sur": sur, f"{k}_relerr": stat_err(k, sur, s),
                      f"{k}_maxabs_relerr": float(np.abs(e_ind).max())})
            if k != "tau":
                ok &= bool(np.isfinite(r[f"{k}_relerr"]) and abs(r[f"{k}_relerr"]) <= TOL_PASS)
        r["pass"] = ok
        if st is not None:
            f = st[st.source == vt]
            okf = True
            for k in keys:
                s = float(info["target_fileaxis"][k])
                e_ind = np.array([stat_err(k, v, s) for v in f[f"sur_{k}"]], dtype=float)
                sur = float(np.nanmean(f[f"sur_{k}"])) if np.isfinite(f[f"sur_{k}"]).any() else np.nan
                r.update({f"file_{k}_src": s, f"file_{k}_sur": sur, f"file_{k}_relerr": stat_err(k, sur, s),
                          f"file_{k}_maxabs_relerr": float(np.nanmax(np.abs(e_ind))) if len(e_ind) else np.nan})
                if k != "tau":
                    okf &= bool(np.isfinite(r[f"file_{k}_relerr"]) and abs(r[f"file_{k}_relerr"]) <= TOL_PASS)
            r["n_files"], r["file_pass"] = len(f), okf
        for k in keys:
            r[f"window_{k}_src"] = float(info["target_window"][k])
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / f"validation_{case}.csv", index=False)
    pct = lambda v: f"{100 * v:+.1f}%" if np.isfinite(v) else "nan"
    lines = [f"### {case} (night 6: statistics over the anomaly-computation period; stationary AR(1) GRF, "
             f"per-time domain mean removed)", "",
             "Source = statistics over the period; surrogate = mean of independent validation realizations of the "
             "same length (pass: |mean rel. err| <= 10 % for var, r1, L_lat, L_lon). Files = production files over "
             "the loader time axis vs the source's mean over windows of that length (information only: the source "
             "is not stationary, so short windows of it are not realizations of one AR(1) process). L not reached "
             "within max_lag is censored at max_lag.", "",
             "| type | period (n) | var src/sur | r1 src/sur | tau src/sur | L_lat src/sur | L_lon src/sur "
             "| max ind. err var/r1/Llat/Llon | pass | files: mean err var/r1/Llat/Llon (n) | files within 10 % (info) "
             "| matched window src var/r1/Llat/Llon |",
             "|" + "---|" * 12]
    for _, r in t.iterrows():
        cells = [r.source, f"{r.period} ({r.n_time})"]
        cells += [f"{r[f'{k}_src']:.4g} / {r[f'{k}_sur']:.4g}" for k in keys]
        cells.append("/".join(f"{100 * r[f'{k}_maxabs_relerr']:.1f}%" for k in ("var", "r1", "L_lat", "L_lon")))
        cells.append("yes" if r["pass"] else "NO")
        if "file_pass" in t.columns and pd.notna(r.get("file_pass")):
            cells.append("/".join(pct(r[f"file_{k}_relerr"]) for k in ("var", "r1", "L_lat", "L_lon"))
                         + f" ({int(r.n_files)})")
            cells.append("yes" if r["file_pass"] else "NO")
        else:
            cells += ["-", "-"]
        cells.append("/".join(f"{r[f'window_{k}_src']:.3g}" for k in ("var", "r1", "L_lat", "L_lon")))
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    (outdir / f"validation_{case}.md").write_text("\n".join(lines) + "\n")
    return t


def run_case6(case, outdir, mode, draws1, draws2, k2, nproc, n_val=3, max_lag=60, log=print):
    cfg = CASES6[case]
    out = Path(outdir) / case
    out.mkdir(parents=True, exist_ok=True)
    log(f"[{case}] night 6: loading the final set via the run script's builder")
    m, c = build_case(case, spec="night6")
    fd, periodic = analysis_arrays(c["full_data"])
    tmask, win = analysis_window(m, case, c, fd.time.values)
    shape = (fd.sizes["time"], fd.sizes["lat"], fd.sizes["lon"])
    log(f"[{case}] analysis grid {shape}, periodic lon {periodic}, loader axis {fd.time.values[0]}..{fd.time.values[-1]}, "
        f"window {win} ({int(tmask.sum())} steps), vars {list(map(str, c['var_names']))}")
    calib_path = out / "calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    calib.pop("_meta", None)
    calib.pop("sources", None)
    src, sd, cellmask, todo = {}, {}, {}, []
    for ti, vt in enumerate(cfg["types"]):
        t0 = time.time()
        x, period = stats_period_source(case, m, vt, fd, log)
        tgt, vmap = field_stats(x, None, None, max_lag, censor=True)
        tgt_axis, n_w = windowed_stats(x, shape[0], max_lag)
        cv = np.asarray(fd.sel(var=analysis_var_name(m, case, vt)).values, dtype=np.float32)
        tgt_win = field_stats(cv, tmask, None, max_lag, censor=True)[0]
        del x, cv
        sd[vt] = np.sqrt(np.where(np.isfinite(vmap), vmap, 1.0)).astype(np.float32)
        cellmask[vt] = np.isfinite(vmap)
        src[vt] = dict(period=period, target=tgt, target_fileaxis=tgt_axis, n_fileaxis_windows=n_w,
                       target_window=tgt_win)
        log(f"[{case}] {vt}: period {period} ({time.time() - t0:.0f} s)\n    target {tgt}\n    file-axis {tgt_axis} "
            f"({n_w} windows)\n    matched window {tgt_win}")
        if vt not in calib or not _same_stats(calib[vt].get("target"), tgt):
            todo.append((ti, vt, period["n_time"]))
    meta = dict(spec="night6", window=win, shape=shape, periodic=periodic, loader_axis=[str(fd.time.values[0]),
                str(fd.time.values[-1])], var_names=list(map(str, c["var_names"])))
    _G.clear()
    _G.update(case=case, src=src, sd=sd, cellmask=cellmask, yx=shape[1:], periodic=periodic, max_lag=max_lag,
              restandardize=cfg["restandardize"], n_real=cfg["n_real"])
    if todo:
        log(f"[{case}] calibrating {[t[1] for t in todo]} with {min(nproc, len(todo))} processes")
        with mp.get_context("fork").Pool(min(nproc, len(todo))) as pool:
            for vt, res in pool.imap_unordered(_calib_task6, todo):
                calib[vt] = dict(target=src[vt]["target"], **res)
                calib_path.write_text(json.dumps(dict(calib, _meta=meta), indent=1, default=float))
    calib_path.write_text(json.dumps(dict(calib, _meta=meta, sources=src), indent=1, default=float))
    _G["calib"] = calib
    vtasks = [(ti, vt, src[vt]["period"]["n_time"], j) for ti, vt in enumerate(cfg["types"]) for j in range(n_val)]
    log(f"[{case}] {len(vtasks)} validation realizations")
    with mp.get_context("fork").Pool(min(nproc, len(vtasks))) as pool:
        val = pd.DataFrame(list(pool.imap_unordered(_val_task6, vtasks))).sort_values(["source", "realization"])
    val.to_csv(out / "validation_realizations.csv", index=False)
    vt_tab = validation_table6(case, src, val, None, Path(outdir), calib)
    log(Path(outdir, f"validation_{case}.md").read_text())
    if not vt_tab["pass"].all():
        log(f"[{case}] VALIDATION FAILED for {list(vt_tab.source[~vt_tab['pass']])}: no production files written")
        return None
    lay = native_layout(case, dict(time=fd.time.values, lat=fd.lat.values, lon=fd.lon.values), periodic,
                        time_slice=getattr(m, "TIME_SLICE", None) if case == "pnw" else None)
    tasks, keep = [], set()
    if mode in ("exp1", "both"):
        for ti, vt in enumerate(cfg["types"]):
            for d in draws1:
                dd = out / "exp1" / vt / f"draw{d:02d}"
                dd.mkdir(parents=True, exist_ok=True)
                link_real_extras(case, dd, spec="night6")
                tasks.append(("exp1", d, 1, vt, ti))
        keep.add(("exp1", draws1[0], cfg["types"][0]))
    if mode in ("exp2", "both"):
        for d in draws2:
            dd = out / "exp2" / f"draw{d:02d}"
            dd.mkdir(parents=True, exist_ok=True)
            link_real_extras(case, dd, spec="night6")
            for s in range(1, k2 + 1):
                vt = type_for_slot(cfg["exp2_types"], s)
                tasks.append(("exp2", d, s, vt, cfg["types"].index(vt)))
        keep.add(("exp2", draws2[0], None))
    _G.update(shape=shape, tmask=tmask, outcase=str(out), lay=lay, keep=keep)
    log(f"[{case}] generating {len(tasks)} surrogates with {nproc} processes")
    del c, fd
    rows = []
    with mp.get_context("fork").Pool(nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(_gen_task6, tasks)):
            rows.append(r)
            if (i + 1) % 20 == 0:
                log(f"  {i + 1}/{len(tasks)}")
    st = pd.DataFrame(rows).sort_values(["exp", "source", "draw", "slot"])
    st.to_csv(out / "surrogate_stats.csv", index=False)
    st[["case", "exp", "draw", "slot", "name", "source", "seed"]].to_csv(out / "seeds.csv", index=False)
    with open(out / "sha256.txt", "w") as f:
        for _, r in st.iterrows():
            f.write(f"{r.sha256}  {Path(r.file).relative_to(out)}\n")
    checks = {}
    if mode in ("exp1", "both"):
        vt = cfg["types"][0]
        checks["exp1"] = loader_check6(case, out / "exp1" / vt / f"draw{draws1[0]:02d}",
                                       [(surrogate_name(case, 1, vt), vt)], log)
    if mode in ("exp2", "both"):
        sur = [(surrogate_name(case, s, type_for_slot(cfg["exp2_types"], s)), type_for_slot(cfg["exp2_types"], s))
               for s in range(1, k2 + 1)]
        checks["exp2"] = loader_check6(case, out / "exp2" / f"draw{draws2[0]:02d}", sur, log)
    (out / "loader_check.json").write_text(json.dumps(checks, indent=1))
    with open(out / "exp1_args.txt", "w") as f:
        f.write("# type|draw|data arguments (final set + the exp1 surrogate of type, draw)\n")
        for vt in cfg["types"]:
            for d in draws1:
                f.write(f"{vt}|{d}|{run_args6(case, out, 'exp1', d, vtype=vt)}\n")
    with open(out / "exp2_args.txt", "w") as f:
        f.write("# K|draw|data arguments (final set + exp2 slots 1..K of draw; K=0 = final set)\n")
        for k in (0, 1, 2, 4, 8, 16, 24, 32):
            for d in draws2:
                f.write(f"{k}|{d}|{run_args6(case, out, 'exp2', d, k=k)}\n")
    validation_table6(case, src, val, st, Path(outdir), calib)
    return st, checks


# -----------------------------------------------------------------------------
# Night 7 (opt-in: --spec night7). Linkless surrogate systems for the null of approach A (brief 2026-11-18):
# every final-set variable of a case is replaced by a surrogate with its night-6 per-cell variance map, memory and
# spatial scale and the lag-0 correlation matrix R of the real final set, but no lagged cross-terms:
#   variant "A": independent unit-variance white noises are mixed with a Cholesky factor L of a matrix C in the
#                innovation (in Fourier space, before each variable's own smoothing kernel and AR(1)); C is iterated
#                from the analytic attenuation R / f (f = spatial kernel overlap x AR(1) overlap) until the measured
#                pooled lag-0 correlation of the generated system equals R (|diff| < 0.03 aimed, 0.10 required);
#   variant "indep": R = identity, same base noise seeds (common draws);
#   "addK": ordered sequences of independent linkless fields (the "indep" generator, one per slot) that are added to
#           the real final set (night-6 exp2 file naming, --extra-vars/--extra-path).
# R = pooled correlation, over the cells where every variable is finite and over the times common to all variables
# of the case, of the per-cell standardized anomalies with the per-time domain mean removed (same estimator for
# the source and for the generated systems).
# -----------------------------------------------------------------------------
BASE_SEED7 = 20261118
VAR_IDX7 = {"A": 1, "indep": 2, "addK": 3, "syn": 4, "val": 8, "calib": 9}
K_ADD7 = 16
R_TOL_CALIB7 = 0.03
R_TOL_PASS7 = 0.10
GROUP_THR7 = 0.7
NT_VAL_CAP7 = 4000
# Debby: file names the run script reads from --system-path (Tb, precip) or via --vo-input / --extra-inputs
DEBBY_FILES7 = {"Tb": ("BT_anomalies_TC_argon.nc4", "Tb", True), "precip": ("P_anomalies_TC_argon2.nc4", "precipitation", True),
                "vo": ("vo.nc", "vo", False), "CAPE": ("CAPE.nc", "CAPE", False), "DIV200": ("DIV200.nc", "DIV200", False),
                "MSLP": ("MSLP.nc", "MSLP", False)}


def _md7(df):
    """Pipe-table markdown of a DataFrame (index as first column)."""
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [str(v) for v in r.values]) + " |")
    return "\n".join(lines)


def surrogate_seed7(case, variant, draw, slot):
    """Night-7 seed: BASE_SEED7 + 1e6 case_idx + 1e4 variant (A 1, indep 2, addK 3, syn 4, val 8, calib 9)
    + 100 draw + slot. The "indep" system uses the "A" base-noise seeds (common draws; slot = type index + 1)."""
    return int(BASE_SEED7 + 1000000 * CASE_IDX[case] + 10000 * VAR_IDX7[variant] + 100 * int(draw) + int(slot))


def nearest_corr7(C):
    """Nearest symmetric positive semi-definite matrix with unit diagonal (eigenvalue clipping, rescaling)."""
    C = 0.5 * (np.asarray(C, dtype=float) + np.asarray(C, dtype=float).T)
    w, v = np.linalg.eigh(C)
    C = (v * np.clip(w, 1e-6, None)) @ v.T
    d = 1.0 / np.sqrt(np.diag(C))
    C = C * d[:, None] * d[None, :]
    np.fill_diagonal(C, 1.0)
    return C


def mixing_factor7(C):
    """Lower Cholesky factor L (rows of unit norm) of the nearest correlation matrix of C."""
    return np.linalg.cholesky(nearest_corr7(C))


def corr_groups7(R, names, thr=GROUP_THR7):
    """Connected components of the graph |R_ij| >= thr (list of lists of names; singletons included)."""
    n = len(names)
    seen, out = set(), []
    for i in range(n):
        if i in seen:
            continue
        comp, stack = [], [i]
        seen.add(i)
        while stack:
            a = stack.pop()
            comp.append(a)
            for b in range(n):
                if b not in seen and abs(R[a, b]) >= thr:
                    seen.add(b)
                    stack.append(b)
        out.append([names[j] for j in sorted(comp)])
    return out


def attenuation7(p_a, p_b):
    """Analytic lag-0 correlation factor of two fields built from the same white noise with their own spatial
    Gaussian kernels (std s cells) and AR(1) phi: sqrt(2 s_a s_b / (s_a^2 + s_b^2)) per axis x
    sqrt((1 - phi_a^2)(1 - phi_b^2)) / (1 - phi_a phi_b)."""
    f = 1.0
    for k in ("s_lat", "s_lon"):
        sa, sb = max(p_a[k], 1e-3), max(p_b[k], 1e-3)
        f *= float(np.sqrt(2.0 * sa * sb / (sa ** 2 + sb ** 2)))
    return f * float(np.sqrt((1 - p_a["phi"] ** 2) * (1 - p_b["phi"] ** 2)) / (1 - p_a["phi"] * p_b["phi"]))


def system_unit_fields7(nt, yx, params, L, rngs, periodic_lon=False, chunk=48):
    """Unit-variance AR(1) GRFs of a linkless system: list of (nt, ny, nx) float32, one per variable. Variable v has
    its own phi, s_lat, s_lon (params[v]); its innovation is sum_j L[v, j] w_j, w_j independent white noise
    (rngs[j]) on a common padded grid, filtered with v's kernel (as unit_grf_ar1). L = identity: independent fields."""
    V, (ny, nx) = len(params), yx
    py = max(int(p["pad_lat"]) for p in params)
    px = 0 if periodic_lon else max(int(p["pad_lon"]) for p in params)
    NY, NX = ny + 2 * py, nx + 2 * px
    fy = np.fft.fftfreq(NY)[:, None]
    Gs = []
    for p in params:
        G = np.exp(-2.0 * np.pi ** 2 * (p["s_lat"] ** 2 * fy ** 2 + p["s_lon"] ** 2 * np.fft.rfftfreq(NX)[None, :] ** 2))
        var = np.mean(np.exp(-4.0 * np.pi ** 2 * (p["s_lat"] ** 2 * fy ** 2
                                                  + p["s_lon"] ** 2 * np.fft.fftfreq(NX)[None, :] ** 2)))
        Gs.append(G / np.sqrt(var))
    a = [float(p["phi"]) for p in params]
    b = [float(np.sqrt(max(1.0 - p["phi"] ** 2, 0.0))) for p in params]
    out = [np.empty((nt, ny, nx), dtype=np.float32) for _ in range(V)]
    prev = [None] * V
    for t0 in range(0, nt, chunk):
        c = min(chunk, nt - t0)
        F = np.stack([np.fft.rfft2(rng.standard_normal((c, NY, NX))) for rng in rngs])
        for v in range(V):
            e = np.fft.irfft2(np.tensordot(L[v], F, axes=(0, 0)) * Gs[v], s=(NY, NX))[:, py:py + ny, px:px + nx]
            for i in range(c):
                prev[v] = e[i] if prev[v] is None else a[v] * prev[v] + b[v] * e[i]
                out[v][t0 + i] = prev[v]
        del F
    return out


def zfield7(x, cmask, sel=None):
    """Per-cell standardized anomalies (over the whole time axis of x), restricted to the times sel and the cells
    cmask (nan elsewhere), with the per-time domain mean over cmask removed. float32."""
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x, axis=0, dtype=np.float64)
    sd = np.nanstd(x, axis=0, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ((x - mu) / np.where(sd > 0, sd, np.nan)).astype(np.float32)
    if sel is not None:
        z = z[sel]
    z[:, ~cmask] = np.nan
    z -= np.nanmean(z, axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    return z


def corr_matrix7(zs):
    """Pooled (over cells and times) correlation matrix of the fields zs (same shape, nan = excluded)."""
    V = len(zs)
    ss = [float(np.nansum(z * z, dtype=np.float64)) for z in zs]
    R = np.eye(V)
    for i in range(V):
        for j in range(i + 1, V):
            R[i, j] = R[j, i] = float(np.nansum(zs[i] * zs[j], dtype=np.float64)) / np.sqrt(ss[i] * ss[j])
    return R


def source_corr7(xs, times, cmask):
    """R of the source fields xs (list of (n_v, ny, nx), times[v] the DatetimeIndex of each) over the common times."""
    common = times[0]
    for t in times[1:]:
        common = common.intersection(t)
    zs = [zfield7(x, cmask, sel=pd.DatetimeIndex(t).get_indexer(common)) for x, t in zip(xs, times)]
    return corr_matrix7(zs), len(common)


def measure_system7(nt, yx, params, L, sd_maps, cs, restandardize, periodic, cmask, seed, nR):
    """One realization of the system: (fields as written, fields as the case sees them, R measured on the first nR
    steps)."""
    rngs = [np.random.default_rng(seed + j) for j in range(len(params))]
    us = system_unit_fields7(nt, yx, params, L, rngs, periodic)
    xw, xs = [], []
    for u, sd, c in zip(us, sd_maps, cs):
        a, b = field6(u, sd, c, restandardize)
        xw.append(a)
        xs.append(b)
    del us
    R = corr_matrix7([zfield7(x[:nR], cmask) for x in xs])
    return xw, xs, R


def calibrate_mixing7(R, params, yx, cfg7, log=print, max_iter=12, damp=0.7):
    """Iterate C until the measured lag-0 correlation of the generated system equals R. Returns (C, L, history)."""
    V = len(params)
    C = np.eye(V)
    for i in range(V):
        for j in range(i + 1, V):
            C[i, j] = C[j, i] = float(np.clip(R[i, j] / attenuation7(params[i], params[j]), -0.995, 0.995))
    hist, best = [], None
    for it in range(max_iter):
        L = mixing_factor7(C)
        Rm = np.zeros((V, V))
        for j in range(cfg7["n_rep"]):
            seed = surrogate_seed7(cfg7["case"], "calib", 90 + j, 1)
            Rm += measure_system7(cfg7["nt_cal"], yx, params, L, cfg7["sd"], cfg7["c"], cfg7["restandardize"],
                                  cfg7["periodic"], cfg7["cmask"], seed, cfg7["nt_cal"])[2]
        Rm /= cfg7["n_rep"]
        err = R - Rm
        worst = float(np.abs(err[np.triu_indices(V, 1)]).max()) if V > 1 else 0.0
        hist.append(dict(it=it, worst=worst, C=C.tolist(), R_measured=Rm.tolist()))
        log(f"    mixing it {it}: max |R - R_measured| = {worst:.4f}")
        if best is None or worst < best[0]:
            best = (worst, C.copy(), L.copy())
        if worst < R_TOL_CALIB7:
            break
        C = nearest_corr7(C) + damp * err        # anti-windup: update from the feasible (projected) matrix
        np.fill_diagonal(C, 1.0)
        C = np.clip(C, -0.995, 0.995)
        np.fill_diagonal(C, 1.0)
    return best[1], best[2], hist


def file_spec7(case, vt):
    """(file name, variable name, rename latitude/longitude -> lat/lon) of a system variable, as the run script
    reads it from --system-path (Debby: vo and the extras via --vo-input / --extra-inputs)."""
    if case == "debby":
        return DEBBY_FILES7[vt]
    return file_name(case, vt), vt, False


def write_surrogate7(path, name, x_analysis, lay, attrs, rename_ll=False):
    """write_surrogate with an optional latitude/longitude -> lat/lon rename (Debby Tb and precip files)."""
    xn = x_analysis[lay["it"]][:, lay["iy"]][:, :, lay["ix"]]
    order = [{lay["tdim"]: 0, lay["ydim"]: 1, lay["xdim"]: 2}[d] for d in lay["dims"]]
    q, scale = quantize(np.transpose(xn, order))
    da = xr.DataArray(q.astype(np.float32) * scale, dims=lay["dims"], coords=lay["coords"], name=name, attrs=attrs)
    if rename_ll:
        da = da.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in da.dims})
    enc = {name: dict(dtype="int16", scale_factor=float(scale), add_offset=0.0, _FillValue=np.int16(-32768),
                      zlib=True, complevel=1, shuffle=True)}
    tmp = Path(str(path) + ".tmp")
    da.to_dataset().to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)
    return scale


def system_args7(case, d):
    """Data arguments of a run on the system directory d (Tb, precip, vo, extras from the system)."""
    d = Path(d)
    if case == "debby":
        ex = [("CAPE", "CAPE"), ("DIV200", "DIV200"), ("MSLP", "MSLP")]
        return (f"--system-path {d} --vo-input {d / 'vo.nc'}:vo --extra-inputs "
                + " ".join(f"{a}={d / DEBBY_FILES7[a][0]}:{DEBBY_FILES7[a][1]}" for a, _ in ex))
    if case == "pinatubo":
        return f"--system-path {d} --extra-vars " + " ".join(PIN_EXTRAS6)
    return f"--system-path {d} --extra-vars " + " ".join(PNW_EXTRAS6) + f" --extra-path {d}"


def addk_args7(case, d, k):
    """Data arguments of a run on the final set + the first k addK fields of directory d (night-6 exp2 mechanism)."""
    d = Path(d)
    names = [surrogate_name(case, s, type_for_slot(CASES6[case]["exp2_types"], s)) for s in range(1, k + 1)]
    if case == "debby":
        ex = DEBBY_EXTRAS6 + [(n, str(d / f"{n}.nc"), n) for n in names]
        return (f"--vo-input {DEBBY_VO[0]}:{DEBBY_VO[1]} --extra-inputs "
                + " ".join(f"{a}={p}:{v}" for a, p, v in ex))
    if case == "pinatubo":
        return "--extra-vars " + " ".join(PIN_EXTRAS6 + names) + (f" --extra-path {d}" if names else "")
    return "--extra-vars " + " ".join(PNW_EXTRAS6 + names) + f" --extra-path {d if names else PNW_IN5}"


def t1_tables7(outroot, case, names, R, n_common, info):
    """T1_corr_<case>.csv and the combined T1_groups.md of the cases present."""
    outroot = Path(outroot)
    pd.DataFrame(R, index=names, columns=names).to_csv(outroot / f"T1_corr_{case}.csv", float_format="%.4f")
    (outroot / f"T1_groups_{case}.json").write_text(json.dumps(
        dict(case=case, names=names, n_common_times=int(n_common), groups=corr_groups7(R, names), **info), indent=1))
    lines = ["# Night 7 T1: lag-0 correlation matrices R of the final sets and groups (|r| >= 0.7)", "",
             "R = pooled correlation, over the cells where every variable is finite and the times common to all "
             "variables of the case, of per-cell standardized anomalies with the per-time domain mean removed "
             "(Debby: 432 common hours 2006-08-11..28, Tb/precip exist only there; PNW: full anomaly record; "
             "Pinatubo: 1991 loader axis). Groups = connected components of |r| >= 0.7 (same threshold, all cases).", ""]
    for c in ("debby", "pinatubo", "pnw"):
        p = outroot / f"T1_groups_{c}.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        Rm = pd.read_csv(outroot / f"T1_corr_{c}.csv", index_col=0)
        lines += [f"## {c} ({j['n_common_times']} common times)", "", _md7(Rm.round(3)), "",
                  "Groups: " + "; ".join("{" + ", ".join(g) + "}" for g in j["groups"]), ""]
        big = [(a, b, Rm.loc[a, b]) for i, a in enumerate(Rm.index) for b in Rm.columns[i + 1:] if abs(Rm.loc[a, b]) >= 0.5]
        lines += ["Pairs with |r| >= 0.5: " + (", ".join(f"{a}-{b} {r:+.3f}" for a, b, r in big) or "none"), ""]
        if c == "pnw":
            lines += [f"SLHF-SSHF: r = {Rm.loc['slhf', 'sshf']:+.3f} ("
                      f"{'strongly correlated (>= 0.7)' if abs(Rm.loc['slhf', 'sshf']) >= GROUP_THR7 else 'not strongly correlated (< 0.7)'})", ""]
    (outroot / "T1_groups.md").write_text("\n".join(lines))


def _val_task7(task):
    variant, j = task
    g = _G
    V = len(g["types"])
    L = g["L"][variant]
    seed = surrogate_seed7(g["case"], "val", j, 1)
    xw, xs, Rm = measure_system7(g["nt_val"], g["yx"], g["params"], L, g["sd_list"], g["c_list"], g["restandardize"],
                                 g["periodic"], g["cmask"], seed, min(g["nR"], g["nt_val"]))
    rows = []
    for i, vt in enumerate(g["types"]):
        n_v = min(g["src"][vt]["period"]["n_time"], g["nt_val"])
        m, _ = field_stats(xs[i][:n_v], None, g["cellmask"][vt], g["max_lag"], censor=True)
        rows.append(dict(source=vt, realization=j, seed=seed, n_time=n_v, **{f"sur_{k}": v for k, v in m.items()}))
    return variant, rows, Rm


def _gen_task7(task):
    kind, variant, draw = task
    g = _G
    case, types = g["case"], g["types"]
    t0 = time.time()
    if kind == "sys":
        d = Path(g["outcase"]) / variant / f"draw{draw:02d}"
        rngs = [np.random.default_rng(surrogate_seed7(case, "A", draw, i + 1)) for i in range(len(types))]
        us = system_unit_fields7(g["shape"][0], g["yx"], g["params"], g["L"][variant], rngs, g["periodic"])
        names = [file_spec7(case, vt) for vt in types]
        seeds = [surrogate_seed7(case, "A", draw, i + 1) for i in range(len(types))]
        slots = list(range(1, len(types) + 1))
        tt = types
    else:
        d = Path(g["outcase"]) / "addK" / f"draw{draw:02d}"
        tt = [type_for_slot(g["add_types"], s) for s in range(1, K_ADD7 + 1)]
        slots = list(range(1, K_ADD7 + 1))
        seeds = [surrogate_seed7(case, "addK", draw, s) for s in slots]
        names = [(file_name(case, surrogate_name(case, s, vt)), surrogate_name(case, s, vt), False)
                 for s, vt in zip(slots, tt)]
        us = []
        for vt, sd in zip(tt, seeds):
            p = g["calib"][vt]["params"]
            us.append(unit_grf_ar1(*g["shape"], p["phi"], p["s_lat"], p["s_lon"], np.random.default_rng(sd),
                                   periodic_lon=g["periodic"], pad=(p["pad_lat"], p["pad_lon"])))
    d.mkdir(parents=True, exist_ok=True)
    if kind == "add":
        link_real_extras(case, d, spec="night6")
    rows, zs = [], []
    for u, vt, (fn, vn, ren), seed, slot in zip(us, tt, names, seeds, slots):
        p = g["calib"][vt]["params"]
        x, _ = field6(u, g["sd"][vt], p["c"], g["restandardize"])
        q, scale = quantize(x)
        xq = q.astype(np.float32) * scale
        xs = standardize_time(xq) if g["restandardize"] else xq
        ma, _ = field_stats(xs, None, g["cellmask"][vt], g["max_lag"], censor=True)
        mw, _ = field_stats(xs, g["tmask"], g["cellmask"][vt], g["max_lag"], censor=True)
        path = d / fn
        write_surrogate7(path, vn, xq, g["lay"],
                         dict(long_name=f"linkless-system surrogate of {vt} (TraCE-ST revision-1 night 7 {variant})",
                              source_variable=vt, seed=seed, phi=p["phi"], s_lat_cells=p["s_lat"],
                              s_lon_cells=p["s_lon"], variance_factor=p["c"], stationary=1, domain_mean_removed=1,
                              generator="data_processing/surrogates/make_grf_surrogates.py --spec night7"), ren)
        rows.append(dict(case=case, kind=kind, variant=variant, draw=draw, slot=slot, name=vn, source=vt, seed=seed,
                         file=str(path), bytes=path.stat().st_size, sha256=sha256(path),
                         **{f"sur_{k}": v for k, v in ma.items()}, **{f"win_{k}": v for k, v in mw.items()}))
        if kind == "sys":
            zs.append(zfield7(xs[:g["nR"]], g["cmask"]))
        del x, q, xq, xs
    Rm = corr_matrix7(zs) if zs else None
    for r in rows:
        r["secs"] = round(time.time() - t0, 1)
    return rows, Rm


def run_case7(case, outdir, variants, draws, draws_add, nproc, n_val=3, max_lag=60, t1_only=False, nt_val_cap=NT_VAL_CAP7,
              log=print):
    cfg = CASES6[case]
    types = cfg["types"]
    V = len(types)
    outdir = Path(outdir)
    outroot = outdir.parent if outdir.name == "surrogates" else outdir
    out = Path(outdir) / case
    out.mkdir(parents=True, exist_ok=True)
    log(f"[{case}] night 7: loading the final set via the run script's builder")
    m, c = build_case(case, spec="night6")
    fd, periodic = analysis_arrays(c["full_data"])
    tmask, win = analysis_window(m, case, c, fd.time.values)
    shape = (fd.sizes["time"], fd.sizes["lat"], fd.sizes["lon"])
    log(f"[{case}] analysis grid {shape}, periodic lon {periodic}, window {win}, vars {list(map(str, c['var_names']))}")
    src, sd, cellmask, xs_src, times = {}, {}, {}, [], []
    for ti, vt in enumerate(types):
        t0 = time.time()
        x, period, tt = stats_period_source(case, m, vt, fd, log, return_times=True)
        tgt, vmap = field_stats(x, None, None, max_lag, censor=True)
        sd[vt] = np.sqrt(np.where(np.isfinite(vmap), vmap, 1.0)).astype(np.float32)
        cellmask[vt] = np.isfinite(vmap)
        if t1_only:
            xs_src.append(x)
            times.append(pd.DatetimeIndex(tt))
            src[vt] = dict(period=period, target=tgt)
            continue
        tgt_axis, n_w = windowed_stats(x, shape[0], max_lag)
        cv = np.asarray(fd.sel(var=analysis_var_name(m, case, vt)).values, dtype=np.float32)
        tgt_win = field_stats(cv, tmask, None, max_lag, censor=True)[0]
        src[vt] = dict(period=period, target=tgt, target_fileaxis=tgt_axis, n_fileaxis_windows=n_w, target_window=tgt_win)
        xs_src.append(x)
        times.append(pd.DatetimeIndex(tt))
        log(f"[{case}] {vt}: period {period} ({time.time() - t0:.0f} s) target {tgt}")
    cmask = np.all([cellmask[vt] for vt in types], axis=0)
    r_path = outroot / f"T1_corr_{case}.csv"
    if r_path.exists() and not t1_only:
        R = pd.read_csv(r_path, index_col=0).loc[types, types].values
        n_common = json.loads((outroot / f"T1_groups_{case}.json").read_text())["n_common_times"]
        log(f"[{case}] R read from {r_path}")
    else:
        R, n_common = source_corr7(xs_src, times, cmask)
        t1_tables7(outroot, case, types, R, n_common, dict(
            periods={vt: src[vt]["period"] for vt in types}, n_cells=int(cmask.sum())))
        log(f"[{case}] T1: R over {n_common} common times, {int(cmask.sum())} cells:\n{pd.DataFrame(R, index=types, columns=types).round(3)}")
        log(f"[{case}] groups: {corr_groups7(R, types)}")
    del xs_src
    if t1_only:
        return None
    meta = dict(spec="night7", window=win, shape=shape, periodic=periodic, max_lag=max_lag,
                var_names=list(map(str, c["var_names"])))
    # marginal calibration: night-6 calibration of the case when its target statistics are unchanged, else new
    calib_path = out / "calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
    n6 = SCRATCH / "revision1/night6/surrogates" / case / "calibration.json"
    old = json.loads(n6.read_text()) if n6.exists() else {}
    todo = []
    for ti, vt in enumerate(types):
        for cand in (calib.get(vt), old.get(vt)):
            if cand and _same_stats(cand.get("target"), src[vt]["target"]) and "params" in cand:
                calib[vt] = cand
                break
        else:
            todo.append((ti, vt, src[vt]["period"]["n_time"]))
    calib.pop("_meta", None)
    calib.pop("sources", None)
    _G.clear()
    _G.update(case=case, src=src, sd=sd, cellmask=cellmask, yx=shape[1:], periodic=periodic, max_lag=max_lag,
              restandardize=cfg["restandardize"], n_real=cfg["n_real"])
    log(f"[{case}] marginal calibration reused for {[t for t in types if t not in [x[1] for x in todo]]}, "
        f"new for {[t[1] for t in todo]}")
    if todo:
        with mp.get_context("fork").Pool(min(nproc, len(todo))) as pool:
            for vt, res in pool.imap_unordered(_calib_task6, todo):
                calib[vt] = dict(target=src[vt]["target"], **res)
    calib_path.write_text(json.dumps(dict(calib, _meta=meta, sources=src), indent=1, default=float))
    params = [calib[vt]["params"] for vt in types]
    sd_list = [sd[vt] for vt in types]
    c_list = [calib[vt]["params"]["c"] for vt in types]
    nR = int(n_common)
    nt_val = int(min(max(src[vt]["period"]["n_time"] for vt in types), nt_val_cap))
    cfg7 = dict(case=case, n_rep={"debby": 3, "pinatubo": 60, "pnw": 3}[case], nt_cal=int(min(nR, 1000)), sd=sd_list,
                c=c_list, restandardize=cfg["restandardize"], periodic=periodic, cmask=cmask)
    L = {"indep": np.eye(V)}
    mix = {}
    if "A" in variants:
        log(f"[{case}] mixing calibration (A): target R, nt_cal {cfg7['nt_cal']}, n_rep {cfg7['n_rep']}")
        C, LA, hist = calibrate_mixing7(R, params, shape[1:], cfg7, log)
        L["A"] = LA
        mix = dict(C=C.tolist(), L=LA.tolist(), R=R.tolist(), types=types, history=hist)
        (out / "mixing_A.json").write_text(json.dumps(mix, indent=1))
    _G.update(types=types, params=params, sd_list=sd_list, c_list=c_list, cmask=cmask, nR=nR, nt_val=nt_val, L=L,
              calib=calib)
    vtasks = [(v, j) for v in variants if v != "addK" for j in range(n_val)]
    log(f"[{case}] {len(vtasks)} validation realizations of {nt_val} steps")
    res = {v: [] for v in L}
    Rms = {v: [] for v in L}
    with mp.get_context("fork").Pool(min(nproc, len(vtasks))) as pool:
        for v, rows, Rm in pool.imap_unordered(_val_task7, vtasks):
            res[v].extend(rows)
            Rms[v].append(Rm)
    status = {}
    for v in [x for x in variants if x != "addK"]:
        val = pd.DataFrame(res[v]).sort_values(["source", "realization"])
        tmpd = out / f"_tab_{v}"
        tmpd.mkdir(exist_ok=True)
        tab = validation_table6(case, src, val, None, tmpd, calib)
        Rmean = np.mean(Rms[v], axis=0)
        target = R if v == "A" else np.eye(V)
        dR = Rmean - target
        iu = np.triu_indices(V, 1)
        worstR = float(np.abs(dR[iu]).max()) if V > 1 else 0.0
        margin_ok = bool(tab["pass"].all())
        status[v] = dict(marginal_pass=margin_ok, corr_max_abs_diff=worstR, corr_pass=worstR <= R_TOL_PASS7,
                         corr_target="R (T1)" if v == "A" else "identity")
        md = (tmpd / f"validation_{case}.md").read_text().replace(
            f"### {case} (night 6", f"### {case}, system {v} (night 7; marginals as night 6, on the system realizations")
        cr = pd.DataFrame(Rmean, index=types, columns=types)
        md += (f"\nLag-0 cross-correlation of the generated system ({n_val} realizations of {nt_val} steps, R measured on "
               f"the first {min(nR, nt_val)}) vs target ({status[v]['corr_target']}): max |diff| = {worstR:.3f} "
               f"(pass <= {R_TOL_PASS7}: {'yes' if status[v]['corr_pass'] else 'NO'}).\n\n"
               + _md7(cr.round(3)) + "\n\nDifference (generated - target):\n\n"
               + _md7(pd.DataFrame(dR, index=types, columns=types).round(3)) + "\n")
        (outdir / f"validation_{case}_{v}.md").write_text(md)
        tab.assign(variant=v, corr_max_abs_diff=worstR).to_csv(outdir / f"validation_{case}_{v}.csv", index=False)
        pd.DataFrame(dR, index=types, columns=types).to_csv(out / f"validation_corr_diff_{v}.csv", float_format="%.4f")
        log(md)
    (out / "validation_status.json").write_text(json.dumps(status, indent=1))
    run_v = [v for v in variants if v == "addK" or (status[v]["marginal_pass"] and (v == "indep" or status[v]["corr_pass"]))]
    if "addK" in variants and not status.get("indep", dict(marginal_pass=True))["marginal_pass"]:
        run_v.remove("addK")
    dropped = [v for v in variants if v not in run_v]
    log(f"[{case}] production variants {run_v}; dropped (validation failed) {dropped}")
    if not run_v:
        return None
    lay = native_layout(case, dict(time=fd.time.values, lat=fd.lat.values, lon=fd.lon.values), periodic,
                        time_slice=getattr(m, "TIME_SLICE", None) if case == "pnw" else None)
    _G.update(shape=shape, tmask=tmask, outcase=str(out), lay=lay, restandardize=cfg["restandardize"],
              add_types=cfg["exp2_types"])
    tasks = [("sys", v, d) for v in run_v if v != "addK" for d in draws] + [("add", "addK", d) for d in draws_add if "addK" in run_v]
    log(f"[{case}] generating {len(tasks)} directories with {nproc} processes")
    del c, fd
    rows, corr_rows = [], []
    with mp.get_context("fork").Pool(nproc) as pool:
        for i, (r, Rm) in enumerate(pool.imap_unordered(_gen_task7, tasks)):
            rows.extend(r)
            if Rm is not None:
                corr_rows += [dict(variant=r[0]["variant"], draw=r[0]["draw"], a=types[p], b=types[q], r=Rm[p, q],
                                   target=(R if r[0]["variant"] == "A" else np.eye(V))[p, q])
                              for p in range(V) for q in range(p + 1, V)]
            log(f"  {i + 1}/{len(tasks)}")
    st = pd.DataFrame(rows).sort_values(["kind", "variant", "draw", "slot"])
    st.to_csv(out / "surrogate_stats.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(out / "corr_realized.csv", index=False)
    st[["case", "variant", "draw", "slot", "name", "source", "seed"]].assign(
        seed_formula="BASE_SEED7 20261118 + 1e6 case + 1e4 variant (A 1; indep uses the A base-noise seeds; addK 3) + 100 draw + slot"
    ).to_csv(out / "seeds.csv", index=False)
    with open(out / "sha256.txt", "w") as f:
        for _, r in st.iterrows():
            f.write(f"{r.sha256}  {Path(r.file).relative_to(out)}\n")
    # the run script's own loaders read the first draw of each variant (same values as generated)
    checks = {}
    for v in run_v:
        if v == "addK":
            continue
        checks[v] = loader_check7(case, out / v / f"draw{draws[0]:02d}", types, log)
    if "addK" in run_v:
        sur = [(surrogate_name(case, s, type_for_slot(cfg["exp2_types"], s)), type_for_slot(cfg["exp2_types"], s))
               for s in range(1, K_ADD7 + 1)]
        checks["addK"] = loader_check_addk7(case, out / "addK" / f"draw{draws_add[0]:02d}", sur, log)
    (out / "loader_check.json").write_text(json.dumps(checks, indent=1))
    with open(out / "system_args.txt", "w") as f:
        f.write("# variant|draw|data arguments of a run on the system (all variables surrogates)\n")
        for v in run_v:
            if v != "addK":
                for d in draws:
                    f.write(f"{v}|{d}|{system_args7(case, out / v / f'draw{d:02d}')}\n")
    if "addK" in run_v:
        with open(out / "addK_args.txt", "w") as f:
            f.write("# K|draw|data arguments (final set + addK slots 1..K of the draw; K=0 = final set)\n")
            for k in (0, 2, 4, 8, 16):
                for d in draws_add:
                    f.write(f"{k}|{d}|{addk_args7(case, out / 'addK' / f'draw{d:02d}', k)}\n")
    log(f"[{case}] done: {len(st)} files, {st.bytes.sum() / 1e9:.2f} GB; status {status}")
    return st, status, checks


def loader_check_addk7(case, drawdir, sur, log=print):
    """The run script's loader reads final set + the addK fields of drawdir: all present, no NaN."""
    m, c = build_case(case, sur, drawdir, spec="night6")
    fd, _ = analysis_arrays(c["full_data"])
    nan = {n: float(np.isnan(np.asarray(fd.sel(var=n.upper() if case == "pnw" else n).values)).mean()) for n, _ in sur}
    res = dict(case=case, dir=str(drawdir), n_vars=len(c["var_names"]), max_nan_frac=max(nan.values()),
               ok=bool(len(c["var_names"]) == len(CASES6[case]["types"]) + len(sur) and max(nan.values()) == 0.0))
    log(f"[{case}] addK loader check {drawdir}: {res}")
    return res


def loader_check7(case, drawdir, types, log=print):
    """The run script's own loader reads the system directory: same values as the generated fields (compared with
    the values read back through xarray from the files themselves, as the case sees them)."""
    drawdir = Path(drawdir)
    m = load_script(CASES[case]["script"])
    if case == "debby":
        os.environ["TST_DEBBY_SYSTEM_PATH"] = str(drawdir)
        ex = [(a, str(drawdir / DEBBY_FILES7[a][0]), DEBBY_FILES7[a][1]) for a in ("CAPE", "DIV200", "MSLP")]
        c = m.build_case_debby(extra_inputs=ex, vo_input=(str(drawdir / "vo.nc"), "vo"))
        os.environ.pop("TST_DEBBY_SYSTEM_PATH", None)
    elif case == "pinatubo":
        os.environ["TST_PINATUBO_SYSTEM_PATH"] = str(drawdir)
        os.environ["TST_PINATUBO_EXTRA"] = ",".join(PIN_EXTRAS6)
        try:
            c = m.build_case_pinatubo()
        finally:
            for k in ("TST_PINATUBO_SYSTEM_PATH", "TST_PINATUBO_EXTRA"):
                os.environ.pop(k, None)
    else:
        m.PATH_FILES = str(drawdir).rstrip("/") + "/"
        c = m.build_case_pnw(extra_vars=PNW_EXTRAS6, extra_path=str(drawdir))
    fd, _ = analysis_arrays(c["full_data"])
    res = dict(case=case, dir=str(drawdir), var_names=list(map(str, c["var_names"])), n_vars=len(c["var_names"]),
               nan_frac={}, std={}, mean_abs={})
    for vt in types:
        got = np.asarray(fd.sel(var=analysis_var_name(m, case, vt)).values, dtype=np.float64)
        res["nan_frac"][vt] = float(np.isnan(got).mean())
        res["std"][vt] = float(np.nanstd(got))
        res["mean_abs"][vt] = float(np.nanmean(np.abs(got)))
    res["ok"] = bool(len(res["var_names"]) == len(types) and max(res["nan_frac"].values()) == 0.0)
    log(f"[{case}] loader check {drawdir}: ok={res['ok']} n_vars={res['n_vars']} nan_frac max {max(res['nan_frac'].values()):.2g}")
    return res



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--max-lag", type=int, default=60)
    ap.add_argument("--spec", choices=["night5", "night6", "night7"], default="night5",
                    help="night6 (opt-in): stationary GRF from anomaly-period statistics, domain mean removed, final sets")
    ap.add_argument("--mode", choices=["exp1", "exp2", "both"], default="both", help="night6: which experiment's files")
    ap.add_argument("--draws2", type=int, default=5, help="night6: exp2 draws")
    ap.add_argument("--k2", type=int, default=K2_MAX, help="night6: exp2 slots per draw")
    ap.add_argument("--n-val", type=int, default=3, help="night6: validation realizations per type")
    ap.add_argument("--variants", nargs="+", default=["A", "indep", "addK"], choices=["A", "indep", "addK"],
                    help="night7: linkless system with R (A), with R = identity (indep), addK sequences")
    ap.add_argument("--draws-add", type=int, default=5, help="night7: addK draws")
    ap.add_argument("--t1-only", action="store_true", help="night7: only R, groups (T1_corr_<case>.csv, T1_groups.md)")
    ap.add_argument("--nt-val-cap", type=int, default=NT_VAL_CAP7, help="night7: max length of validation realizations")
    a = ap.parse_args()
    t0 = time.time()
    if a.spec == "night7":
        res = run_case7(a.case, a.outdir, a.variants, list(range(a.draws)), list(range(a.draws_add)), a.nproc,
                        a.n_val, a.max_lag, a.t1_only, a.nt_val_cap, log=lambda s: print(s, flush=True))
        if res is None and not a.t1_only:
            sys.exit(3)
        print(f"[{a.case}] night 7 finished, {time.time() - t0:.0f} s", flush=True)
        return
    if a.spec == "night6":
        res = run_case6(a.case, a.outdir, a.mode, list(range(a.draws)), list(range(a.draws2)), a.k2, a.nproc,
                        a.n_val, a.max_lag, log=lambda s: print(s, flush=True))
        if res is None:
            sys.exit(3)
        st, checks = res
        print(f"[{a.case}] done: {len(st)} files, {st.bytes.sum() / 1e9:.2f} GB, loader checks "
              f"{ {k: v['ok'] for k, v in checks.items()} }, {time.time() - t0:.0f} s", flush=True)
        return
    st = run_case(a.case, a.outdir, list(range(a.draws)), a.k, a.nproc, a.max_lag,
                  log=lambda s: print(s, flush=True))
    print(f"[{a.case}] done: {len(st)} files, {st.bytes.sum() / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
