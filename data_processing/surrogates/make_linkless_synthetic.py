"""Linkless surrogate system for the synthetic case (revision 1, night 7 T3b; approach A of the brief).

The whole candidate set of one synthetic case (A, B, C and the extra_mix variables) is replaced by a system that
keeps each variable's per-cell time mean, per-cell standard deviation, pooled lag-1 autocorrelation (AR(1) memory)
and spatial e-folding lengths (lat, lon), and the lag-0 correlation matrix R between the variables, but has no lagged
cross-terms: independent unit-variance fields are mixed with the Cholesky factor of R in the innovation, then AR(1)
in time and Gaussian spatial smoothing per variable (make_grf_surrogates.system_unit_fields7), x = mean + sd * u,
clipped at 0 as the synthetic variables are. variant "indep": R = identity (same base noise seeds).

Statistics are estimated from the case's own fields (one realization of the synthetic generator per mixture strength).
Seeds: BASE + 1e6 * 4 (synthetic) + 1e4 * variant (A 1, indep 2) ... see seed_for. The realized statistics of the
generated system are returned (and written as JSON when log_dir is given) for the validation table.

Used by Scripts_Paper/3_SyntheticRuns_Multivariate.py --linkless {A,indep} --linkless-draw D (opt-in).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_SEED = 20261118
CASE_IDX_SYN = 4
VAR_IDX = {"A": 1, "indep": 2}
INV_E = float(np.exp(-1.0))
MAX_LAG = 60


def _grf():
    spec = importlib.util.spec_from_file_location("make_grf_surrogates", HERE / "make_grf_surrogates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def seed_for(draw, slot):
    """Base-noise seed of draw `draw`, base field `slot` (common to the A and indep variants)."""
    return int(BASE_SEED + 1000000 * CASE_IDX_SYN + 10000 * VAR_IDX["A"] + 100 * int(draw) + int(slot))


def _efold(ac):
    """First lag (1..) where ac < 1/e, linearly interpolated; censored at len(ac) when never reached."""
    for k in range(1, len(ac)):
        if ac[k] < INV_E:
            return float(k - 1 + (ac[k - 1] - INV_E) / max(ac[k - 1] - ac[k], 1e-12))
    return float(len(ac))


def field_stats(x, max_lag=MAX_LAG):
    """x (T, NY, NX): mean map, sd map, standardized anomalies z, pooled lag-1 rho, e-folding lengths (lat, lon)."""
    x = np.asarray(x, dtype=np.float32)
    m = x.mean(0)
    a = x - m
    sd = np.sqrt((a.astype(np.float64) ** 2).mean(0)).astype(np.float32)
    ok = sd > 1e-6
    z = np.where(ok[None], a / np.where(ok, sd, 1.0)[None], 0.0).astype(np.float32)
    den = float((z.astype(np.float64) ** 2).sum()) + 1e-30
    rho = float((z[1:].astype(np.float64) * z[:-1]).sum() / den)
    nlat, nlon = z.shape[1], z.shape[2]
    ac_lat = [1.0] + [float((z[:, k:, :].astype(np.float64) * z[:, :-k, :]).sum() / den * z.shape[1] / (nlat - k)) for k in range(1, min(max_lag, nlat - 1) + 1)]
    ac_lon = [1.0] + [float((z[:, :, k:].astype(np.float64) * z[:, :, :-k]).sum() / den * z.shape[2] / (nlon - k)) for k in range(1, min(max_lag, nlon - 1) + 1)]
    return dict(mean=m, sd=sd, z=z, rho=rho, L_lat=_efold(ac_lat), L_lon=_efold(ac_lon))


def _phi_for(r_hat, nt):
    """AR(1) coefficient whose pooled lag-1 estimate, after removing each cell's time mean over nt steps (as field_stats
    does for the source and the generated fields), equals r_hat: r_hat ~ phi - (1 + 3 phi) / nt (short series)."""
    phi = float(r_hat)
    for _ in range(20):
        phi = float(r_hat) + (1.0 + 3.0 * phi) / float(nt)
    return float(np.clip(phi, 0.0, 0.995))


def corr_matrix(zs, step=3):
    """Pooled lag-0 correlation of the standardized anomalies over every `step`-th cell and all times."""
    Z = np.stack([z[:, ::step, ::step].reshape(-1).astype(np.float64) for z in zs])
    return np.corrcoef(Z)


def linkless_system(arrs, names, draw, variant="A", clip=True, log_dir=None, tag=""):
    """arrs: list of (T, NY, NX) arrays of the real case (A, B, C, extras). Returns (list of surrogate arrays, report)."""
    G = _grf()
    st = [field_stats(x) for x in arrs]
    V = len(arrs)
    params = []
    for s in st:
        sl, sn = max(s["L_lat"] / 2.0, 1e-3), max(s["L_lon"] / 2.0, 1e-3)   # Gaussian kernel: corr(d) = exp(-d^2 / 4 s^2)
        params.append(dict(phi=_phi_for(s["rho"], arrs[0].shape[0]), s_lat=sl, s_lon=sn,
                           pad_lat=int(np.ceil(4.0 * sl)) + 1, pad_lon=int(np.ceil(4.0 * sn)) + 1))
    R = corr_matrix([s["z"] for s in st])
    if variant == "A":
        X = np.eye(V)
        for i in range(V):
            for j in range(V):
                if i != j:
                    X[i, j] = R[i, j] / G.attenuation7(params[i], params[j])
        Lmix = G.mixing_factor7(X)
    elif variant == "indep":
        Lmix = np.eye(V)
    else:
        raise ValueError(f"variant {variant!r}")
    T, NY, NX = np.asarray(arrs[0]).shape
    rngs = [np.random.default_rng(seed_for(draw, j + 1)) for j in range(V)]
    us = G.system_unit_fields7(T, (NY, NX), params, Lmix, rngs)
    out, gst = [], []
    for v in range(V):
        x = st[v]["mean"][None] + st[v]["sd"][None] * us[v]
        if clip:
            x = np.clip(x, 0.0, None)
        out.append(x.astype(np.float32))
        gst.append(field_stats(us[v]))
    Rg = corr_matrix([g["z"] for g in gst])
    rep = dict(variant=variant, draw=int(draw), tag=tag, names=list(names),
               R_source=R.round(4).tolist(), R_generated=Rg.round(4).tolist(),
               max_abs_corr_diff=float(np.abs(Rg - R).max()) if variant == "A" else float(np.abs(Rg - np.eye(V)).max()),
               max_abs_corr_diff_vs_R=float(np.abs(Rg - R).max()),
               rho_source=[float(s["rho"]) for s in st], rho_generated=[float(g["rho"]) for g in gst],
               L_lat_source=[s["L_lat"] for s in st], L_lat_generated=[g["L_lat"] for g in gst],
               L_lon_source=[s["L_lon"] for s in st], L_lon_generated=[g["L_lon"] for g in gst])
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(log_dir) / f"linkless_{variant}_d{int(draw):02d}{tag}.json", "w") as f:
            json.dump(rep, f)
    return out, rep


# ---------------------------------------------------------------------------------------------------------------
# Night 8: the "typical field" of a case = an independent linkless field with the MEDIAN statistics of the case's
# variables (per-cell mean and sd map: per-cell median; pooled lag-1 rho, e-folding L_lat and L_lon: median over the
# variables). Used as the decoy of the filter and as the "unrelated" added variable of the new S10.
# ---------------------------------------------------------------------------------------------------------------
TYPICAL_BASE_SEED = 20261123


def typical_seed(case_idx, member):
    """Seed of typical-field realization `member` of case `case_idx` (synthetic = 4): 20261123 + 1e6 * case + member."""
    return int(TYPICAL_BASE_SEED + 1000000 * int(case_idx) + int(member))


def typical_targets(arrs):
    """Median statistics over the (T, NY, NX) arrays of the case's variables."""
    st = [field_stats(x) for x in arrs]
    return dict(mean=np.median(np.stack([s["mean"] for s in st]), 0).astype(np.float32),
                sd=np.median(np.stack([s["sd"] for s in st]), 0).astype(np.float32),
                rho=float(np.median([s["rho"] for s in st])), L_lat=float(np.median([s["L_lat"] for s in st])),
                L_lon=float(np.median([s["L_lon"] for s in st])), nt=int(np.asarray(arrs[0]).shape[0]))


def typical_field(targets, seed, clip=True, shape=None):
    """One linkless realization (T, NY, NX) float32 with the typical-field `targets` (see typical_targets)."""
    G = _grf()
    T = targets["nt"]
    NY, NX = targets["sd"].shape
    sl, sn = max(targets["L_lat"] / 2.0, 1e-3), max(targets["L_lon"] / 2.0, 1e-3)
    params = [dict(phi=_phi_for(targets["rho"], T), s_lat=sl, s_lon=sn, pad_lat=int(np.ceil(4.0 * sl)) + 1,
                   pad_lon=int(np.ceil(4.0 * sn)) + 1)]
    u = G.system_unit_fields7(T, (NY, NX), params, np.eye(1), [np.random.default_rng(int(seed))])[0]
    # the targets are sample sds of a T-step series (cell mean removed): inflate by the expected deflation of the
    # sample variance of an AR(1) process of coefficient phi (variance of the sample mean), so that the realized
    # sample variance, as measured by field_stats, equals the target
    phi = params[0]["phi"]
    vmean = ((1.0 + phi) / (1.0 - phi) - 2.0 * phi * (1.0 - phi ** T) / (T * (1.0 - phi) ** 2)) / T
    x = targets["mean"][None] + targets["sd"][None] * u / np.sqrt(max(1.0 - vmean, 0.2))
    if clip:
        x = np.clip(x, 0.0, None)
    return x.astype(np.float32)
