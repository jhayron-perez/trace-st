"""Night-5 T6: GRF surrogates (data_processing/surrogates/make_grf_surrogates.py) and the opt-in
--extra-path of 6_PinatuboRuns.py (default None = original PATH_FILES_E3SM_EXTRA)."""
import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np

from tests.test_fraction_full_length import load_script
from tests.test_pinatubo_extra_vars import PRE, h

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("make_grf_surrogates",
                                               ROOT / "data_processing" / "surrogates" / "make_grf_surrogates.py")
grf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grf)


def _synthetic(seed, nt=400, ny=60, nx=80, tau=5.0, s=(2.0, 3.0), sd=2.5):
    """Known truth: AR(1) phi = exp(-1/tau), Gaussian kernel s cells (corr e-folding 2 s), sd."""
    u = grf.unit_grf_ar1(nt, ny, nx, np.exp(-1.0 / tau), s[0], s[1], np.random.default_rng(seed))
    return sd * u


class TestGRFGenerator(unittest.TestCase):
    def test_known_field(self):
        tau, s, sd = 5.0, (2.0, 3.0), 2.5
        x = _synthetic(1, tau=tau, s=s, sd=sd)
        m, _ = grf.field_stats(x, remove_common=False)
        # truth for the generator itself (no window demeaning bias beyond 10 %)
        self.assertLess(abs(np.var(x) / sd ** 2 - 1), 0.10)
        self.assertLess(abs(m["r1"] / np.exp(-1.0 / tau) - 1), 0.10)
        self.assertLess(abs(m["L_lat"] / (2 * s[0]) - 1), 0.10)
        self.assertLess(abs(m["L_lon"] / (2 * s[1]) - 1), 0.10)

    def test_calibration_reproduces_source(self):
        # source = an independent synthetic field; surrogate calibrated on its stats, generated with a new seed
        src = _synthetic(2, nt=200, tau=4.0, s=(1.5, 2.5), sd=1.7)
        tmask = np.zeros(200, bool)
        tmask[40:] = True
        tgt, vmap = grf.field_stats(src, tmask)
        sd_out = np.std(src[~tmask], axis=0)
        p, _ = grf.calibrate(tgt, src.shape, tmask, vmap, sd_out, np.isfinite(vmap), 7, False, False,
                             log=lambda s: None)
        u = grf.unit_grf_ar1(*src.shape, p["phi"], p["s_lat"], p["s_lon"], np.random.default_rng(8))
        x = grf.scale_field(u, tmask, vmap, sd_out, p["c"], False)
        m, _ = grf.field_stats(x, tmask)
        for k in ("var", "r1", "L_lat", "L_lon"):
            self.assertLess(abs(grf.rel_err(m[k], tgt[k])), 0.10, k)

    def test_restandardized_window_variance(self):
        # Pinatubo path: loader standardizes over the whole period; window variance still matched
        src = _synthetic(3, nt=215, ny=30, nx=40, tau=3.0, s=(1.0, 1.5))
        src[150:170] *= 2.0
        src = grf.standardize_time(src)
        tmask = np.zeros(215, bool)
        tmask[150:170] = True
        tgt, vmap = grf.field_stats(src, tmask)
        p, _ = grf.calibrate(tgt, src.shape, tmask, vmap, None, np.isfinite(vmap), 9, False, True,
                             log=lambda s: None)
        u = grf.unit_grf_ar1(*src.shape, p["phi"], p["s_lat"], p["s_lon"], np.random.default_rng(10))
        x = grf.standardize_time(grf.scale_field(u, tmask, vmap, None, p["c"], True, apply_std=False))
        m, _ = grf.field_stats(x, tmask)
        self.assertLess(abs(grf.rel_err(m["var"], tgt["var"])), 0.10)

    def test_same_seed_identical(self):
        a = grf.unit_grf_ar1(50, 20, 30, 0.7, 1.2, 2.0, np.random.default_rng(42))
        b = grf.unit_grf_ar1(50, 20, 30, 0.7, 1.2, 2.0, np.random.default_rng(42))
        c = grf.unit_grf_ar1(50, 20, 30, 0.7, 1.2, 2.0, np.random.default_rng(43))
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_periodic_lon(self):
        x = grf.unit_grf_ar1(100, 30, 90, 0.5, 1.0, 2.0, np.random.default_rng(5), periodic_lon=True)
        seam = np.mean(x[:, :, 0] * x[:, :, -1])
        inner = np.mean(x[:, :, 1] * x[:, :, 0])
        self.assertLess(abs(seam - inner), 0.1)

    def test_round_robin_and_seeds(self):
        t = grf.CASES["debby"]["types"]
        self.assertEqual([grf.type_for_slot(t, s) for s in (1, 7, 8, 16)], ["Tb", "MSLP", "Tb", "vo"])
        seeds = {grf.surrogate_seed(c, d, s) for c in grf.CASE_IDX for d in range(20) for s in range(1, 17)}
        self.assertEqual(len(seeds), 3 * 20 * 16)
        self.assertEqual(grf.surrogate_name("pnw", 1, "z500"), "s01z500")
        self.assertEqual(grf.surrogate_name("pinatubo", 4, "AEROD_v"), "S04AERODv")

    def test_quantize(self):
        x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
        q, scale = grf.quantize(x)
        self.assertEqual(q.dtype, np.int16)
        self.assertLess(np.max(np.abs(q * scale - x)), scale)


class TestNight6Generator(unittest.TestCase):
    """Night 6 (--spec night6): stationary AR(1) GRF, per-time domain mean removed, anomaly-period statistics."""

    def test_domain_mean_removed(self):
        u = grf.unit_grf_ar1(120, 30, 40, 0.8, 3.0, 4.0, np.random.default_rng(11))
        sd = np.random.default_rng(12).uniform(0.5, 2.0, (30, 40)).astype(np.float32)
        x, xs = grf.field6(u, sd, 1.3, False)
        self.assertLess(np.max(np.abs(x.mean(axis=(1, 2)))), 1e-5)
        self.assertIs(x, xs)
        _, xr = grf.field6(u, sd, 1.3, True)          # Pinatubo view: re-standardized per cell
        self.assertTrue(np.allclose(xr.std(0), 1.0, atol=1e-4))

    def test_stationary_variance(self):
        # one sd map at every time: the variance does not depend on time (no in/out-of-window split)
        sd = np.full((40, 60), 2.0, np.float32)
        sd[:, 30:] = 1.0
        u = grf.unit_grf_ar1(2000, 40, 60, 0.5, 1.0, 1.0, np.random.default_rng(13))
        x, _ = grf.field6(u, sd, 1.0, False)
        v1, v2 = x[:1000].var(0), x[1000:].var(0)
        self.assertLess(abs(v1[:, :30].mean() / v2[:, :30].mean() - 1), 0.05)
        self.assertLess(abs(v1[:, :30].mean() / 4.0 - 1), 0.10)          # sd map kept (small domain-mean loss)
        self.assertLess(abs(v1[:, 30:].mean() / 1.0 - 1), 0.10)

    def test_ar1_rho(self):
        for phi in (0.3, 0.7, 0.95):
            u = grf.unit_grf_ar1(3000, 20, 30, phi, 1.0, 1.5, np.random.default_rng(14))
            x, _ = grf.field6(u, np.ones((20, 30), np.float32), 1.0, False)
            m, _ = grf.field_stats(x)
            self.assertLess(abs(m["r1"] / phi - 1), 0.05, phi)

    def test_calibration_long_source(self):
        # source with a common (domain-wide) component: the calibrated, domain-mean-removed surrogate of the same
        # length matches var, r1, L_lat, L_lon within 10 % on an independent seed
        src = _synthetic(21, nt=600, ny=40, nx=60, tau=4.0, s=(1.5, 2.5), sd=1.5)
        src += 0.5 * np.random.default_rng(22).standard_normal((600, 1, 1)).astype(np.float32)
        tgt, vmap = grf.field_stats(src)
        sd = np.sqrt(vmap).astype(np.float32)
        p, _ = grf.calibrate(tgt, src.shape, np.ones(600, bool), None, None, np.isfinite(vmap), 23, False, False,
                             log=lambda s: None, field_fn=lambda u, c: grf.field6(u, sd, c, False)[1])
        u = grf.unit_grf_ar1(*src.shape, p["phi"], p["s_lat"], p["s_lon"], np.random.default_rng(24),
                             pad=(p["pad_lat"], p["pad_lon"]))
        m, _ = grf.field_stats(grf.field6(u, sd, p["c"], False)[1])
        for k in ("var", "r1", "L_lat", "L_lon"):
            self.assertLess(abs(grf.rel_err(m[k], tgt[k])), 0.10, k)

    def test_seeds_sets_and_names(self):
        n6 = {grf.surrogate_seed6(c, e, d, s) for c in grf.CASE_IDX for e in ("exp1", "exp2")
              for d in range(20) for s in range(1, 33)}
        self.assertEqual(len(n6), 3 * 2 * 20 * 32)
        n5 = {grf.surrogate_seed(c, d, s) for c in grf.CASE_IDX for d in list(range(20)) + [99] for s in range(1, 17)}
        self.assertFalse(n6 & n5)
        self.assertEqual(grf.CASES6["debby"]["types"], ["vo", "Tb", "precip", "CAPE", "DIV200", "MSLP"])
        self.assertEqual(grf.CASES6["pinatubo"]["exp2_types"], ["TMSO203", "BURDENSO403", "T050"])
        self.assertEqual([e[0] for e in grf.DEBBY_EXTRAS6], ["CAPE", "DIV200", "MSLP"])
        t = grf.CASES6["pnw"]["exp2_types"]
        self.assertEqual([grf.type_for_slot(t, s) for s in (1, 7, 8, 32)], ["z500", "t2m", "z500", "slhf"])
        a = grf.run_args6("pinatubo", "/o/pinatubo", "exp2", 3, k=2)
        self.assertEqual(a, "--extra-vars TMSO203 BURDENSO403 T050 S01TMSO203 S02BURDENSO403 "
                            "--extra-path /o/pinatubo/exp2/draw03")
        a = grf.run_args6("pnw", "/o/pnw", "exp1", 0, vtype="z10")
        self.assertEqual(a, "--extra-vars t2m s01z10 --extra-path /o/pnw/exp1/z10/draw00")

    def test_windowed_stats(self):
        x = _synthetic(31, nt=400, ny=20, nx=30)
        s1, n1 = grf.windowed_stats(x, 400)
        self.assertEqual(n1, 1)
        self.assertEqual(s1, grf.field_stats(x)[0])
        s2, n2 = grf.windowed_stats(x, 100)
        self.assertEqual(n2, 4)
        self.assertLess(s2["var"], s1["var"] * 1.05)

    def test_default_path_unchanged(self):
        # night-5 defaults: extended sets, seeds, calibrate without field_fn identical to field_fn=scale_field
        self.assertEqual(grf.EXTRAS["night5"], (grf.DEBBY_EXTRAS, grf.PIN_EXTRAS, grf.PNW_EXTRAS))
        self.assertEqual(grf.PIN_EXTRAS, ["TMSO203", "BURDENSO403", "FLUT", "T050", "FSDS"])
        self.assertEqual(grf.surrogate_seed("debby", 0, 1), 20561117)
        src = _synthetic(41, nt=120, ny=20, nx=30, tau=3.0, s=(1.0, 1.5))
        tmask = np.zeros(120, bool)
        tmask[30:] = True
        tgt, vmap = grf.field_stats(src, tmask)
        sd_out = np.std(src[~tmask], axis=0)
        kw = dict(log=lambda s: None, max_iter=3)
        p0, h0 = grf.calibrate(tgt, src.shape, tmask, vmap, sd_out, np.isfinite(vmap), 5, False, False, **kw)
        p1, h1 = grf.calibrate(tgt, src.shape, tmask, vmap, sd_out, np.isfinite(vmap), 5, False, False,
                               field_fn=lambda u, c: grf.scale_field(u, tmask, vmap, sd_out, c, False), **kw)
        self.assertEqual(p0, p1)
        self.assertEqual(h0, h1)

    def test_censored_length(self):
        # e-folding beyond max_lag: nan by default (night 5), max_lag with censor=True (night 6)
        x = grf.unit_grf_ar1(40, 20, 30, 0.5, 8.0, 15.0, np.random.default_rng(51))
        m0, _ = grf.field_stats(x, max_lag=3)
        m1, _ = grf.field_stats(x, max_lag=3, censor=True)
        self.assertTrue(np.isnan(m0["L_lon"]))
        self.assertEqual(m1["L_lon"], 3.0)
        self.assertEqual(m0["r1"], m1["r1"])


class TestPinatuboExtraPath(unittest.TestCase):
    def test_default_unchanged(self):
        os.environ.pop("TST_PINATUBO_EXTRA", None)
        os.environ.pop("TST_PINATUBO_EXTRA_PATH", None)
        m = load_script("6_PinatuboRuns.py")
        self.assertEqual(h(m.build_case_pinatubo()["full_data"]), PRE)
        os.environ["TST_PINATUBO_EXTRA"] = "TMSO203,FLUT"
        try:
            ref = m.build_case_pinatubo()["full_data"]
            os.environ["TST_PINATUBO_EXTRA_PATH"] = m.PATH_FILES_E3SM_EXTRA
            same = m.build_case_pinatubo()["full_data"]
        finally:
            os.environ.pop("TST_PINATUBO_EXTRA", None)
            os.environ.pop("TST_PINATUBO_EXTRA_PATH", None)
        self.assertEqual(h(ref), h(same))

    def test_cli_default_none(self):
        import sys
        m = load_script("6_PinatuboRuns.py")
        argv = sys.argv
        sys.argv = ["6_PinatuboRuns.py", "--outdir", "x"]
        try:
            self.assertIsNone(m.parse_args().extra_path)
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
