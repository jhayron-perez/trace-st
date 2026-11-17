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
