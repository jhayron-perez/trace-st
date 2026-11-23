"""Night 8: typical field (median statistics of a case's variables, independent, no lagged links) and the opt-in
--extra-mix type "typical" of 3_SyntheticRuns_Multivariate.py (existing mixes unchanged)."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

from tests.test_fraction_full_length import load_script

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("make_linkless_synthetic",
                                               ROOT / "data_processing" / "surrogates" / "make_linkless_synthetic.py")
ll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ll)


class TypicalFieldTest(unittest.TestCase):
    def test_seed_formula(self):
        self.assertEqual(ll.typical_seed(4, 105), 20261123 + 4000000 + 105)

    def test_stats_within_10_percent_on_independent_realizations(self):
        rng = np.random.default_rng(0)
        arrs = [1.0 + s * ll.typical_field(dict(mean=np.zeros((20, 30), np.float32), sd=np.ones((20, 30), np.float32),
                                               rho=r, L_lat=3.0, L_lon=4.0, nt=120), 100 + i, clip=False)
                for i, (s, r) in enumerate([(0.5, 0.6), (0.8, 0.8), (1.2, 0.7)])]
        tg = ll.typical_targets(arrs)
        st = [ll.field_stats(ll.typical_field(tg, 1000 + k, clip=False)) for k in range(12)]
        self.assertAlmostEqual(np.mean([s["rho"] for s in st]) / tg["rho"], 1.0, delta=0.1)
        self.assertAlmostEqual(np.mean([s["L_lat"] for s in st]) / tg["L_lat"], 1.0, delta=0.1)
        self.assertAlmostEqual(np.mean([(s["sd"] ** 2).mean() for s in st]) / (tg["sd"] ** 2).mean(), 1.0, delta=0.1)

    def test_deterministic_and_independent(self):
        tg = dict(mean=np.zeros((20, 30), np.float32), sd=np.ones((20, 30), np.float32), rho=0.7, L_lat=3.0, L_lon=4.0, nt=80)
        a, b, c = ll.typical_field(tg, 5, False), ll.typical_field(tg, 5, False), ll.typical_field(tg, 6, False)
        np.testing.assert_array_equal(a, b)
        self.assertLess(abs(np.corrcoef(a.ravel(), c.ravel())[0, 1]), 0.1)


class TypicalMixTest(unittest.TestCase):
    def test_mix_names_and_base_fields_unchanged(self):
        m = load_script("3_SyntheticRuns_Multivariate.py")
        base = m.build_multivariate_case(alpha_mix=0.6, extra_mix=["response", "nonlinear"], seed=11)
        mix = m.build_multivariate_case(alpha_mix=0.6, extra_mix=["typical", "response", "nonlinear"], seed=11)
        self.assertEqual([str(v) for v in mix["full_data_run"]["var"].values], ["A", "B", "C", "Z1", "R1_B", "Q1_A"])
        self.assertEqual(mix["extra_vars"][0], ("Z1", "typical"))
        for v in ("A", "B", "C"):
            np.testing.assert_array_equal(mix["full_data_run"].sel(var=v).values, base["full_data_run"].sel(var=v).values)
        # the response and nonlinear variables do not depend on the typical field
        np.testing.assert_array_equal(mix["full_data_run"].sel(var="R1_B").values, base["full_data_run"].sel(var="R1_B").values)
        z = mix["full_data_run"].sel(var="Z1").values
        self.assertTrue(np.isfinite(z).all())
        again = m.build_multivariate_case(alpha_mix=0.6, extra_mix=["typical", "response", "nonlinear"], seed=11)
        np.testing.assert_array_equal(z, again["full_data_run"].sel(var="Z1").values)

    def test_two_typical_fields_are_different_realizations(self):
        m = load_script("3_SyntheticRuns_Multivariate.py")
        c = m.build_multivariate_case(alpha_mix=0.2, extra_mix=["typical", "response", "nonlinear", "typical"], seed=11)
        d = c["full_data_run"]
        self.assertFalse(np.allclose(d.sel(var="Z1").values, d.sel(var="Z2").values))


    def test_trial_ids_flag_default_none(self):
        import sys
        m = load_script("3_SyntheticRuns_Multivariate.py")
        old = sys.argv
        try:
            sys.argv = ["x", "--outdir", "o"]
            self.assertIsNone(m.parse_args().trial_ids)
            sys.argv = ["x", "--outdir", "o", "--trial-ids", "3", "9"]
            self.assertEqual(m.parse_args().trial_ids, [3, 9])
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
