"""Night-7 T3b: linkless surrogate system of the synthetic case (data_processing/surrogates/make_linkless_synthetic.py)
and the opt-in --linkless of 3_SyntheticRuns_Multivariate.py (default off = original case)."""
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


def _fields(seed=3, nt=300, ny=40, nx=50, r=0.6):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((nt, ny, nx)).astype(np.float32)
    other = rng.standard_normal((nt, ny, nx)).astype(np.float32)
    x1 = 2.0 + 0.5 * base
    x2 = 1.0 + 0.3 * (r * base + np.sqrt(1 - r ** 2) * other)
    return [x1, x2]


class LinklessSyntheticTest(unittest.TestCase):
    def test_A_keeps_correlation_indep_removes_it(self):
        arrs = _fields()
        outA, repA = ll.linkless_system(arrs, ["a", "b"], 0, "A", clip=False)
        outI, repI = ll.linkless_system(arrs, ["a", "b"], 0, "indep", clip=False)
        self.assertLess(repA["max_abs_corr_diff_vs_R"], 0.1)
        self.assertLess(repI["max_abs_corr_diff"], 0.1)
        for v in range(2):
            self.assertAlmostEqual(float(outA[v].mean()), float(arrs[v].mean()), delta=0.05)
            self.assertAlmostEqual(float(outA[v].std()), float(arrs[v].std()), delta=0.1 * float(arrs[v].std()))

    def test_common_draw_seeds_and_no_lagged_link(self):
        arrs = _fields()
        o1, _ = ll.linkless_system(arrs, ["a", "b"], 1, "A", clip=False)
        o2, _ = ll.linkless_system(arrs, ["a", "b"], 1, "A", clip=False)
        np.testing.assert_array_equal(o1[0], o2[0])
        # white-in-time innovations with the same R: lagged cross-correlation equals rho_a * rho_b-free zero-lag structure
        za = (o1[0] - o1[0].mean(0)).reshape(len(o1[0]), -1)
        zb = (o1[1] - o1[1].mean(0)).reshape(len(o1[1]), -1)
        c0 = np.corrcoef(za.ravel(), zb.ravel())[0, 1]
        c1 = np.corrcoef(za[:-1].ravel(), zb[1:].ravel())[0, 1]
        self.assertLess(abs(c1), abs(c0) + 1e-6)

    def test_default_case_unchanged_and_opt_in_replaces_variables(self):
        m = load_script("3_SyntheticRuns_Multivariate.py")
        base = m.build_multivariate_case(alpha_mix=0.6, extra_mix=["noise", "response"], seed=11)
        again = m.build_multivariate_case(alpha_mix=0.6, extra_mix=["noise", "response"], seed=11)
        np.testing.assert_array_equal(base["full_data_run"].values, again["full_data_run"].values)
        sur = m.build_multivariate_case(alpha_mix=0.6, extra_mix=["noise", "response"], seed=11,
                                        linkless=dict(variant="A", draw=0))
        self.assertEqual(list(sur["full_data_run"]["var"].values), list(base["full_data_run"]["var"].values))
        self.assertEqual(sur["full_data_run"].shape, base["full_data_run"].shape)
        self.assertFalse(np.allclose(sur["full_data_run"].values, base["full_data_run"].values))


if __name__ == "__main__":
    unittest.main()
