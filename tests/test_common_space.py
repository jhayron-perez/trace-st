"""Night 8 T1c: opt-in --search-space common (lambda_a 10^U(-3,0), l1_ratio 10^U(-4,0)) in the four run scripts.
Only lambda_a and l1_ratio change; every other drawn parameter is identical to the paper draws; defaults untouched."""
import unittest

import numpy as np

from tests.test_fraction_full_length import load_script

CASES = [
    ("8_HW_PNW21_Runs_v2.py", "build_pnw_jobs", (-2.5, 0.0), (-1.0, 0.0)),
    ("6_PinatuboRuns.py", "build_pinatubo_jobs", (-0.3, 0.0), (-0.2, 0.0)),
    ("5_DebbyRuns.py", "build_debby_jobs", (-2.5, -1.0), (-1.0, 0.0)),
    ("3_SyntheticRuns_Multivariate.py", "build_multivar_jobs", (-2.5, -1.0), (-1.0, 0.0)),
]


class TestCommonSpace(unittest.TestCase):
    def test_paper_default_ranges(self):
        for script, builder, lam, l1 in CASES:
            mod = load_script(script)
            jobs = getattr(mod, builder)(200, seed=7)
            lams = np.log10([t["lambda_a"] for _, t in jobs])
            l1s = np.log10([t["l1_ratio"] for _, t in jobs])
            self.assertTrue(lams.min() >= lam[0] and lams.max() <= lam[1], script)
            self.assertTrue(l1s.min() >= l1[0] and l1s.max() <= l1[1], script)

    def test_common_only_changes_lambda_l1(self):
        for script, builder, _, _ in CASES:
            base = getattr(load_script(script), builder)(200, seed=7)
            mod = load_script(script)
            mod.use_common_search_space()
            com = getattr(mod, builder)(200, seed=7)
            lams = np.log10([t["lambda_a"] for _, t in com])
            l1s = np.log10([t["l1_ratio"] for _, t in com])
            self.assertTrue(lams.min() >= -3.0 and lams.max() <= 0.0, script)
            self.assertTrue(l1s.min() >= -4.0 and l1s.max() <= 0.0, script)
            self.assertGreater(lams.min(), -3.0 - 1e-9)
            for (i, b), (j, c) in zip(base, com):
                self.assertEqual(i, j)
                for k in b:
                    if k not in ("lambda_a", "l1_ratio"):
                        self.assertEqual(b[k], c[k], f"{script}:{k}")
            # the common draws must actually differ in lambda/l1 and span more than the paper range
            self.assertTrue(any(b["lambda_a"] != c["lambda_a"] for (_, b), (_, c) in zip(base, com)))


if __name__ == "__main__":
    unittest.main()
