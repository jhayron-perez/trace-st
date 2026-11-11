"""Task G (R2 l215): opt-in longer T1 record for n_steps_time = 40.

- default build_synthetic_case() is byte-identical to the pre-change code (sha256 fixture made
  at adb4bb9 by tests/grouping_fixture.py); evaluate_job default unchanged is covered by
  tests/regress_t1.py.
- n_pre > 0 prepends days before the origin: days 0..39 of A and B, the time axis and the
  truth are unchanged; prepended truth = origin; 'hold' keeps the blob at the origin.
- a 40-step run reproduces the 30-step run exactly on its first 31 centers.
"""
import unittest
from pathlib import Path

import numpy as np

from tests.grouping_fixture import t1_case_hashes
from tests.test_fraction_full_length import load_script

FIX = Path(__file__).parent / "data" / "grouping_default_adb4bb9.npz"


class TestLongerN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t1 = load_script("1_SyntheticRuns_Univariate.py")
        cls.base = cls.t1.build_synthetic_case()

    def test_default_case_equals_pre_change(self):
        self.assertEqual(str(np.load(FIX)["t1_case_sha256"]), str(t1_case_hashes()["t1_case_sha256"]))

    def test_n_pre_rule(self):
        f = self.t1.n_pre_days_for
        self.assertEqual(f(30), 0)
        self.assertEqual(f(37), 0)
        self.assertEqual(f(40), 3)
        self.assertEqual(f(40, record_length=45), 5)
        with self.assertRaises(ValueError):
            f(40, record_length=39)

    def test_extension_keeps_paper_days(self):
        fd0, sp0, end0 = self.base
        for mode in ("hold", "absent"):
            fd, sp, end = self.t1.build_synthetic_case(n_pre=3, pre_origin=mode)
            self.assertEqual(end, end0)
            self.assertEqual(fd.sizes["time"], 43)
            self.assertEqual(fd.isel(time=slice(3, None)).values.tobytes(), fd0.values.tobytes())
            np.testing.assert_array_equal(fd.time.values[3:], fd0.time.values)
            self.assertEqual(str(fd.time.values[0])[:10], "2000-12-29")
            for a, b in zip(sp, sp0):
                self.assertEqual(np.asarray(a["centers"])[3:].tobytes(), np.asarray(b["centers"]).tobytes())
                np.testing.assert_array_equal(np.asarray(a["centers"])[:3], np.repeat(np.asarray(b["centers"])[:1], 3, 0))
        # hold: blob present at each origin before day 0; absent: not
        hold = self.t1.build_synthetic_case(n_pre=3, pre_origin="hold")[0]
        absent = self.t1.build_synthetic_case(n_pre=3, pre_origin="absent")[0]
        for spec in sp0:
            y0, x0 = spec["centers"][0]
            h = float(hold.sel(var="A").isel(time=1).sel(lat=y0, lon=x0, method="nearest"))
            a = float(absent.sel(var="A").isel(time=1).sel(lat=y0, lon=x0, method="nearest"))
            self.assertGreater(h, 0.6)
            self.assertLess(a, 0.6)

    def test_40_steps_extend_30_steps(self):
        job = self.t1.build_jobs(n_per_method=1000, seed=self.t1.SEARCH_SEED)[0]
        r30 = self.t1.evaluate_job(job)
        r40 = self.t1.evaluate_job(job, n_steps=40, n_pre=self.t1.n_pre_days_for(40))
        self.assertTrue(r40["ok"], r40["job_row"]["error"])
        self.assertTrue(any(t["raw_len"] == 41 for t in r40["track_rows"]))
        for a, b in zip(r30["track_rows"], r40["track_rows"]):
            self.assertEqual(a["centers_raw"], b["centers_raw"][:31])
            self.assertEqual(a["parents_raw"], b["parents_raw"][:31])


if __name__ == "__main__":
    unittest.main()
