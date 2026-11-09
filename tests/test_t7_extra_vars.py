"""T7 opt-in flags in 3_SyntheticRuns_Multivariate.py: default case unchanged; extras appended and non-causal."""
import unittest

import numpy as np

from tests.test_fraction_full_length import load_script


class TestT7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script("3_SyntheticRuns_Multivariate.py")

    def test_default_case_identical(self):
        a = self.m.build_multivariate_case(alpha_mix=0.6)
        b = self.m.build_multivariate_case(alpha_mix=0.6, n_noise=0, n_decoy=0, n_moving=0)
        self.assertEqual(a["full_data_run"].sizes["var"], 3)
        self.assertEqual(a["full_data_run"].values.tobytes(), b["full_data_run"].values.tobytes())

    def test_extras(self):
        base = self.m.build_multivariate_case(alpha_mix=0.6)
        c = self.m.build_multivariate_case(alpha_mix=0.6, n_noise=5, n_decoy=4, n_moving=4)
        fd = c["full_data_run"]
        self.assertEqual(fd.sizes["var"], 16)
        # first three variables (A, B, C) unchanged -> extras never enter B
        self.assertEqual(fd.isel(var=slice(0, 3)).values.tobytes(), base["full_data_run"].values.tobytes())
        # moving blobs pass within ~20 units of the event during t0-5..t0
        t0 = c["injection_start_idx"]
        for k in range(4):
            v = fd.sel(var=f"M{k + 1}").values
            dmin = []
            for t in range(t0 - 5, t0 + 1):
                clean = v[t] - 0  # blob + noise; take argmax location
                iy, ix = np.unravel_index(np.argmax(clean), clean.shape)
                dmin.append(np.hypot(c["ys"][iy] - 100.0, c["xs"][ix] - 100.0))
            self.assertLess(min(dmin), 22.0)

    def test_summary(self):
        c = self.m.build_multivariate_case(alpha_mix=0.6, n_noise=1, n_decoy=1, n_moving=0)
        p = self.m.build_base_params_multivar(); p["box_size"] = 30; p["radius"] = 4; p["timewindow"] = "6d"
        ens = {"members": [dict(member_id=0, ok=True, elapsed_sec=1.0,
                                centers=np.array([[100, 100], [101, 99], [102, 98]], float),
                                parents=np.array([2, 1, 4]))]}
        r = self.m.summarize_all_vars(c, ens, p)
        self.assertEqual(r["t7_p"], 5)
        self.assertAlmostEqual(r["t7_share_density_v1v3"] + r["t7_share_density_target"]
                               + r["t7_share_density_noise"] + r["t7_share_density_decoy"], 1.0)
        self.assertGreater(r["t7_samples_per_feature"], 0)


if __name__ == "__main__":
    unittest.main()
