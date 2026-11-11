"""Night-2 opt-in generator flags in 3_SyntheticRuns_Multivariate.py (C1 target driver, C2 parent sigma,
D2 random moving placement, D3 offset / timing / W override). Defaults must be byte-identical to the
pre-change generator (hashes computed with commit 1c7aaa2's file)."""
import argparse
import hashlib
import unittest

import numpy as np

from tests.test_fraction_full_length import load_script

PRE = {  # sha256(full_data_run.values)[:24] from the pre-change generator (1c7aaa2)
    (): "ca2906d281407e9abd693430",
    (("n_decoy", 1),): "395cc2bebff9a645643e91d6",
    (("n_noise", 3), ("n_moving", 2)): "0162e00d1801e12a03515a21",
    (("alpha_mix", 0.2), ("n_decoy", 2), ("n_moving", 6)): "a17ce85fd5603fe52df2235c",
}


def h(case):
    return hashlib.sha256(case["full_data_run"].values.tobytes()).hexdigest()[:24]


class TestNight2Generator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script("3_SyntheticRuns_Multivariate.py")

    def test_defaults_byte_identical(self):
        for kw, ref in PRE.items():
            self.assertEqual(h(self.m.build_multivariate_case(**dict(kw))), ref, kw)
        # explicit defaults are also identical
        c = self.m.build_multivariate_case(n_noise=3, n_moving=2, target_driver="clean", parent_sigma=0.10,
                                           moving_placement="event", moving_offset=15.0, moving_tc_lag=2.0)
        self.assertEqual(h(c), PRE[(("n_noise", 3), ("n_moving", 2))])

    def test_observed_driver(self):
        a = 0.6
        clean = self.m.build_multivariate_case(alpha_mix=a)
        obs = self.m.build_multivariate_case(alpha_mix=a, target_driver="observed")
        np.testing.assert_array_equal(clean["A"], obs["A"])
        np.testing.assert_array_equal(clean["C"], obs["C"])
        t0, tau = obs["injection_start_idx"], obs["tau_mix"]
        # noise_B = B before injection is identical; reconstruct B from the observed A, C
        noise_B = self.m.make_spatiotemporal_ar1_noise(obs["A"].shape, phi=0.80, sigma=0.10,
                                                       rng=np.random.default_rng(11 + 1))
        for t in (t0, obs["T"] - 1):
            exp = np.clip(a * obs["A"][t - tau] + (1 - a) * obs["C"][t - tau] + noise_B[t], 0, None)
            np.testing.assert_allclose(obs["B"][t], exp, rtol=0, atol=1e-6)
        self.assertGreater(float(np.abs(obs["B"][t0] - clean["B"][t0]).max()), 1e-3)
        np.testing.assert_array_equal(obs["B"][: t0], clean["B"][: t0])

    def test_parent_sigma(self):
        c1 = self.m.build_multivariate_case(parent_sigma=0.2)
        c0 = self.m.build_multivariate_case()
        np.testing.assert_array_equal(c1["A_clean"], c0["A_clean"])
        r = np.std(c1["A"] - c1["A_clean"]) / np.std(c0["A"] - c0["A_clean"])
        self.assertGreater(r, 1.5)

    def test_moving_placement_random_and_offsets(self):
        ev = self.m.build_multivariate_case(n_moving=2)
        rd = self.m.build_multivariate_case(n_moving=2, moving_placement="random")
        rd2 = self.m.build_multivariate_case(n_moving=2, moving_placement="random")
        self.assertEqual(h(rd), h(rd2))
        np.testing.assert_array_equal(ev["full_data_run"].values[:, :3], rd["full_data_run"].values[:, :3])
        self.assertNotEqual(h(ev), h(rd))
        far = self.m.build_multivariate_case(n_moving=2, moving_offset=50.0)
        early = self.m.build_multivariate_case(n_moving=2, moving_tc_lag=15.0)
        for c in (far, early):
            np.testing.assert_array_equal(ev["full_data_run"].values[:, :3], c["full_data_run"].values[:, :3])
            self.assertNotEqual(h(ev), h(c))
        # far placement: moving blob stays >= ~50 units from the event at closest approach
        ys, xs = ev["ys"], ev["xs"]; ey, ex = ev["event"]
        Mf = self.m.build_multivariate_case(n_moving=1, moving_offset=50.0, clip_to_positive=False)
        # locate blob peak per time on the clean part by argmax of the smoothed field
        dmin = min(np.hypot(*(np.array(np.unravel_index(np.argmax(Mf["full_data_run"].values[t, 3]), (len(ys), len(xs))))
                               * [ys[1] - ys[0], xs[1] - xs[0]] - [ey, ex])) for t in range(Mf["T"]))
        self.assertGreater(dmin, 40.0)

    def test_cli_defaults_and_w_override(self):
        ns = argparse.Namespace(target_driver="clean", parent_sigma=None, moving_placement="event",
                                moving_offset=15.0, moving_tc_lag=2.0)
        self.assertEqual(self.m.night2_case_kwargs(ns), {})
        jobs = self.m.build_multivar_jobs(5, seed=11)
        j2 = self.m.override_timewindow(jobs, ["10d", "11d", "12d"], 11)
        for (i, a), (k, b) in zip(jobs, j2):
            self.assertEqual(i, k)
            self.assertIn(b["timewindow"], ("10d", "11d", "12d"))
            self.assertEqual({x: y for x, y in a.items() if x != "timewindow"},
                             {x: y for x, y in b.items() if x != "timewindow"})


if __name__ == "__main__":
    unittest.main()


class TestNight2PCMCIHook(unittest.TestCase):
    def test_pcmci_generator_same_base_and_hook(self):
        p = load_script("3_SyntheticRuns_Multivariate_PCMCI_Dyno.py")
        self.assertEqual(h(p.build_multivariate_case()), PRE[()])          # PCMCI's own copy = EN default
        self.assertEqual(h(p._en_module().build_multivariate_case()), PRE[()])
        ns = argparse.Namespace(n_noise=0, n_decoy=0, n_moving=0, decoy_phi=0.8, decoy_beta=0.8,
                                decoy_sigma=0.1, target_driver="clean")
        self.assertEqual(p.night2_case_kwargs(ns), {})                      # default: original code path
        ns.n_decoy = 1; ns.target_driver = "observed"
        kw = p.night2_case_kwargs(ns)
        self.assertEqual(kw, dict(n_decoy=1, decoy_phi=0.8, decoy_beta=0.8, decoy_sigma=0.1, target_driver="observed"))
        c = p._en_module().build_multivariate_case(**kw)
        self.assertEqual(list(c["full_data_run"]["var"].values), ["A", "B", "C", "D1_A"])
