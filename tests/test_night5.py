"""Night-5 changes (all opt-in; defaults unchanged).

- trace_st.retention.harmonized_retained / harmonized_case (rule 3 of the night-5 brief).
- 3_SyntheticRuns_Multivariate.py: decoy_lag, decoy_nonlinear="rectified", extra_mix (T3); defaults identical to v11.
- --force-prob-rule in the run scripts (draws unchanged; default None = original path).
"""
import unittest

import numpy as np
import pandas as pd

from trace_st import retention


class TestHarmonizedRetention(unittest.TestCase):
    def test_two_thirds_exact(self):
        # 20/30 passes (exactly 2/3), 19/30 fails; length 288/360 = 0.8 passes, 287 fails
        r = retention.harmonized_retained([20, 19, 30, 30], [30, 30, 30, 30], [300, 300, 288, 287], 360)
        self.assertEqual(list(r), [True, False, True, False])

    def test_nan_is_dropped(self):
        r = retention.harmonized_retained([30, np.nan], [30, 30], [np.nan, 300], 360)
        self.assertEqual(list(r), [False, False])

    def test_sweep_thresholds(self):
        r = retention.harmonized_retained([15, 14], [30, 30], [30, 30], 30, min_member_frac=0.5, min_len_frac=0.9)
        self.assertEqual(list(r), [True, False])

    def test_case_pnw_counts_start_point(self):
        t = pd.DataFrame(dict(n_success=[30, 30], n_requested=[30, 30], median_steps=[25, 24]))
        # length = median_steps - 1: 24 >= 0.8 * 30 passes, 23 fails (= the old PNW length clause, median_steps >= 25)
        self.assertEqual(list(retention.harmonized_case("pnw", t)), [True, False])

    def test_case_debby_uses_n_steps_time(self):
        t = pd.DataFrame(dict(n_members_success=[30, 30], n_members_requested=[30, 30],
                              median_traj_length=[13.0, 12.0], n_steps_time=[16, 16]))
        self.assertEqual(list(retention.harmonized_case("pinatubo", t)), [True, False])

    def test_existing_rules_untouched(self):
        self.assertIs(retention.RULES["debby"], retention.debby_retained)
        self.assertEqual(sorted(retention.RULES), ["debby", "pinatubo", "pnw", "t1"])


# ---------------------------------------------------------------- generator (T3) and --force-prob-rule
import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from tests.test_fraction_full_length import load_script  # noqa: E402

V11 = Path("/glade/u/home/jhayron/Causality_CaStLe/trace-st-rev1-run-v11")


def load_v11_script(name):
    p = V11 / "Scripts_Paper" / name
    spec = importlib.util.spec_from_file_location("v11_" + name.replace(".py", "").lstrip("0123456789_"), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestGeneratorNight5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script("3_SyntheticRuns_Multivariate.py")
        cls.old = load_v11_script("3_SyntheticRuns_Multivariate.py") if V11.exists() else None

    def _bytes(self, c):
        return c["full_data_run"].values.tobytes()

    @unittest.skipIf(not V11.exists(), "v11 worktree missing")
    def test_defaults_identical_to_v11(self):
        for kw in (dict(), dict(n_noise=1, n_decoy=2, n_moving=1), dict(n_decoy=1, decoy_driver="child", decoy_phi=0.5),
                   dict(n_moving=1, moving_placement="random")):
            a = self.m.build_multivariate_case(alpha_mix=0.6, **kw)
            b = self.old.build_multivariate_case(alpha_mix=0.6, **kw)
            self.assertEqual(self._bytes(a), self._bytes(b), kw)
            self.assertEqual(a["extra_vars"], b["extra_vars"])

    def test_explicit_defaults_identical(self):
        a = self.m.build_multivariate_case(alpha_mix=0.2, n_decoy=2)
        b = self.m.build_multivariate_case(alpha_mix=0.2, n_decoy=2, decoy_lag=1, decoy_nonlinear=None, extra_mix=None)
        self.assertEqual(self._bytes(a), self._bytes(b))

    def test_delayed_and_rectified(self):
        kw = dict(decoy_phi=0.0, decoy_beta=1.0, decoy_sigma=1e-12)   # D ~ beta * f(X[t-L]) only
        c = self.m.build_multivariate_case(alpha_mix=0.6, n_decoy=1, decoy_lag=2, **kw)
        D = c["full_data_run"].sel(var="D1_A").values
        A = c["A"]
        np.testing.assert_allclose(D[5], np.clip(A[3], 0, None), atol=1e-6)
        c = self.m.build_multivariate_case(alpha_mix=0.6, n_decoy=1, decoy_nonlinear="rectified", **kw)
        D = c["full_data_run"].sel(var="D1_A").values
        np.testing.assert_allclose(D[5], np.maximum(A[4] - 0.5, 0), atol=1e-6)

    def test_mix_composition(self):
        mix = ["noise", "response", "moving", "delayed", "nonlinear", "noise", "response", "moving", "delayed"]
        base = self.m.build_multivariate_case(alpha_mix=0.6)
        c = self.m.build_multivariate_case(alpha_mix=0.6, extra_mix=mix, decoy_phi=0.5, decoy_beta=0.8)
        fd = c["full_data_run"]
        self.assertEqual(fd.sizes["var"], 12)
        self.assertEqual([n for n, _ in c["extra_vars"]],
                         ["N1", "R1_B", "M1", "L1_A", "Q1_A", "N2", "R2_B", "M2", "L2_C"])
        self.assertEqual(fd.isel(var=slice(0, 3)).values.tobytes(), self._bytes(base))   # never enters B
        # noise N1 equals the original noise variable N1; moving M1 equals the original random-placement blob M1
        o = self.m.build_multivariate_case(alpha_mix=0.6, n_noise=1)
        self.assertEqual(fd.sel(var="N1").values.tobytes(), o["full_data_run"].sel(var="N1").values.tobytes())
        o = self.m.build_multivariate_case(alpha_mix=0.6, n_moving=1, moving_placement="random")
        self.assertEqual(fd.sel(var="M1").values.tobytes(), o["full_data_run"].sel(var="M1").values.tobytes())
        # prefixes of the composition are prefixes of the variables (p = 5 is the start of p = 8 ...)
        c5 = self.m.build_multivariate_case(alpha_mix=0.6, extra_mix=mix[:2], decoy_phi=0.5, decoy_beta=0.8)
        self.assertEqual(c5["full_data_run"].values.tobytes(), fd.isel(var=slice(0, 5)).values.tobytes())

    def test_mix_beta_per_type(self):
        mix = ["noise", "response", "moving", "delayed", "nonlinear"]
        a = self.m.build_multivariate_case(alpha_mix=0.2, extra_mix=mix, decoy_phi=0.5, decoy_beta=0.8,
                                           mix_beta={"delayed": 0.2})
        b = self.m.build_multivariate_case(alpha_mix=0.2, extra_mix=mix, decoy_phi=0.5, decoy_beta=0.8)
        c = self.m.build_multivariate_case(alpha_mix=0.2, extra_mix=mix, decoy_phi=0.5, decoy_beta=0.2)
        fa, fb, fc = (x["full_data_run"] for x in (a, b, c))
        for v in ("A", "B", "C", "N1", "R1_B", "M1", "Q1_A"):
            self.assertEqual(fa.sel(var=v).values.tobytes(), fb.sel(var=v).values.tobytes(), v)
        self.assertEqual(fa.sel(var="L1_A").values.tobytes(), fc.sel(var="L1_A").values.tobytes())

    def test_force_prob_rule_draws_unchanged(self):
        jobs = self.m.build_multivar_jobs(20, seed=11)
        jobs2 = self.m.build_multivar_jobs(20, seed=11)
        for (_, t), (_, t2) in zip(jobs, jobs2):
            t2["prob_rule"] = "linear"
            self.assertEqual({k: v for k, v in t.items() if k != "prob_rule"},
                             {k: v for k, v in t2.items() if k != "prob_rule"})


class TestForceProbRuleCLI(unittest.TestCase):
    """The flag exists, defaults to None (= original path) in every run script."""

    def test_flags(self):
        for name in ("5_DebbyRuns.py", "6_PinatuboRuns.py", "8_HW_PNW21_Runs_v2.py", "3_SyntheticRuns_Multivariate.py",
                     "1_SyntheticRuns_Univariate.py"):
            m = load_script(name)
            old = sys.argv
            try:
                sys.argv = [name, "--outdir", "/tmp/x"] + (["--n-trials", "2"] if "Univariate" not in name else [])
                a = m.parse_args()
                self.assertIsNone(a.force_prob_rule, name)
                sys.argv += ["--force-prob-rule", "linear"]
                self.assertEqual(m.parse_args().force_prob_rule, "linear")
            finally:
                sys.argv = old

    def test_pnw_draws_unchanged(self):
        m = load_script("8_HW_PNW21_Runs_v2.py")
        a = [t for _, t in m.build_pnw_jobs(30)]
        b = [t for _, t in m.build_pnw_jobs(30)]
        n_soft = sum(t["prob_rule"] == "softmax" for t in a)
        self.assertGreater(n_soft, 0)
        for t in b:
            t["prob_rule"] = "linear"; t["beta_softmax"] = 0.0
        for x, y in zip(a, b):
            self.assertEqual({k: v for k, v in x.items() if k not in ("prob_rule", "beta_softmax")},
                             {k: v for k, v in y.items() if k not in ("prob_rule", "beta_softmax")})


# ---------------------------------------------------------------- T5 adaptive group lasso (variant {"fit": "agl"})
class TestAGL(unittest.TestCase):
    def setUp(self):
        from trace_st import castle_core as mc
        from tests.test_night4 import toy_data, KW, parent_vars
        self.mc, self.toy, self.KW, self.parent_vars = mc, toy_data, KW, parent_vars

    def test_zero_lambda_is_ols(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((400, 6)); y = X @ np.array([1.0, -0.5, 0, 0, 0.3, 0]) + 0.1 * rng.standard_normal(400)
        w, b0 = self.mc._agl_fit(X, y, np.arange(6), np.repeat([0, 1, 2], 2), lam=0.0, fit_intercept=True, max_iter=20000, tol=1e-12)
        A = np.column_stack([np.ones(400), X]); ref = np.linalg.lstsq(A, y, rcond=None)[0]
        np.testing.assert_allclose(w, ref[1:], atol=1e-5); self.assertAlmostEqual(b0, ref[0], places=5)

    def test_group_sparsity(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((500, 6)); y = X[:, 0] + X[:, 1] + 0.2 * rng.standard_normal(500)
        w, _ = self.mc._agl_fit(X, y, np.arange(6), np.repeat([0, 1, 2], 2), lam=0.05)
        self.assertTrue(np.all(np.abs(w[:2]) > 0.5)); self.assertTrue(np.all(w[2:] == 0.0))

    def test_toy_selects_true_parent(self):
        d = self.toy(seed=0)
        r = self.mc.mv_CaStLe_GRANGER(d, variant={"fit": "agl"}, **self.KW)
        self.assertIn(1, self.parent_vars(r)); self.assertNotIn(2, self.parent_vars(r))

    def test_other_variants_unchanged(self):
        d = self.toy(seed=2)
        a = self.mc.mv_CaStLe_GRANGER(d, variant={"thin": 1}, **self.KW)
        b = self.mc.mv_CaStLe_GRANGER(d, variant={"thin": 1, "fit": "en"}, **self.KW)
        self.assertEqual(a["val_matrix"].tobytes(), b["val_matrix"].tobytes())


if __name__ == "__main__":
    unittest.main()
