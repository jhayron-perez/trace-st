"""Night-6 T4 synthetic p-curve changes (all opt-in; defaults unchanged).

- 3_SyntheticRuns_Multivariate.py: --extra-mix continued to p = 35 (no code change needed; checked here),
  t_pad / --t-pad (prepended days), scale_timewindow_to_p3 / --timewindow-scale-to-p3 (fixed-ratio curve).
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.test_fraction_full_length import load_script

V16 = Path("/glade/u/home/jhayron/Causality_CaStLe/trace-st-rev1-run-v16")
CYCLE = ["noise", "response", "moving", "delayed", "nonlinear"]
KW = dict(decoy_phi=0.5, decoy_beta=0.8, decoy_sigma=0.1, mix_beta={"delayed": 0.2})   # night-5 T3 arguments


def mix(p):
    return [CYCLE[i % 5] for i in range(p - 3)]


def load_v16_script(name):
    p = V16 / "Scripts_Paper" / name
    spec = importlib.util.spec_from_file_location("v16_" + name.replace(".py", "").lstrip("0123456789_"), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestExtraMixP35(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script("3_SyntheticRuns_Multivariate.py")
        cls.c35 = cls.m.build_multivariate_case(alpha_mix=0.6, extra_mix=mix(35), **KW)

    def test_names_and_distinct_fields(self):
        fd = self.c35["full_data_run"]
        names = [str(v) for v in fd["var"].values]
        self.assertEqual(len(names), 35)
        self.assertEqual(len(set(names)), 35)
        self.assertEqual([t for _, t in self.c35["extra_vars"]], mix(35))
        self.assertEqual(names[-2:], ["N7", "R7_B"])
        raw = [fd.isel(var=i).values.tobytes() for i in range(35)]
        self.assertEqual(len(set(raw)), 35)   # no two variables share a field (distinct seeds per instance)

    def test_prefix_of_p16(self):
        # p <= 16 fields are the first variables of p = 35 (night-5 runs are prefixes of the continued cycle)
        c16 = self.m.build_multivariate_case(alpha_mix=0.6, extra_mix=mix(16), **KW)
        self.assertEqual(c16["full_data_run"].values.tobytes(),
                         self.c35["full_data_run"].isel(var=slice(0, 16)).values.tobytes())

    @unittest.skipIf(not V16.exists(), "v16 worktree missing")
    def test_defaults_identical_to_v16(self):
        old = load_v16_script("3_SyntheticRuns_Multivariate.py")
        for kw in (dict(), dict(extra_mix=mix(16), **KW), dict(extra_mix=mix(35), **KW),
                   dict(n_noise=1, n_decoy=2, n_moving=1), dict(n_moving=1, moving_placement="random")):
            a = self.m.build_multivariate_case(alpha_mix=0.2, **kw)
            b = old.build_multivariate_case(alpha_mix=0.2, **kw)
            self.assertEqual(a["full_data_run"].values.tobytes(), b["full_data_run"].values.tobytes(), kw)
            self.assertEqual(a["T"], b["T"])
        a = self.m.build_multivariate_case(alpha_mix=0.2, extra_mix=mix(8), t_pad=0, **KW)
        b = old.build_multivariate_case(alpha_mix=0.2, extra_mix=mix(8), **KW)
        self.assertEqual(a["full_data_run"].values.tobytes(), b["full_data_run"].values.tobytes())


class TestTPad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script("3_SyntheticRuns_Multivariate.py")

    def test_padded_case(self):
        m, pad = self.m, 25
        base = m.build_multivariate_case(alpha_mix=0.6, extra_mix=mix(8), **KW)
        c = m.build_multivariate_case(alpha_mix=0.6, extra_mix=mix(8), t_pad=pad, **KW)
        self.assertEqual(c["T"], 40 + pad)
        self.assertEqual(c["full_data_run"].sizes["time"], 40 + pad)
        self.assertEqual(c["injection_start_idx"], 40 + pad - 8)
        np.testing.assert_array_equal(c["centers_A"][pad:], base["centers_A"])
        np.testing.assert_array_equal(c["centers_C"][pad:], base["centers_C"])
        self.assertTrue(np.all(c["centers_A"][:pad] == base["centers_A"][0]))
        # moving blob M1: same track over the original 40 days, held at its first position before
        mb = base["full_data_run"].sel(var="M1").values
        mc = c["full_data_run"].sel(var="M1").values
        self.assertEqual(mc.shape[0], mb.shape[0] + pad)
        # without noise the moving blob and A are exactly the original fields, held at t = 0 during the pad
        zero = lambda shape, **k: np.zeros(shape, dtype=np.float32)  # noqa: E731
        with mock.patch.object(m, "make_spatiotemporal_ar1_noise", zero):
            base = m.build_multivariate_case(alpha_mix=0.6, extra_mix=mix(8), **KW)
            c = m.build_multivariate_case(alpha_mix=0.6, extra_mix=mix(8), t_pad=pad, **KW)
        for v in ("M1", "A", "C", "B"):
            mb = base["full_data_run"].sel(var=v).values
            mc = c["full_data_run"].sel(var=v).values
            np.testing.assert_array_equal(mc[pad:], mb, v)
            np.testing.assert_array_equal(mc[:pad], np.repeat(mb[:1], pad, axis=0), v)
        self.assertEqual([n for n, _ in c["extra_vars"]], [n for n, _ in base["extra_vars"]])
        with self.assertRaises(ValueError):
            m.build_multivariate_case(alpha_mix=0.6, n_moving=1, t_pad=pad)

    def test_track_full_length_with_pad(self):
        # W + 29 = T + t_pad is enough for 30 steps (the --t-pad check in main)
        m, w = self.m, 30
        trial = dict(m.build_multivar_jobs(1, seed=11)[0][1], timewindow=f"{w}d", box_size=20, radius=2,
                     prob_rule="linear", lambda_a=0.05)
        params = m.build_params_from_trial_multivar(trial)
        c = m.build_multivariate_case(alpha_mix=0.6, t_pad=w + 29 - 40)
        out = m.run_ensemble(c, params, M=1, seed=3)["members"][0]
        self.assertTrue(out["ok"], out["error"])
        self.assertEqual(len(out["parents"]), 31)


class TestFixedRatio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script("3_SyntheticRuns_Multivariate.py")

    def test_scale(self):
        jobs = self.m.build_multivar_jobs(50, seed=11)
        self.assertEqual([t for _, t in self.m.scale_timewindow_to_p3(jobs, 3)],
                         [dict(t, timewindow_drawn=t["timewindow"]) for _, t in jobs])
        for p in (5, 8, 12, 16, 20, 24, 28, 35):
            for (_, t), (_, s) in zip(jobs, self.m.scale_timewindow_to_p3(jobs, p)):
                w, wp = int(t["timewindow"][:-1]), int(s["timewindow"][:-1])
                self.assertLessEqual(abs((wp - 1) - (w - 1) * p / 3), 0.5)
                self.assertEqual({k: v for k, v in s.items() if k not in ("timewindow", "timewindow_drawn")},
                                 {k: v for k, v in t.items() if k != "timewindow"})
        self.assertEqual(self.m.scale_timewindow_to_p3([(0, dict(timewindow="7d"))], 35)[0][1]["timewindow"], "71d")

    def test_ratio_matches_summary_definition(self):
        # N_samples / N_features as summarize_all_vars computes it, p = 3 vs p = 12 with the scaled window
        m = self.m
        trial = dict(m.build_multivar_jobs(1, seed=11)[0][1], timewindow="5d", box_size=30, radius=4)
        r = []
        for p in (3, 12):
            t = m.scale_timewindow_to_p3([(0, trial)], p)[0][1]
            params = m.build_params_from_trial_multivar(t)
            c = m.build_multivariate_case(alpha_mix=0.6, extra_mix=mix(p) or ["noise"], t_pad=20, **KW)
            ens = dict(members=[])
            row = m.summarize_all_vars(c, ens, params)
            r.append(row["t7_n_samples"] / (p * (2 * 4 + 1) ** 2))
        self.assertAlmostEqual(r[0], r[1], places=9)

    def test_cli_defaults_and_pad_check(self):
        m = self.m
        with mock.patch.object(sys, "argv", ["x", "--outdir", "/nonexistent_never_used"]):
            a = m.parse_args()
        self.assertFalse(a.timewindow_scale_to_p3)
        self.assertEqual(a.t_pad, 0)
        self.assertNotIn("t_pad", m.night2_case_kwargs(a))
        with tempfile.TemporaryDirectory() as d:
            argv = ["x", "--outdir", d, "--n-trials", "50", "--timewindow-scale-to-p3", "--extra-mix"] + mix(35)
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as cm:
                    m.main()
            self.assertIn("--t-pad >= 60", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
