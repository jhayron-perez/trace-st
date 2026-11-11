"""Task E (R1 major 2): opt-in params['grouping'].

- default (absent) and grouping='dbscan' reproduce the pre-change code exactly
  (fixture made at adb4bb9 with tests/grouping_fixture.py): run_track toy tracks for
  granger/pcmci/dynotears + direct pick_direction_then_weighted_delta calls.
- 'strongest' == averaging_winners=False; 'variable_sign' stalls on two opposite clusters
  while dbscan picks a direction; 'sectors8' labels; 'kmeans' deterministic and splits opposites.
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trace_st import movement, trajectory
from tests.grouping_fixture import pick_outputs
from tests.native_fixture import toy_outputs
from tests.test_prob_rule import BASE_PARAMS, _toy_data

FIX = Path(__file__).parent / "data" / "grouping_default_adb4bb9.npz"


def opposite_stencil(radius=2):
    """Variable 1, positive links: 3 links to the east (strong) and 3 to the west (equal strength)."""
    s = 2 * radius + 1
    def pos(di, dj):
        return (di + radius) * s + (dj + radius)
    east = [(0, 2), (-1, 2), (1, 2)]
    west = [(0, -2), (-1, -2), (1, -2)]
    rows = [("p1_c1", 0.50, (pos(*d), 0, 1)) for d in east] + [("p1_c1", 0.45, (pos(*d), 0, 1)) for d in west]
    df = pd.DataFrame(rows, columns=["vars", "causal_value", "position"])
    df["causal_value_abs"] = df["causal_value"].abs()
    return df


KW = dict(radius=2, spaceres=1.0, eps=0.15, min_samples=2, montecarlo=False, prefer_sign="both",
          alpha=3.0, score_mode="mean")


class TestDefaultUnchanged(unittest.TestCase):
    def test_tracks_equal_pre_change(self):
        ref = np.load(FIX)
        for extra in (None, {"grouping": "dbscan"}):
            new = toy_outputs(extra)
            for k, v in new.items():
                self.assertEqual(ref[f"track_{k}"].tobytes(), v.tobytes(), f"{k} differs ({extra})")

    def test_pick_equal_pre_change(self):
        ref = np.load(FIX)
        for extra in (None, {"grouping": "dbscan"}):
            for k, v in pick_outputs(extra).items():
                self.assertEqual(ref[k].tobytes(), v.tobytes(), f"{k} differs ({extra})")


class TestGroupings(unittest.TestCase):
    def test_strongest_equals_averaging_false(self):
        p = dict(BASE_PARAMS, n_steps_time=4, seed=5)
        a = trajectory.run_track(_toy_data(seed=3), dict(p, grouping="strongest"), date_end="2000-02-09",
                                 return_debug=False, return_cluster_summaries=False)
        b = trajectory.run_track(_toy_data(seed=3), dict(p, averaging_winners=False), date_end="2000-02-09",
                                 return_debug=False, return_cluster_summaries=False)
        self.assertEqual(np.asarray(a["centers"]).tobytes(), np.asarray(b["centers"]).tobytes())
        self.assertEqual(np.asarray(a["parents"]).tobytes(), np.asarray(b["parents"]).tobytes())

    def test_variable_sign_stalls_dbscan_moves(self):
        df = opposite_stencil()
        d_lat, d_lon, *_ = movement.pick_direction_then_weighted_delta(df, **KW)
        self.assertAlmostEqual(d_lon, 2.0)  # dbscan: east cluster wins, full step east
        v_lat, v_lon, *_ = movement.pick_direction_then_weighted_delta(df, grouping="variable_sign", **KW)
        self.assertLess(abs(v_lon), 0.5)    # merged mean ~ 0.25 (only the strength imbalance)
        self.assertAlmostEqual(v_lat, 0.0)
        s_lat, s_lon, *_ = movement.pick_direction_then_weighted_delta(df, grouping="sectors8", **KW)
        k_lat, k_lon, *_ = movement.pick_direction_then_weighted_delta(df, grouping="kmeans", **KW)
        self.assertAlmostEqual(k_lon, 2.0)
        self.assertGreater(s_lon, 1.9)      # E sector (exactly east link), or E sector incl. NE/SE

    def test_sectors8_labels(self):
        ang = np.deg2rad([0, 22, 23, 45, 90, 180, -90, -23, 340])
        df = pd.DataFrame({"u": np.cos(ang), "v": np.sin(ang)})
        np.testing.assert_array_equal(movement._sectors8_labels(df), [0, 0, 1, 1, 2, 4, 6, 7, 0])

    def test_kmeans_deterministic_and_bounded(self):
        rng = np.random.default_rng(0)
        ang = np.concatenate([rng.normal(0, 0.1, 10), rng.normal(np.pi, 0.1, 10)])
        df = pd.DataFrame({"u": np.cos(ang), "v": np.sin(ang)})
        a = movement._kmeans_labels(df, seed=3)
        b = movement._kmeans_labels(df, seed=3)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(a)), 2)
        self.assertEqual(len(np.unique(a[:10])), 1)
        ang = rng.uniform(-np.pi, np.pi, 60)
        lab = movement._kmeans_labels(pd.DataFrame({"u": np.cos(ang), "v": np.sin(ang)}), seed=1)
        self.assertLessEqual(len(np.unique(lab)), 6)
        one = movement._kmeans_labels(pd.DataFrame({"u": [1.0, 1.0], "v": [0.0, 0.0]}), seed=1)
        np.testing.assert_array_equal(one, [0, 0])

    def test_all_groupings_run_track(self):
        for g in ("variable_sign", "sectors8", "kmeans"):
            out = trajectory.run_track(_toy_data(seed=3), dict(BASE_PARAMS, n_steps_time=3, seed=5, grouping=g),
                                       date_end="2000-02-09")
            self.assertEqual(len(out["centers"]), 4, g)

    def test_bad_grouping_raises(self):
        with self.assertRaises(ValueError):
            trajectory.run_track(_toy_data(seed=3), dict(BASE_PARAMS, grouping="nope"), date_end="2000-02-09")


class TestScriptPlumbing(unittest.TestCase):
    """--grouping reaches run_track params only when not dbscan; T1 job selection keeps paper ids."""

    @staticmethod
    def _capture(mod, call):
        from unittest import mock
        seen = []

        def fake(full_data, params, **k):
            seen.append(dict(params))
            return {"centers": [[float(params["starting_lat"]), float(params["starting_lon"])]], "parents": [1],
                    "winners": [], "debug": []}

        with mock.patch.object(mod.tst.trajectory, "run_track", side_effect=fake):
            try:
                call()
            except Exception:
                pass
        return seen

    def test_t1(self):
        from tests.test_fraction_full_length import load_script
        t1 = load_script("1_SyntheticRuns_Univariate.py")
        jobs = t1.build_jobs(n_per_method=1000, seed=t1.SEARCH_SEED)
        sel = t1.select_jobs(jobs, ["granger"], 200)
        self.assertEqual([j[1] for j in sel], list(range(200)))
        self.assertTrue(all(a is b for a, b in zip(sel, jobs[:200])))
        sel2 = t1.select_jobs(jobs, ["pcmci"], None, t1.parse_trial_ids("3,1,500"))
        self.assertEqual([(m, i) for m, i, _ in sel2], [("pcmci", 1), ("pcmci", 3), ("pcmci", 500)])
        self.assertEqual(sel2[0][2], jobs[1001][2])
        job = jobs[0]
        d = self._capture(t1, lambda: t1.evaluate_job(job))
        self.assertTrue(len(d) == 3 and all("grouping" not in p for p in d))
        g = self._capture(t1, lambda: t1.evaluate_job(job, "native", "kmeans"))
        self.assertTrue(len(g) == 3 and all(p["grouping"] == "kmeans" for p in g))

    def test_t2(self):
        from tests.test_fraction_full_length import load_script
        t2 = load_script("3_SyntheticRuns_Multivariate.py")
        tid, trial = t2.build_multivar_jobs(1)[0]
        run = lambda **k: t2.evaluate_trial_across_alphas_full(tid, trial, alpha_grid=[0.6], M=2, **k)
        d = self._capture(t2, lambda: run())
        self.assertTrue(len(d) > 0 and all("grouping" not in p for p in d))
        g = self._capture(t2, lambda: run(grouping="variable_sign"))
        self.assertTrue(len(g) > 0 and all(p["grouping"] == "variable_sign" for p in g))


if __name__ == "__main__":
    unittest.main()
