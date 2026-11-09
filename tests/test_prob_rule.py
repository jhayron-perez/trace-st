"""Fix (a): run_track must pass params['prob_rule'] to cluster selection.

Run: python -m unittest discover -s tests -v   (castle env)
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from trace_st import movement, trajectory


class TestPickClusterRules(unittest.TestCase):
    def setUp(self):
        self.S = pd.Series([1.0, 2.0, 3.0, 4.0], index=[0, 1, 2, 3])

    def test_linear_is_S_over_sum(self):
        _, p = movement._pick_cluster(self.S, montecarlo=True, prob_rule="linear",
                                      rng=np.random.default_rng(0))
        np.testing.assert_allclose(p.values, self.S.values / self.S.values.sum(), rtol=0, atol=1e-15)

    def test_softmax_beta4(self):
        _, p = movement._pick_cluster(self.S, montecarlo=True, prob_rule="softmax",
                                      beta_softmax=4.0, rng=np.random.default_rng(0))
        w = np.exp(4.0 * (self.S.values - self.S.values.max()))
        np.testing.assert_allclose(p.values, w / w.sum(), rtol=0, atol=1e-15)
        self.assertFalse(np.allclose(p.values, self.S.values / self.S.values.sum()))


def _toy_data(n_t=40, ny=15, nx=15, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_t, ny, nx))
    b = np.zeros_like(a)
    b[1:] = 0.8 * np.roll(a[:-1], 1, axis=2) + 0.2 * rng.standard_normal((n_t - 1, ny, nx))
    da = np.stack([a, b])
    import xarray as xr
    return xr.DataArray(
        da, dims=("var", "time", "lat", "lon"),
        coords=dict(var=["A", "B"], time=pd.date_range("2000-01-01", periods=n_t, freq="D"),
                    lat=np.arange(ny, dtype=float), lon=np.arange(nx, dtype=float)))


BASE_PARAMS = dict(timeres="1D", spaceres=1.0, box_size=8, radius=1, starting_lat=7.0,
                   starting_lon=7.0, timewindow="20D", child_of_interest=2, n_steps_time=2,
                   montecarlo=True, beta_softmax=4.0, eps_dbscan=0.3, min_samples_dbscan=1,
                   score_mode="sum", alpha=1.0, prefer_sign="both", cd_method="granger",
                   cd_kwargs=dict(lambda_a=0.01, l1_ratio=0.5, dependence_threshold=1e-7))


class TestRunTrackPassesProbRule(unittest.TestCase):
    def _captured_rules(self, params):
        seen = []
        real = trajectory.pick_direction_then_weighted_delta

        def spy(*a, **k):
            seen.append(k.get("prob_rule", "<not passed>"))
            return real(*a, **k)

        with mock.patch.object(trajectory, "pick_direction_then_weighted_delta", side_effect=spy):
            trajectory.run_track(_toy_data(), params, date_end="2000-02-09")
        return seen

    def test_softmax_reaches_selection(self):
        seen = self._captured_rules(dict(BASE_PARAMS, prob_rule="softmax"))
        self.assertTrue(len(seen) > 0)
        self.assertTrue(all(r == "softmax" for r in seen), seen)

    def test_default_is_linear(self):
        seen = self._captured_rules(dict(BASE_PARAMS))
        self.assertTrue(len(seen) > 0)
        self.assertTrue(all(r == "linear" for r in seen), seen)


if __name__ == "__main__":
    unittest.main()
