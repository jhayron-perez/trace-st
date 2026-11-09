"""Fix (e): opt-in strength_mode.

- "native" (and absent) gives exactly the outputs of the pre-refit code (fixture made at 035116f
  with tests/native_fixture.py, same seed) for granger, pcmci and dynotears.
- "refit": standardized OLS on the engine-selected parents recovers known coefficients on a toy system.
"""
import unittest
from pathlib import Path

import numpy as np

from trace_st import castle_core as mcastle
from tests.native_fixture import toy_outputs

FIX = Path(__file__).parent / "data" / "native_toy_035116f.npz"


class TestNativeUnchanged(unittest.TestCase):
    def test_native_equals_pre_commit(self):
        ref = np.load(FIX)
        for extra in (None, {"strength_mode": "native"}):
            new = toy_outputs(extra)
            for k in ref.files:
                self.assertEqual(ref[k].tobytes(), new[k].tobytes(), f"{k} differs ({extra})")

    def test_refit_runs_and_changes_values(self):
        ref = np.load(FIX)
        new = toy_outputs({"strength_mode": "refit"})
        self.assertFalse(np.array_equal(ref["granger_winner_vals"][:1], new["granger_winner_vals"][:1]))


class TestRefitOLS(unittest.TestCase):
    def test_recovers_known_coefficients(self):
        rng = np.random.default_rng(0)
        V, NY, NX, T, r = 2, 12, 12, 400, 1
        a = rng.standard_normal((NY, NX, T))
        b = np.zeros((NY, NX, T))
        # child B(i,j,t) = 0.6 A(i,j-1,t-1) - 0.4 A(i+1,j,t-1) + 0.3 B(i,j,t-1) + noise
        b[:, :, 0] = rng.standard_normal((NY, NX))
        for t in range(1, T):
            b[:, :, t] = (0.6 * np.roll(a[:, :, t - 1], 1, axis=1)
                          - 0.4 * np.roll(a[:, :, t - 1], -1, axis=0)
                          + 0.3 * b[:, :, t - 1] + 0.5 * rng.standard_normal((NY, NX)))
        data = np.stack([a, b])
        rs = mcastle.get_MV_reduced_space(data, False, True, r)
        N = rs.shape[1]
        s = (2 * r + 1) ** 2
        child = s + s // 2  # center node of variable 2
        # identify true parent columns by regressing on all nodes (unique max-|coef| picks)
        R = rs.shape[0] // T
        idx = np.arange(R * T).reshape(R, T)
        X = rs[idx[:, :-1].ravel()]
        y = rs[idx[:, 1:].ravel(), child]
        coef = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)[0][:N]
        parents = np.argsort(-np.abs(coef))[:3]
        graph = np.empty((N, N, 2), dtype=object); graph[:] = ""
        graph[parents, child, 1] = "-->"
        val = np.zeros((N, N, 2)); val[parents, child, 1] = 99.0  # engine-native placeholder
        out = mcastle.refit_link_strengths(data, graph, val, radius=r, child_variable=2)
        got = out[parents, child, 1]
        Xp = X[:, parents]
        b_ols = np.linalg.lstsq(np.column_stack([Xp, np.ones(len(Xp))]), y, rcond=None)[0][:3]
        expect = b_ols * Xp.std(axis=0) / y.std()  # standardized OLS on the selected parents
        np.testing.assert_allclose(got, expect, atol=1e-8)
        # and the unstandardized values are the true ones (0.6, -0.4, 0.3 up to sampling error)
        unstd = got * y.std() / X[:, parents].std(axis=0)
        np.testing.assert_allclose(np.sort(unstd), np.sort([0.6, -0.4, 0.3]), atol=0.03)
        # untouched entries stay as they were
        mask = np.ones_like(val, bool); mask[parents, child, 1] = False
        self.assertTrue(np.array_equal(out[mask], val[mask]))


if __name__ == "__main__":
    unittest.main()
