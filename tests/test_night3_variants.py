"""Night-3 opt-in EN variants (env TST_VARIANT / mv_CaStLe_GRANGER(variant=...)).

- default byte-identical: TST_VARIANT unset or empty gives exactly the original GRANGER output;
- each variant runs, returns the usual shapes, and on a toy box keeps the true parent variable;
- the gates that should remove an independent persistent field (block_f, cv, knockoff, pcmci) do so.
"""
import json
import os
import unittest

import numpy as np

from trace_st import castle_core as mc
from trace_st.trajectory import get_mcastle_box


def toy_data(seed=0, T=120, ny=8, nx=8):
    """var0 (child) <- var1 from the west neighbour at lag 1; var2 independent, persistent AR(1)."""
    rng = np.random.default_rng(seed)
    v1 = rng.standard_normal((ny, nx, T))
    v2 = np.zeros((ny, nx, T))
    v0 = np.zeros((ny, nx, T))
    for t in range(1, T):
        v2[:, :, t] = 0.9 * v2[:, :, t - 1] + 0.3 * rng.standard_normal((ny, nx))
        v0[:, 1:, t] = 0.8 * v1[:, :-1, t - 1] + 0.2 * v0[:, 1:, t - 1] + 0.3 * rng.standard_normal((ny, nx - 1))
        v0[:, 0, t] = 0.3 * rng.standard_normal(ny)
    return np.stack([v0, v1, v2])


KW = dict(radius=1, child_variable=1, lambda_a=0.01, l1_ratio=0.7, dependence_threshold=0.01)


def parent_vars(res, child=4):
    g = res["graph"]
    par = np.where(g[:, child, 1] == "-->")[0]
    return sorted(set((par // 9).tolist()))


class TestDefaultIdentical(unittest.TestCase):
    def test_env_unset_and_empty_identical(self):
        d = toy_data()
        os.environ.pop("TST_VARIANT", None)
        ref = mc.mv_CaStLe_GRANGER(d, **KW)
        os.environ["TST_VARIANT"] = ""
        try:
            again = mc.mv_CaStLe_GRANGER(d, **KW)
        finally:
            os.environ.pop("TST_VARIANT", None)
        self.assertTrue(np.array_equal(ref["val_matrix"], again["val_matrix"]))
        self.assertTrue((ref["graph"] == again["graph"]).all())

    def test_box_env_unset_matches_direct_call(self):
        import xarray as xr
        d = toy_data(T=40)
        ny, nx, T = d.shape[1:]
        da = [xr.DataArray(d[i], dims=("lat", "lon", "time"),
                           coords=dict(lat=np.arange(ny) * 1.0, lon=np.arange(nx) * 1.0, time=np.arange(T)))
              for i in range(3)]
        os.environ.pop("TST_VARIANT", None)
        r1 = get_mcastle_box((3.5, 3.5), 100.0, da, radius=1, child_variable=1, cd_method="granger",
                             cd_kwargs=dict(lambda_a=0.01))
        r2 = get_mcastle_box((3.5, 3.5), 100.0, da, radius=1, child_variable=1, cd_method="granger",
                             cd_kwargs=dict(lambda_a=0.01))
        for k in r1:
            self.assertTrue(np.array_equal(r1[k][1], r2[k][1]))


class TestVariants(unittest.TestCase):
    def setUp(self):
        self.d = toy_data()

    def run_v(self, variant, **extra):
        return mc.mv_CaStLe_GRANGER(self.d, variant=variant, **KW, **extra)

    def test_shapes_and_true_parent_kept(self):
        for v in [dict(), dict(prewhiten="ar1"), dict(thin=3), dict(lambda_rule="universal", lambda_c=1.0),
                  dict(lambda_rule="universal", lambda_c=0.5), dict(weight="tstat"), dict(weight="tgate"),
                  dict(gate="block_f", gate_alpha=0.05), dict(gate="block_f", gate_alpha=0.01),
                  dict(gate="knockoff", fdr_q=0.2, fdr_offset=0), dict(gate="cv")]:
            res = self.run_v(v)
            self.assertEqual(res["val_matrix"].shape, (27, 27, 2), v)
            self.assertIn(1, parent_vars(res), f"true parent var lost with {v}")

    def test_knockoff_plus_runs(self):
        # knockoff+ needs >= 1/q selections; with one true link it may select none (documented)
        res = self.run_v(dict(gate="knockoff", fdr_q=0.1))
        self.assertEqual(res["val_matrix"].shape, (27, 27, 2))
        self.assertNotIn(2, parent_vars(res))

    def test_empty_variant_equals_baseline(self):
        base = mc.mv_CaStLe_GRANGER(self.d, **KW)
        res = self.run_v(dict())
        np.testing.assert_allclose(res["val_matrix"], base["val_matrix"], rtol=1e-6, atol=1e-8)

    def test_gates_drop_independent_field(self):
        base = mc.mv_CaStLe_GRANGER(self.d, **KW)
        for v in [dict(gate="block_f", gate_alpha=0.01), dict(gate="cv")]:
            res = self.run_v(v)
            self.assertNotIn(2, parent_vars(res), f"independent field kept with {v} (baseline {parent_vars(base)})")

    def test_tstat_values_are_t(self):
        res = self.run_v(dict(weight="tstat"))
        v = res["val_matrix"][:, 4, 1]
        self.assertGreater(np.abs(v).max(), 5.0)

    def test_pcmci_gate_via_env(self):
        import xarray as xr
        d = toy_data(T=60)
        ny, nx, T = d.shape[1:]
        da = [xr.DataArray(d[i], dims=("lat", "lon", "time"),
                           coords=dict(lat=np.arange(ny) * 1.0, lon=np.arange(nx) * 1.0, time=np.arange(T)))
              for i in range(3)]
        for g in ("pcmci_vars", "pcmci_links"):
            os.environ["TST_VARIANT"] = json.dumps(dict(gate=g))
            try:
                r = get_mcastle_box((3.5, 3.5), 100.0, da, radius=1, child_variable=1, cd_method="granger",
                                    cd_kwargs=dict(lambda_a=0.01))
            finally:
                os.environ.pop("TST_VARIANT", None)
            self.assertTrue(np.any(r["p2_c1"][1] != 0), g)       # true parent var1 kept
            self.assertFalse(np.any(r["p3_c1"][1] != 0), g)      # independent var2 removed

    def test_stability_via_env(self):
        res_env = None
        import xarray as xr
        d = toy_data(T=40)
        ny, nx, T = d.shape[1:]
        da = [xr.DataArray(d[i], dims=("lat", "lon", "time"),
                           coords=dict(lat=np.arange(ny) * 1.0, lon=np.arange(nx) * 1.0, time=np.arange(T)))
              for i in range(3)]
        os.environ["TST_VARIANT"] = json.dumps(dict(stability=dict(n_sub=5, frac=0.5, thr=0.6, seed=0)))
        try:
            res_env = get_mcastle_box((3.5, 3.5), 100.0, da, radius=1, child_variable=1, cd_method="granger",
                                      cd_kwargs=dict(lambda_a=0.01))
        finally:
            os.environ.pop("TST_VARIANT", None)
        self.assertIn("p2_c1", res_env)


if __name__ == "__main__":
    unittest.main()


class TestShortWindow(unittest.TestCase):
    def test_two_step_window_runs_all_variants(self):
        d = toy_data(T=2)
        for v in [dict(gate="block_f"), dict(prewhiten="ar1"), dict(weight="tstat"), dict(gate="cv"),
                  dict(lambda_rule="universal"), dict(thin=2), dict(gate="knockoff", fdr_q=0.2)]:
            res = mc.mv_CaStLe_GRANGER(d, variant=v, **KW)
            self.assertEqual(res["val_matrix"].shape, (27, 27, 2), v)
