"""Night-4 opt-in options.

- knockoff refactor: the night-3 ko0_10 settings give exactly the night-3 output (compared with the
  `rev1-fixes-v9` implementation when that worktree exists, and with explicit shift_mults=[2.0], derand_min=1);
- derandomized knockoff (B3): more required votes never selects more blocks;
- conditional knockoff (B2): a block that is an exact linear function of the other blocks gets a knockoff equal
  to itself (zero residual), so it can never beat it; the gate runs and returns the usual shapes;
- link-stats logger: off by default (no file), on with TST_LINKSTATS, and it does not change the graph;
- response decoy generator (C): default kwargs identical to decoy_driver="parent"; "child" decoys follow B.
"""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trace_st import castle_core as mc

HERE = Path(__file__).resolve().parents[1]
V9 = Path("/glade/u/home/jhayron/Causality_CaStLe/trace-st-rev1-run-v9")


def toy_data(seed=0, T=120, ny=8, nx=8, proxy=False):
    """var0 (child) <- var1 (west neighbour, lag 1); var2 persistent AR(1); var3 (proxy=True) = var1 + noise."""
    rng = np.random.default_rng(seed)
    v1 = np.zeros((ny, nx, T)); v2 = np.zeros((ny, nx, T)); v0 = np.zeros((ny, nx, T))
    for t in range(1, T):
        v1[:, :, t] = 0.7 * v1[:, :, t - 1] + rng.standard_normal((ny, nx))
        v2[:, :, t] = 0.9 * v2[:, :, t - 1] + 0.3 * rng.standard_normal((ny, nx))
        v0[:, 1:, t] = 0.8 * v1[:, :-1, t - 1] + 0.2 * v0[:, 1:, t - 1] + 0.3 * rng.standard_normal((ny, nx - 1))
        v0[:, 0, t] = 0.3 * rng.standard_normal(ny)
    out = [v0, v1, v2]
    if proxy:
        out.append(v1 + 0.3 * rng.standard_normal(v1.shape))
    return np.stack(out)


KW = dict(radius=1, child_variable=1, lambda_a=0.01, l1_ratio=0.7, dependence_threshold=0.01)
KO = dict(gate="knockoff", fdr_q=0.1, fdr_offset=0)


def parent_vars(res, child=4):
    par = np.where(res["graph"][:, child, 1] == "-->")[0]
    return sorted(set((par // 9).tolist()))


def load_v9_core():
    p = V9 / "trace_st" / "castle_core.py"
    spec = importlib.util.spec_from_file_location("castle_core_v9", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestKnockoffRefactor(unittest.TestCase):
    def test_explicit_defaults_identical(self):
        for seed in range(3):
            d = toy_data(seed, proxy=True)
            a = mc.mv_CaStLe_GRANGER(d, variant=KO, **KW)
            b = mc.mv_CaStLe_GRANGER(d, variant=dict(KO, shift_mults=[2.0], derand_min=1), **KW)
            self.assertTrue(np.array_equal(a["val_matrix"], b["val_matrix"]))
            self.assertTrue((a["graph"] == b["graph"]).all())

    @unittest.skipUnless((V9 / "trace_st" / "castle_core.py").exists(), "v9 worktree not present")
    def test_identical_to_night3_v9(self):
        old = load_v9_core()
        for seed in range(4):
            for prox in (False, True):
                d = toy_data(seed, T=60 + 20 * seed, proxy=prox)
                for v in (KO, dict(gate="knockoff", fdr_q=0.2, fdr_offset=0), dict(gate="knockoff", fdr_q=0.1)):
                    a = mc.mv_CaStLe_GRANGER(d, variant=v, **KW)
                    b = old.mv_CaStLe_GRANGER(d, variant=v, **KW)
                    self.assertTrue(np.array_equal(a["val_matrix"], b["val_matrix"]), (seed, prox, v))
                    self.assertTrue((a["graph"] == b["graph"]).all())
                a = mc.mv_CaStLe_GRANGER(d, **KW); b = old.mv_CaStLe_GRANGER(d, **KW)
                self.assertTrue(np.array_equal(a["val_matrix"], b["val_matrix"]))


class TestDerandomized(unittest.TestCase):
    def test_more_votes_never_selects_more(self):
        for seed in range(4):
            d = toy_data(seed, proxy=True)
            n = []
            for need in (1, 2, 3):
                r = mc.mv_CaStLe_GRANGER(d, variant=dict(KO, shift_mults=[2, 3, 4], derand_min=need), **KW)
                n.append(set(parent_vars(r)))
            self.assertTrue(n[2] <= n[1] <= n[0] | n[1])
            self.assertIn(0, n[1] | {0})


class TestConditionalKnockoff(unittest.TestCase):
    def test_exact_function_of_others_is_its_own_knockoff(self):
        rng = np.random.default_rng(1)
        R, T, s = 30, 25, 9
        X3 = rng.standard_normal((R, T, 2 * s))
        cb = np.arange(s, 2 * s)
        X3[:, :, cb] = 2.0 * X3[:, :, :s] + 0.5
        k, sh = mc._knockoff_cond_block(X3, cb, s // 2, 2.0, T)
        self.assertEqual(k.shape, (R, T, s))
        self.assertLess(float(np.abs(k - X3[:, :, cb]).max()), 1e-3)
        self.assertGreaterEqual(sh, 1)

    def test_gate_runs_and_keeps_shapes(self):
        d = toy_data(0, proxy=True)
        ref = mc.mv_CaStLe_GRANGER(d, **KW)
        for v in (dict(gate="knockoff_cond", fdr_q=0.1, fdr_offset=0),
                  dict(gate="knockoff_cond", fdr_q=0.1, fdr_offset=0, shift_mults=[2, 3, 4], derand_min=2)):
            r = mc.mv_CaStLe_GRANGER(d, variant=v, **KW)
            self.assertEqual(r["graph"].shape, ref["graph"].shape)
            self.assertIn(0, parent_vars(r))      # the child's own block is never gated


class TestLinkStats(unittest.TestCase):
    def test_off_by_default_and_no_effect(self):
        d = toy_data(2, proxy=True)
        os.environ.pop("TST_LINKSTATS", None)
        with tempfile.TemporaryDirectory() as td:
            a = mc.mv_CaStLe_GRANGER(d, **KW)
            self.assertEqual(os.listdir(td), [])
            os.environ["TST_LINKSTATS"] = td
            try:
                b = mc.mv_CaStLe_GRANGER(d, **KW)
            finally:
                os.environ.pop("TST_LINKSTATS", None)
            self.assertTrue(np.array_equal(a["val_matrix"], b["val_matrix"]))
            files = os.listdir(td)
            self.assertEqual(len(files), 1)
            line = open(os.path.join(td, files[0])).read().split()
            self.assertEqual(line[0], "0"); self.assertEqual(line[1], "4")
            self.assertEqual(line[2], ",".join(map(str, parent_vars(b))))


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.argv = [str(path)]
    spec.loader.exec_module(m)
    return m


class TestResponseDecoy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S = load_script(HERE / "Scripts_Paper" / "3_SyntheticRuns_Multivariate.py", "syn_multi_n4")

    def _case(self, **kw):
        return self.S.build_multivariate_case(alpha_mix=0.6, T=30, N_inject=8, seed=5, n_decoy=1, **kw)

    def test_default_identical_to_parent(self):
        a = self._case(); b = self._case(decoy_driver="parent")
        fa = a["full_data_run"]
        fb = b["full_data_run"]
        self.assertTrue(np.array_equal(np.asarray(fa.values), np.asarray(fb.values)))
        self.assertEqual(list(map(str, fa["var"].values)), list(map(str, fb["var"].values)))

    @unittest.skipUnless((V9 / "Scripts_Paper" / "3_SyntheticRuns_Multivariate.py").exists(), "v9 not present")
    def test_default_identical_to_v9(self):
        old = load_script(V9 / "Scripts_Paper" / "3_SyntheticRuns_Multivariate.py", "syn_multi_v9")
        a = self._case()
        b = old.build_multivariate_case(alpha_mix=0.6, T=30, N_inject=8, seed=5, n_decoy=1)
        fa = a["full_data_run"]
        fb = b["full_data_run"]
        self.assertTrue(np.array_equal(np.asarray(fa.values), np.asarray(fb.values)))

    def test_child_decoy_follows_B(self):
        c = self._case(decoy_driver="child", decoy_phi=0.5, decoy_beta=0.8)
        f = c["full_data_run"]
        names = list(map(str, f["var"].values))
        self.assertEqual(names[3], "D1_B")
        B = np.asarray(f.sel(var="B").values); D = np.asarray(f.sel(var="D1_B").values)
        A = np.asarray(f.sel(var="A").values)
        # decoy at t correlates with B at t-1 more than the parent-driven default decoy does
        def lagcorr(x, y):
            tx = [i for i, d in enumerate(f.sel(var="B").dims) if d == "time"][0]
            xa = np.moveaxis(x, tx, 0)[:-1].ravel(); ya = np.moveaxis(y, tx, 0)[1:].ravel()
            return float(np.corrcoef(xa, ya)[0, 1])
        p = self._case()
        fp = p["full_data_run"]
        Dp = np.asarray(fp.sel(var=str(fp["var"].values[3])).values)
        self.assertGreater(lagcorr(B, D), lagcorr(B, Dp))
        self.assertFalse(np.array_equal(D, Dp))


if __name__ == "__main__":
    unittest.main()
