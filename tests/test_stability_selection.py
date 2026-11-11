"""Night-2 D4: opt-in Elastic-Net stability selection in castle_core.mv_CaStLe_GRANGER.
Default (stability=None) must equal the pre-change engine (hashes from commit e744c79's castle_core)."""
import hashlib
import unittest

import numpy as np

from trace_st import castle_core as cc

PRE = {"default": "24a6007136fb46c8", "child2": "409456bdffaa7f05"}


def data():
    return np.random.default_rng(0).normal(size=(3, 12, 12, 8)).cumsum(-1)


class TestStability(unittest.TestCase):
    KW = {"default": dict(), "child2": dict(child_variable=2, lambda_a=0.005, l1_ratio=0.5)}

    def test_default_identical(self):
        for k, kw in self.KW.items():
            r = cc.mv_CaStLe_GRANGER(data(), radius=1, **kw)
            self.assertEqual(hashlib.sha256(r["val_matrix"].tobytes()).hexdigest()[:16], PRE[k], k)

    def test_mechanism(self):
        d = data()
        kw = self.KW["child2"]
        full = cc.mv_CaStLe_GRANGER(d, radius=1, **kw)
        st = cc.mv_CaStLe_GRANGER(d, radius=1, stability=dict(n_sub=20, frac=0.5, thr=0.6), **kw)
        sel_f, sel_s = full["graph"] == "-->", st["graph"] == "-->"
        self.assertTrue(np.all(sel_f[sel_s]))                       # stable subset of full support
        np.testing.assert_array_equal(st["val_matrix"][sel_s], full["val_matrix"][sel_s])  # kept coefs = full fit
        none = cc.mv_CaStLe_GRANGER(d, radius=1, stability=dict(n_sub=5, frac=0.5, thr=1.01), **kw)
        self.assertEqual(int((none["graph"] == "-->").sum()), 0)
        allk = cc.mv_CaStLe_GRANGER(d, radius=1, stability=dict(n_sub=5, frac=0.5, thr=0.0), **kw)
        np.testing.assert_array_equal(allk["val_matrix"], full["val_matrix"])


if __name__ == "__main__":
    unittest.main()
