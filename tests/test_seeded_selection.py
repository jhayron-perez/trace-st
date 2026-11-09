"""Fix (b): Monte Carlo cluster selection is reproducible given a seed.

Run: python -m unittest discover -s tests -v   (castle env)
"""
import unittest

import numpy as np

from trace_st import trajectory
from tests.test_prob_rule import BASE_PARAMS, _toy_data

PARAMS = dict(BASE_PARAMS, n_steps_time=6, prob_rule="linear")


def _run(params, **kw):
    out = trajectory.run_track(_toy_data(seed=3), params, date_end="2000-02-09",
                               return_debug=False, return_cluster_summaries=False, **kw)
    return np.asarray(out["centers"]), np.asarray(out["parents"])


class TestSeeded(unittest.TestCase):
    def test_same_seed_bitwise_identical(self):
        c1, p1 = _run(dict(PARAMS, seed=123))
        c2, p2 = _run(dict(PARAMS, seed=123))
        self.assertEqual(c1.tobytes(), c2.tobytes())
        self.assertEqual(p1.tobytes(), p2.tobytes())

    def test_rng_argument_equivalent_to_seed(self):
        c1, p1 = _run(dict(PARAMS, seed=7))
        c2, p2 = _run(dict(PARAMS), rng=np.random.default_rng(7))
        self.assertEqual(c1.tobytes(), c2.tobytes())
        self.assertEqual(p1.tobytes(), p2.tobytes())

    def test_different_seeds_differ(self):
        runs = {s: _run(dict(PARAMS, seed=s)) for s in range(20)}
        keys = {(c.tobytes(), p.tobytes()) for c, p in runs.values()}
        self.assertGreater(len(keys), 1, "20 different seeds gave identical tracks")


if __name__ == "__main__":
    unittest.main()
