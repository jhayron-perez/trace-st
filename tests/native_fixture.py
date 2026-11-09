"""Build/compare the toy run_track outputs used to check strength_mode='native' == pre-refit code.

python tests/native_fixture.py OUT.npz   # writes the fixture with whatever trace_st is importable
"""
import sys

import numpy as np

from trace_st import trajectory
from tests.test_prob_rule import BASE_PARAMS, _toy_data

ENGINES = {
    "granger": dict(lambda_a=0.01, l1_ratio=0.5, dependence_threshold=1e-7),
    "pcmci": dict(pc_alpha=0.05, graph_p_threshold=0.05),
    "dynotears": dict(lambda_a=0.05, lambda_w=0.05, max_iter=50, dependence_threshold=0.01),
}


def toy_outputs(extra=None):
    res = {}
    for eng, kw in ENGINES.items():
        p = dict(BASE_PARAMS, n_steps_time=4, cd_method=eng, cd_kwargs=kw, seed=5)
        p.update(extra or {})
        out = trajectory.run_track(_toy_data(seed=3), p, date_end="2000-02-09",
                                   return_debug=False, return_cluster_summaries=False)
        res[f"{eng}_centers"] = np.asarray(out["centers"], float)
        res[f"{eng}_parents"] = np.asarray(out["parents"], int)
        res[f"{eng}_winner_vals"] = np.asarray([float(w["causal_value"]) for w in out["winners"]], float)
    return res


if __name__ == "__main__":
    np.savez(sys.argv[1], **toy_outputs())
