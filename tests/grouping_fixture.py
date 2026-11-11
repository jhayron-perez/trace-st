"""Build/compare the default-path outputs used to check grouping='dbscan' (and the T1 record) == pre-change code.

python -m tests.grouping_fixture OUT.npz   # writes the fixture with whatever trace_st is importable
(made at adb4bb9, before the grouping / --n-steps changes: tests/data/grouping_default_adb4bb9.npz)
"""
import hashlib
import sys

import numpy as np
import pandas as pd

from trace_st import movement
from tests.native_fixture import toy_outputs


def random_candidates(seed, n=40, radius=2, n_vars=2):
    """Random candidate tables like run_track builds (vars, causal_value, position)."""
    rng = np.random.default_rng(seed)
    s = (2 * radius + 1) ** 2
    rows = []
    for _ in range(n):
        p = int(rng.integers(1, n_vars + 1))
        pos = int(rng.integers(0, s)) + (p - 1) * s
        rows.append((f"p{p}_c1", float(rng.normal(0, 1)), (pos, 0, 1)))
    df = pd.DataFrame(rows, columns=["vars", "causal_value", "position"])
    df["causal_value_abs"] = np.abs(df["causal_value"].to_numpy(dtype=float))
    return df


PICK_CASES = [
    dict(prefer_sign="both", montecarlo=False, eps=0.15, min_samples=2, score_mode="mean", alpha=3.0),
    dict(prefer_sign="both", montecarlo=True, eps=0.3, min_samples=1, score_mode="sum", alpha=1.0),
    dict(prefer_sign="auto", montecarlo=True, eps=0.5, min_samples=3, score_mode="sum_over_n", alpha=2.0),
]


def pick_outputs(extra=None):
    res = {}
    for ci, case in enumerate(PICK_CASES):
        out = []
        for seed in range(8):
            df = random_candidates(seed)
            kw = dict(case, radius=2, spaceres=1.0, rng=np.random.default_rng(100 + seed), prob_rule="linear")
            kw.update(extra or {})
            dlat, dlon, sign, cl, _ = movement.pick_direction_then_weighted_delta(df, **kw)
            out.append([dlat, dlon, {"pos": 1.0, "neg": -1.0, None: 0.0}[sign], -1.0 if cl is None else float(cl)])
        res[f"pick_case{ci}"] = np.asarray(out, float)
    return res


def t1_case_hashes():
    from tests.test_fraction_full_length import load_script
    t1 = load_script("1_SyntheticRuns_Univariate.py")
    fd, specs, date_end = t1.build_synthetic_case()
    h = hashlib.sha256(fd.values.tobytes())
    h.update(np.asarray(fd["time"].values).astype("int64").tobytes())
    for sp in specs:
        h.update(np.asarray(sp["centers"], float).tobytes())
    h.update(date_end.encode())
    return {"t1_case_sha256": np.array(h.hexdigest())}


def all_outputs(extra=None):
    res = {}
    res.update({f"track_{k}": v for k, v in toy_outputs(extra).items()})
    res.update(pick_outputs(extra))
    return res


if __name__ == "__main__":
    np.savez(sys.argv[1], **all_outputs(), **t1_case_hashes())
