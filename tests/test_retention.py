"""Fix (d): retention rules in trace_st.retention reproduce the paper counts on the original outputs.

Original outputs root: $TST_ORIG_ROOT, else /glade/work/jhayron/trace-st_data/original_submission,
else /glade/derecho/scratch/jhayron/DataCaStLeBTs. Skipped if none is readable.
"""
import os
import unittest
from pathlib import Path

import pandas as pd

from trace_st import retention

CANDIDATES = [os.environ.get("TST_ORIG_ROOT", ""),
              "/glade/work/jhayron/trace-st_data/original_submission",
              "/glade/derecho/scratch/jhayron/DataCaStLeBTs"]
ROOT = next((Path(c) for c in CANDIDATES if c and (Path(c) / "ResultsDebby/hp_search_v2").exists()), None)


@unittest.skipIf(ROOT is None, "original outputs not available")
class TestPaperCounts(unittest.TestCase):
    def test_debby_68(self):
        df = pd.read_csv(ROOT / "ResultsDebby/hp_search_v2/results_jobs_final.csv")
        self.assertEqual(int(retention.debby_retained(df).sum()), 68)

    def test_pinatubo_999(self):
        df = pd.read_csv(ROOT / "ResultsPinatubo/hp_search_v10/results_jobs_final.csv")
        self.assertEqual(int(retention.pinatubo_retained(df).sum()), 999)

    def test_pnw_242(self):
        df = pd.read_csv(ROOT / "Results_HW_PNW21_v3/hyperparam_search_summary.csv")
        self.assertEqual(int(retention.pnw_retained(df).sum()), 242)

    def test_t1_993_935_597(self):
        df = pd.read_csv(ROOT / "ResultsSynthetic/univariate_v0/results_tracks_final.csv",
                         usecols=["method", "trial_id", "track_idx", "raw_len"])
        kept = retention.t1_retained(df).groupby(level="method").sum().to_dict()
        self.assertEqual({k: int(v) for k, v in kept.items()},
                         {"granger": 993, "dynotears": 935, "pcmci": 597})


if __name__ == "__main__":
    unittest.main()
