"""Night-2 A3: opt-in extra E3SM fields in 6_PinatuboRuns.py. Default case must equal the pre-change one
(hash computed with commit f311c6b's script, develop job 8147240); paper variables unchanged with extras."""
import hashlib
import os
import unittest

from tests.test_fraction_full_length import load_script

PRE = "35f63b389c694536da4dad72"


def h(da):
    return hashlib.sha256(da.values.tobytes()).hexdigest()[:24]


class TestPinatuboExtras(unittest.TestCase):
    def test_default_and_extras(self):
        os.environ.pop("TST_PINATUBO_EXTRA", None)
        m = load_script("6_PinatuboRuns.py")
        c = m.build_case_pinatubo()
        self.assertEqual(list(c["var_names"]), m.PINATUBO_VAR_NAMES)
        self.assertEqual(h(c["full_data"]), PRE)
        os.environ["TST_PINATUBO_EXTRA"] = "TMSO203,BURDENSO403,FLUT"
        try:
            c2 = m.build_case_pinatubo()
        finally:
            os.environ.pop("TST_PINATUBO_EXTRA", None)
        self.assertEqual(list(c2["var_names"]), m.PINATUBO_VAR_NAMES + ["TMSO203", "BURDENSO403", "FLUT"])
        self.assertEqual(h(c2["full_data"].sel(var=m.PINATUBO_VAR_NAMES)), PRE)


if __name__ == "__main__":
    unittest.main()
