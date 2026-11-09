"""Fix (c): fraction_full_length in 8_HW_PNW21_Runs_v2 counts tracks with n_steps_time + 1 centers."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"),
                                                  ROOT / "Scripts_Paper" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFractionFullLength(unittest.TestCase):
    def test_toy_list(self):
        pnw = load_script("8_HW_PNW21_Runs_v2.py")
        n = 30
        lens = [n + 1, n + 1, n + 1, n, 10]  # 3 of 5 full-length (start point + 30 steps)
        res = []
        for L in lens:
            c = np.column_stack([np.full(L, 50.0), np.linspace(230, 240, L)])
            res.append(dict(success=True, centers=c, parents=np.ones(L, int), out=None, error=None))
        data = xr.DataArray(np.zeros((1, 5, 5)), dims=("var", "lat", "lon"),
                            coords=dict(var=["z500"], lat=np.linspace(40, 60, 5), lon=np.linspace(220, 250, 5)))
        s = pnw.summarize_ensemble_results(res, params=dict(n_steps_time=n, box_size=10, run_id=0),
                                           full_data_run=data, event_lat=50.0, event_lon=240.0)
        self.assertAlmostEqual(s["fraction_full_length"], 3 / 5)


if __name__ == "__main__":
    unittest.main()
