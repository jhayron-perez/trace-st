"""Night-3 N2 opt-in nulls: PNW placebo time/longitude shift and replace mode; Debby input shifts.

Defaults must reproduce the original behaviour; shifts must move the data by exactly K days / D degrees.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tests.test_fraction_full_length import load_script

ROOT = Path(__file__).resolve().parent.parent


def fake_pnw_loader(path_files, varstems, list_names_vars, time_slice, join="inner"):
    """Deterministic fake anomaly data: value = day-of-year + 1000 * var index + lon/1000, 4x global lon."""
    t = pd.date_range(time_slice[0], time_slice[1], freq="1D")
    lat = np.array([40.0, 50.0]); lon = np.arange(0.0, 360.0, 90.0)
    arrs = []
    for i, _ in enumerate(varstems):
        v = (t.dayofyear.values[None, None, :] + 1000.0 * i + lon[None, :, None] / 1000.0
             + 0.0 * lat[:, None, None])
        arrs.append(v)
    return xr.DataArray(np.stack(arrs), dims=("var", "lat", "lon", "time"),
                        coords=dict(var=list_names_vars, lat=lat, lon=lon, time=t))


class TestPNWPlacebo(unittest.TestCase):
    def setUp(self):
        self.pnw = load_script("8_HW_PNW21_Runs_v2.py")
        self._orig = self.pnw.tst.data_io.load_full_data
        self.pnw.tst.data_io.load_full_data = fake_pnw_loader

    def tearDown(self):
        self.pnw.tst.data_io.load_full_data = self._orig

    def test_default_names_and_dates(self):
        da = self.pnw.load_placebo_vars(["olr"], 2011)
        self.assertEqual(list(da["var"].values), ["MTNLWRF_P2011"])
        self.assertEqual(pd.Timestamp(da.time.values[0]), pd.Timestamp("2021-05-20"))
        self.assertEqual(float(da.isel(var=0, lat=0, lon=0, time=0)), pd.Timestamp("2011-05-20").dayofyear)

    def test_shift_days(self):
        da = self.pnw.load_placebo_vars(["olr"], 2021, shift_days=15)
        self.assertEqual(list(da["var"].values), ["MTNLWRF_P2021S+15"])
        self.assertEqual(pd.Timestamp(da.time.values[0]), pd.Timestamp("2021-05-20"))
        self.assertEqual(float(da.isel(var=0, lat=0, lon=0, time=0)), pd.Timestamp("2021-06-04").dayofyear)
        da = self.pnw.load_placebo_vars(["olr"], 2021, shift_days=-30)
        self.assertEqual(float(da.isel(var=0, lat=0, lon=0, time=0)), pd.Timestamp("2021-04-20").dayofyear)
        self.assertEqual(da.sizes["time"], 42)

    def test_shift_lon(self):
        da = self.pnw.load_placebo_vars(["olr"], 2021, shift_lon=90.0)
        self.assertEqual(list(da["var"].values), ["MTNLWRF_P2021L+90"])
        # field moved east by 90 deg: value at lon 90 is the original value at lon 0
        self.assertAlmostEqual(float(da.sel(lon=90.0).isel(var=0, lat=0, time=0)) % 1, 0.0, places=6)
        self.assertAlmostEqual(float(da.sel(lon=0.0).isel(var=0, lat=0, time=0)) % 1, 0.270, places=6)

    def test_replace_keeps_order_and_p(self):
        self.pnw.regrid_to_2p5 = lambda x: x
        case_add = self.pnw.build_case_pnw(placebo_vars=["olr"], placebo_year=2011)
        case_rep = self.pnw.build_case_pnw(placebo_vars=["olr"], placebo_year=2011, placebo_mode="replace")
        va = [str(v) for v in case_add["full_data"]["var"].values]
        vr = [str(v) for v in case_rep["full_data"]["var"].values]
        self.assertEqual(len(va), 7)
        self.assertEqual(va[-1], "MTNLWRF_P2011")
        self.assertEqual(len(vr), 6)
        self.assertEqual(vr[2], "MTNLWRF_P2011")
        self.assertEqual(vr[0], "Z500")
        with self.assertRaises(ValueError):
            self.pnw.build_case_pnw(placebo_vars=["z500"], placebo_year=2011, placebo_mode="replace")


class TestDebbyShift(unittest.TestCase):
    def setUp(self):
        p = ROOT / "data_processing" / "era5_debby" / "make_debby_extra_inputs.py"
        spec = importlib.util.spec_from_file_location("make_debby_extra_inputs", p)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        t = pd.date_range("2006-07-01", "2006-09-30 23:00", freq="1h")
        lon = np.arange(-50.0, 40.001, 0.25)
        v = (np.arange(t.size, dtype=float)[:, None, None] + 0.0 * lon[None, None, :]
             + np.zeros((1, 3, 1)))
        v = v + (lon[None, None, :] == -50.0) * 0.5
        self.an = xr.DataArray(v, dims=("time", "latitude", "longitude"),
                               coords=dict(time=t, latitude=[10.0, 9.75, 9.5], longitude=lon))

    def test_default_window_identical(self):
        old = self.an.sel(time=slice("2006-08-11 00:00", "2006-08-28 23:00"))
        new = self.m.select_window(self.an, 2006, 0)
        self.assertTrue(new.identical(old))
        self.assertEqual(new.sizes["time"], 432)
        self.assertTrue(self.m.roll_lon(new, 0.0).identical(new))

    def test_shift_days(self):
        old = self.an.sel(time=slice("2006-08-11 00:00", "2006-08-28 23:00"))
        new = self.m.select_window(self.an, 2006, 10)
        self.assertEqual(new.sizes["time"], 432)
        np.testing.assert_array_equal(new.time.values, old.time.values)
        np.testing.assert_array_equal(new.values, old.values + 240.0)
        neg = self.m.select_window(self.an, 2006, -5)
        np.testing.assert_array_equal(neg.values, old.values - 120.0)

    def test_shift_lon(self):
        r = self.m.roll_lon(self.an, 10.0)
        # the marked west-edge column (-50) moved 10 deg east (40 cells)
        self.assertEqual(float(r.sel(longitude=-40.0).isel(time=0, latitude=0)), 0.5)
        self.assertEqual(float(r.sel(longitude=-50.0).isel(time=0, latitude=0)), 0.0)
        np.testing.assert_array_equal(r.longitude.values, self.an.longitude.values)


if __name__ == "__main__":
    unittest.main()
