"""Night 8: opt-in --decoy-pool (trace_st/decoy.py, 8_HW_PNW21_Runs_v2.py, 6_PinatuboRuns.py).
Defaults must be unchanged: no decoy_pool key, var_names without DECOY, ensemble inputs identical."""
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import xarray as xr

from tests.test_fraction_full_length import load_script

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trace_st import decoy  # noqa: E402


def toy(nvar=3, nt=6, ny=4, nx=5):
    rng = np.random.default_rng(0)
    return xr.DataArray(rng.standard_normal((nvar, ny, nx, nt)).astype("float32"), dims=("var", "lat", "lon", "time"),
                        coords=dict(var=[f"V{i}" for i in range(nvar)], lat=np.arange(ny) * 2.5, lon=np.arange(nx) * 2.5,
                                    time=np.array(["2000-01-01"], dtype="datetime64[D]") + np.arange(nt)))


def write_pool(d, fd, n=3, shift=0.0):
    for m in range(n):
        x = np.random.default_rng(100 + m).standard_normal((len(fd.time), len(fd.lat), len(fd.lon))).astype("float32")
        xr.DataArray(x, dims=("time", "lat", "lon"), coords=dict(time=fd.time, lat=fd.lat + shift, lon=fd.lon), name="DECOY").to_netcdf(
            Path(d) / f"decoy_{m:02d}.nc")


class TestDecoyModule(unittest.TestCase):
    def test_append_member_m(self):
        fd = toy()
        with tempfile.TemporaryDirectory() as d:
            write_pool(d, fd)
            decoy._CACHE.clear()
            for m in range(3):
                out = decoy.with_decoy(fd, d, m)
                self.assertEqual(list(out["var"].values), ["V0", "V1", "V2", "DECOY"])
                self.assertEqual(out.dims, fd.dims)
                np.testing.assert_array_equal(out.sel(var=["V0", "V1", "V2"]).values, fd.values)
                want = np.random.default_rng(100 + m).standard_normal((6, 4, 5)).astype("float32").transpose(1, 2, 0)
                np.testing.assert_array_equal(out.sel(var="DECOY").values, want)

    def test_coordinate_mismatch_raises(self):
        fd = toy()
        with tempfile.TemporaryDirectory() as d:
            write_pool(d, fd, n=1, shift=1.0)
            decoy._CACHE.clear()
            with self.assertRaises(ValueError):
                decoy.with_decoy(fd, d, 0)


class TestRunScripts(unittest.TestCase):
    def test_pnw_default_ensemble_inputs_identical(self):
        m = load_script("8_HW_PNW21_Runs_v2.py")
        fd = toy()
        seen = []

        class Ctx:
            def Pool(self, processes):
                class P:
                    def __enter__(s): return s
                    def __exit__(s, *a): return False
                    def map(s, f, args): seen.extend(args); return [dict(success=True) for _ in args]
                return P()
        patcher = mock.patch.object(m.mp, "get_context", lambda *_: Ctx()); patcher.start(); self.addCleanup(patcher.stop)
        m.run_ensemble_tracks(fd, dict(a=1), None, n_members=3, nproc=1, seed=5)
        self.assertTrue(all(a[0] is fd for a in seen))
        seen.clear()
        with tempfile.TemporaryDirectory() as d:
            write_pool(d, fd)
            decoy._CACHE.clear()
            m.run_ensemble_tracks(fd, dict(a=1), None, n_members=3, nproc=1, seed=5, decoy_pool=d)
        self.assertEqual([list(a[0]["var"].values)[-1] for a in seen], ["DECOY"] * 3)
        self.assertEqual([a[3] for a in seen], [0, 1, 2])
        for i, a in enumerate(seen):   # member i got realization i
            want = np.random.default_rng(100 + i).standard_normal((6, 4, 5)).astype("float32").transpose(1, 2, 0)
            np.testing.assert_array_equal(a[0].sel(var="DECOY").values, want)

    def test_flags_exist_default_none(self):
        for s in ("8_HW_PNW21_Runs_v2.py", "6_PinatuboRuns.py"):
            m = load_script(s)
            sys.argv = ["x", "--outdir", "/tmp/x"]
            self.assertIsNone(m.parse_args().decoy_pool)

    def test_pinatubo_case_default_and_decoy(self):
        os.environ.pop("TST_PINATUBO_DECOY", None)
        os.environ.pop("TST_PINATUBO_EXTRA", None)
        m = load_script("6_PinatuboRuns.py")
        c = m.build_case_pinatubo()
        self.assertNotIn("decoy_pool", c)
        self.assertEqual(list(c["var_names"]), m.PINATUBO_VAR_NAMES)
        os.environ["TST_PINATUBO_DECOY"] = "/some/dir"
        try:
            c2 = m.build_case_pinatubo()
        finally:
            os.environ.pop("TST_PINATUBO_DECOY", None)
        self.assertEqual(c2["decoy_pool"], "/some/dir")
        self.assertEqual(list(c2["var_names"]), m.PINATUBO_VAR_NAMES + ["DECOY"])
        np.testing.assert_array_equal(c2["full_data"].values, c["full_data"].values)


if __name__ == "__main__":
    unittest.main()
