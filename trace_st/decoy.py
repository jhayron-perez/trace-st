"""Decoy pool (revision 1, night 8; opt-in): one extra candidate variable "DECOY" per ensemble member.

A pool directory holds decoy_00.nc ... decoy_NN.nc, each a DataArray (any name) on the dimensions of the case's
analysis array (all dims of ``full_data`` except "var", same coordinates). Member m of every configuration appends
realization m as the last variable (``with_decoy``); the pool is therefore the same for every configuration.
Not imported by ``trace_st/__init__``: nothing changes unless a run script is started with --decoy-pool.
"""
from pathlib import Path

import numpy as np
import xarray as xr

DECOY_NAME = "DECOY"
_CACHE = {}


def pool_file(pool_dir, member):
    return Path(pool_dir) / f"decoy_{int(member):02d}.nc"


def _load(pool_dir, member):
    key = (str(pool_dir), int(member))
    if key not in _CACHE:
        with xr.open_dataset(pool_file(pool_dir, member)) as ds:
            name = list(ds.data_vars)[0]
            _CACHE[key] = ds[name].load()
    return _CACHE[key]


def with_decoy(full_data, pool_dir, member, name=DECOY_NAME):
    """full_data (var, ...) with realization `member` of the pool appended as the last variable `name`."""
    dims = [d for d in full_data.dims if d != "var"]
    d = _load(pool_dir, member).transpose(*dims)
    for k in dims:
        a, b = np.asarray(full_data[k].values), np.asarray(d[k].values)
        ok = a.shape == b.shape and (np.array_equal(a, b) if a.dtype.kind in "Mm" else np.allclose(a, b, atol=1e-6))
        if not ok:
            raise ValueError(f"decoy pool {pool_dir} member {member}: coordinate {k} differs from the analysis array")
    d = d.assign_coords({k: full_data[k] for k in dims}).astype(full_data.dtype).expand_dims(var=[name])
    return xr.concat([full_data, d.reset_coords(drop=True)], dim="var").transpose(*full_data.dims)
