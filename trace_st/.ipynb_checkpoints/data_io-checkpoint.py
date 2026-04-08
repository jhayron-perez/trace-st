# NOTE: Auto-organized from original CausalBTs.py and mcastle_utils_vBT.py
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path


def extract_box(data, bounds):
    """
    Extract a lat-lon box from global data, ensuring that longitude is ordered
    continuously eastward and wraparound is handled correctly.

    Accepts either:
      - lat/lon
      - latitude/longitude
    """
    lat_min, lat_max, lon_min, lon_max = bounds

    # ------------------------------------------------------------
    # HARDENING: accept latitude/longitude naming too
    # (does NOTHING if already lat/lon)
    # ------------------------------------------------------------
    rename_map = {}
    if "lat" not in data.coords and "latitude" in data.coords:
        rename_map["latitude"] = "lat"
    if "lon" not in data.coords and "longitude" in data.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        data = data.rename(rename_map)

    # Optional sanity: fail early with a clear message
    if "lat" not in data.coords or "lon" not in data.coords:
        raise ValueError(f"extract_box expected lat/lon coords. Got coords={list(data.coords)} dims={data.dims}")

    # Normalize requested longitudes to [0, 360)
    lon_min = lon_min % 360
    lon_max = lon_max % 360

    # Convert lons to 0–360 if necessary
    if data.lon.min() < 0:
        data = data.assign_coords(lon=(data.lon % 360))

    data = data.sortby("lat", ascending=True)

    # Latitude selection
    data_sel = data.sel(lat=slice(lat_min, lat_max))

    # Longitude selection with wraparound
    if lon_min <= lon_max:
        region = data_sel.sel(lon=slice(lon_min, lon_max))
    else:
        part1 = data_sel.sel(lon=slice(lon_min, float(data_sel.lon.max())))
        part2 = data_sel.sel(lon=slice(0, lon_max))
        region = xr.concat([part1, part2], dim="lon")

    return region

def get_data_mcastle(center,box_size,full_data_list):
    latmintemp = center[0]-box_size/2
    latmaxtemp = center[0]+box_size/2
    lonmintemp = center[1]-box_size/2
    lonmaxtemp = center[1]+box_size/2

    
    list_data = []
    for data_temp in full_data_list:
    # for data_temp in [data4castle_z, data4castle_sst]:
        data_box = extract_box(data_temp, [latmintemp, latmaxtemp, lonmintemp, lonmaxtemp])
        
        rename_map = {}
        if "lat" not in data_box.dims and "latitude" in data_box.dims:
            rename_map["latitude"] = "lat"
        if "lon" not in data_box.dims and "longitude" in data_box.dims:
            rename_map["longitude"] = "lon"
        if rename_map:
            data_box = data_box.rename(rename_map)
            
        data_box = data_box.transpose('lat','lon','time')#.values
        list_data.append(data_box)
    
    full_data_castle = xr.concat(list_data, dim='var').astype('float32').values
    full_data_castle[np.isnan(full_data_castle)] = np.float32(0.0)
    return full_data_castle

def _to_0360_lon(da: xr.DataArray, lon_name: str = "lon") -> xr.DataArray:
    if lon_name not in da.coords:
        return da
    lon = da[lon_name].values
    if np.nanmin(lon) < 0:
        da = da.assign_coords({lon_name: (da[lon_name] % 360)})
        da = da.sortby(lon_name)
    return da

def load_one_variable(path_files: str, varstem: str, time_slice=None) -> xr.DataArray:
    fpath = Path(path_files) / f"{varstem}_daily_anoms.nc"
    if not fpath.exists():
        raise FileNotFoundError(f"Missing file: {fpath}")

    ds = xr.open_dataset(fpath)

    # grab the only variable (robust)
    vname = list(ds.data_vars)[0]
    da = ds[vname]

    # rename coords to match CaStLe conventions
    rename_map = {}
    if "latitude" in da.coords:    rename_map["latitude"]   = "lat"
    if "longitude" in da.coords:   rename_map["longitude"]  = "lon"
    if "valid_time" in da.coords:  rename_map["valid_time"] = "time"
    if rename_map:
        da = da.rename(rename_map)

    # lon convention + lat ascending
    da = _to_0360_lon(da, "lon")
    if "lat" in da.coords:
        da = da.sortby("lat", ascending=True)

    # ensure dim order (handle datasets that might not have all dims)
    want = [d for d in ["lat", "lon", "time"] if d in da.dims]
    da = da.transpose(*want)

    # set a clean name
    da = da.rename(varstem)

    # enforce float32 early (saves memory and keeps mcastle inputs consistent)
    da = da.astype("float32")

    # optional time restriction
    if time_slice is not None and "time" in da.coords:
        da = da.sel(time=slice(time_slice[0], time_slice[1]))

    # enforce float32 for downstream Tigramite / CaStLe

    da = da.astype('float32', copy=False)


    # ensure time is sorted increasing

    if 'time' in da.coords:

        da = da.sortby('time')


    return da


def load_full_data(
    path_files: str,
    varstems: list,
    list_names_vars: list,
    time_slice=None,
    join: str = "inner",
) -> xr.DataArray:
    if len(varstems) != len(list_names_vars):
        raise ValueError("varstems and list_names_vars must have the same length/order.")

    data_arrays = []
    for vs in varstems:
        da = load_one_variable(path_files, vs, time_slice=time_slice)

        # ------------------------------------------------------------
        # HARDENING: remove stray singleton dims / coords like pressure_level
        # Keep ONLY lat/lon/time as dims (plus allow size-1 dims to squeeze out)
        # ------------------------------------------------------------
        # 1) squeeze any singleton dims (pressure_level=1, expver=1, etc.)
        da = da.squeeze(drop=True)

        # 2) drop coords that are not core coordinates
        keep_coords = {"lat", "lon", "time"}
        drop_coords = [c for c in da.coords if c not in keep_coords]
        if drop_coords:
            da = da.drop_vars(drop_coords, errors="ignore")

        # 3) enforce exact dim set/order (this will error loudly if a var is weird)
        expected_dims = ("lat", "lon", "time")
        extra_dims = [d for d in da.dims if d not in expected_dims]
        if extra_dims:
            # if any extra dims remain, they are NOT singleton (since we squeezed),
            # so better to fail with a clear message.
            raise ValueError(
                f"{vs}: unexpected non-singleton dims {extra_dims}. "
                f"Current dims: {da.dims}. You likely still have a level dimension."
            )

        da = da.transpose(*expected_dims)

        data_arrays.append(da)

    # align all variables on common coords
    data_arrays = list(xr.align(*data_arrays, join=join))

    # concat into a single DataArray with dim "var"
    full_data = xr.concat(data_arrays, dim=xr.IndexVariable("var", list_names_vars))

    return full_data