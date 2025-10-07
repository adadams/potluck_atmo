from pathlib import Path

import numpy as np
import xarray as xr

current_directory: Path = Path(__file__).parent

risotto_input_filepath: Path = current_directory / "T3B_risotto_Full.txt"

wavelengths, transit_depths, transit_depth_uncertainties = np.loadtxt(
    risotto_input_filepath
).T

wavelength_coordinate: xr.Variable = xr.Variable(
    dims=("wavelength",), data=wavelengths, attrs={"units": "micron"}
)

transit_depth_dataarray: xr.DataArray = xr.DataArray(
    data=transit_depths,
    dims=("wavelength",),
    coords={"wavelength": wavelength_coordinate},
    attrs={"units": "dimensionless"},
    name="transit_depth",
)

transit_depth_uncertainty_dataarray: xr.DataArray = xr.DataArray(
    data=transit_depth_uncertainties,
    dims=("wavelength",),
    coords={"wavelength": wavelength_coordinate},
    attrs={"units": "dimensionless"},
    name="transit_depth_uncertainty",
)

data_dataset: xr.Dataset = xr.Dataset(
    data_vars={
        "transit_depth": transit_depth_dataarray,
        "transit_depth_uncertainty": transit_depth_uncertainty_dataarray,
    },
    coords={"wavelength": wavelength_coordinate},
).sel(wavelength=slice(0.61, None))

data_dataset.to_netcdf(current_directory / "T3B_risotto_Full_data.nc")
