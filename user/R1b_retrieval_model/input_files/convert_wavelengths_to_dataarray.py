from pathlib import Path

import numpy as np
import xarray as xr

current_directory: Path = Path(__file__).parent
model_directory: Path = current_directory.parent

data_filepath: Path = model_directory / "T3B_APOLLO_test-truncated.dat"
data_wavelo, data_wavehi, data_flux, data_errors, *_ = np.loadtxt(data_filepath).T
data_wavelengths = 0.5 * (data_wavelo + data_wavehi)

data: xr.DataArray = xr.DataArray(
    data=data_flux,
    dims=("wavelength",),
    coords={"wavelength": data_wavelengths},
    name="data",
)

data_error: xr.DataArray = xr.DataArray(
    data=data_errors,
    dims=("wavelength",),
    coords={"wavelength": data_wavelengths},
    name="data_error",
)

data_dataset: xr.Dataset = xr.Dataset(
    data_vars={
        "data": data,
        "data_error": data_error,
    }
)

data_dataset.to_netcdf(current_directory / "T3B_APOLLO_truncated.nc")
