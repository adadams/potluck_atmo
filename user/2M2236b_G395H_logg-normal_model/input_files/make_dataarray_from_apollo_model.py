from pathlib import Path

import numpy as np
import xarray as xr

current_directory: Path = Path(__file__).parent

apollo_model_directory: Path = (
    current_directory / "reference_inputs" / "G395H" / "G395H_logg-normal"
)

apollo_model_filepath: Path = (
    apollo_model_directory
    / "2M2236.Piette.G395H.cloud-free.2024-02-27.logg-normal-prior.retrieved.Spectrum.binned.dat"
)

wavelengths_lower_bins, wavelengths_upper_bins, fluxes, flux_errors, *_ = np.loadtxt(
    apollo_model_filepath
).T

wavelengths: np.ndarray = 0.5 * (wavelengths_lower_bins + wavelengths_upper_bins)

wavelength_coordinate: xr.DataArray = xr.DataArray(
    data=wavelengths,
    dims=("wavelength",),
    name="wavelength",
    attrs={"units": "micron"},
)

data: xr.Dataset = xr.Dataset(
    data_vars={
        "flux": (("wavelength",), fluxes),
        "flux_error": (("wavelength",), flux_errors),
    },
    coords={"wavelength": wavelength_coordinate},
)

print(f"{current_directory=}")
print(f"{data=}")

data.to_netcdf(current_directory / "2M2236b_G395H_logg-normal_apollo_model.nc")
