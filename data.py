from pathlib import Path

import numpy as np
import xarray as xr

from wavelength import get_wavelengths_from_wavelength_bins


def convert_APOLLO_data_to_dataset(filepath: Path) -> xr.Dataset:
    wavelength_bin_starts, wavelength_bin_ends, flux_lambda, flux_lambda_error, *_ = (
        np.loadtxt(filepath).T
    )

    wavelengths = get_wavelengths_from_wavelength_bins(
        wavelength_bin_starts, wavelength_bin_ends
    )

    return xr.Dataset(
        data_vars={
            "wavelength_bin_starts": (("wavelength",), wavelength_bin_starts),
            "wavelength_bin_ends": (("wavelength",), wavelength_bin_ends),
            "flux_lambda": (("wavelength",), flux_lambda),
            "flux_lambda_error": (("wavelength",), flux_lambda_error),
        },
        coords={"wavelength": wavelengths},
    )


if __name__ == "__main__":
    test_data_directory: Path = Path(__file__).parent / "test_inputs"
    test_data_filepath: Path = (
        test_data_directory / "2M2236b_NIRSpec_G395H_R500_APOLLO.dat"
    )

    test_data_dataset: xr.Dataset = convert_APOLLO_data_to_dataset(test_data_filepath)
    test_data_dataset.to_netcdf(test_data_directory / f"{test_data_filepath.stem}.nc")
