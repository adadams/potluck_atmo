from importlib import import_module
from pathlib import Path
from types import ModuleType

import xarray as xr
from matplotlib import pyplot as plt

from calculate_RT import calculate_observed_transmission_spectrum
from model_statistics.calculate_statistics import (
    calculate_log_likelihood,
    calculate_reduced_chi_squared_statistic,
)
from spectrum.bin import resample_spectral_quantity_to_new_wavelengths
from user.input_importers import import_model_id
from user.input_structs import UserForwardModelInputs

model_directory_label: str = "R1b_retrieval"

project_directory: Path = Path.cwd()
user_directory: Path = project_directory / "user"
model_directory: Path = user_directory / f"{model_directory_label}_model"
input_file_directory: Path = model_directory / "input_files"
intermediate_output_directory: Path = model_directory / "intermediate_outputs"
output_file_directory: Path = model_directory / "output_files"

plt.style.use(project_directory / "arthur.mplstyle")

parent_directory: str = "user"

forward_model_module: ModuleType = import_module(
    f"{parent_directory}.{model_directory_label}_model.input_files.{model_directory_label}_forward_model_inputs"
)

default_forward_model_inputs: UserForwardModelInputs = (
    forward_model_module.default_forward_model_inputs
)

precurated_crosssection_catalog: xr.Dataset = (
    forward_model_module.precurated_crosssection_catalog_dataset
)

model_id: str = import_model_id(
    model_directory_label=model_directory_label, parent_directory="user"
)

data: xr.Dataset = forward_model_module.data_dataset
reference_model_wavelengths: xr.DataArray = data.wavelength

fraction_of_reddest_fwhm_to_convolve_with: float = 0.01  # 0.01


def calculate_transit_model(
    forward_model_inputs: UserForwardModelInputs,
    resampling_fwhm_fraction: float = fraction_of_reddest_fwhm_to_convolve_with,
) -> float:
    transit_depths = calculate_observed_transmission_spectrum(
        user_forward_model_inputs=forward_model_inputs,
        precalculated_crosssection_catalog=precurated_crosssection_catalog,
    )

    reference_model_wavelengths: xr.DataArray = forward_model_inputs.output_wavelengths

    transit_depths_sampled_to_data: xr.DataArray = (
        resample_spectral_quantity_to_new_wavelengths(
            reference_model_wavelengths,
            transit_depths.wavelength,
            transit_depths,
            fwhm=resampling_fwhm_fraction
            * (
                reference_model_wavelengths.to_numpy()[-1]
                - reference_model_wavelengths.to_numpy()[-2]
            ),
        )
    ).rename("resampled_transmission_flux")

    return transit_depths_sampled_to_data


def calculate_transit_model_log_likelihood(
    forward_model_inputs: UserForwardModelInputs,
    data: xr.DataArray,
    data_error: xr.DataArray,
    resampling_fwhm_fraction: float = fraction_of_reddest_fwhm_to_convolve_with,
) -> float:
    transit_depths_sampled_to_data: xr.DataArray = calculate_transit_model(
        forward_model_inputs=forward_model_inputs,
        resampling_fwhm_fraction=resampling_fwhm_fraction,
    )

    log_likelihood: float = calculate_log_likelihood(
        transit_depths_sampled_to_data, data, data_error
    )

    return log_likelihood


def calculate_transit_model_reduced_chi_squared_statistic(
    forward_model_inputs: UserForwardModelInputs,
    data: xr.DataArray,
    data_error: xr.DataArray,
    number_of_free_parameters: int,
    resampling_fwhm_fraction: float = fraction_of_reddest_fwhm_to_convolve_with,
) -> float:
    transit_depths_sampled_to_data: xr.DataArray = calculate_transit_model(
        forward_model_inputs=forward_model_inputs,
        resampling_fwhm_fraction=resampling_fwhm_fraction,
    )

    reduced_chi_squared_statistic: float = calculate_reduced_chi_squared_statistic(
        data, data_error, transit_depths_sampled_to_data, number_of_free_parameters
    )

    return reduced_chi_squared_statistic


if __name__ == "__main__":
    test_mixing_ratios: dict[str, float] = [
        {"h2o": -15.171, "ch4": -15.318},
        {"h2o": -14.171, "ch4": -14.318},
        {"h2o": -13.171, "ch4": -13.318},
        {"h2o": -12.171, "ch4": -12.318},
        {"h2o": -11.171, "ch4": -11.318},
        {"h2o": -10.171, "ch4": -10.318},
        {"h2o": -9.171, "ch4": -9.318},
        {"h2o": -8.171, "ch4": -8.318},
        {"h2o": -7.171, "ch4": -7.318},
        {"h2o": -6.171, "ch4": -6.318},
        {"h2o": -5.171, "ch4": -5.318},
        {"h2o": -4.171, "ch4": -4.318},
        {"h2o": -3.171, "ch4": -3.318},
        {"h2o": -2.171, "ch4": -2.318},
        {"h2o": -1.171, "ch4": -1.318},
    ]

    for test_mixing_ratio in test_mixing_ratios:
        print(f"{test_mixing_ratio=}")
        test_forward_model_inputs: UserForwardModelInputs = (
            forward_model_module.build_uniform_mixing_ratio_forward_model(
                uniform_log_abundances=test_mixing_ratio
            )
        )

        transit_depths_sampled_to_data: xr.DataArray = calculate_transit_model(
            forward_model_inputs=test_forward_model_inputs,
            resampling_fwhm_fraction=fraction_of_reddest_fwhm_to_convolve_with,
        )

        log_likelihood: float = calculate_log_likelihood(
            transit_depths_sampled_to_data, data.data, data.data_error
        ).item()
        print(f"{log_likelihood=}")

        reduced_chi_squared_statistic: float = calculate_reduced_chi_squared_statistic(
            data.data,
            data.data_error,
            transit_depths_sampled_to_data,
            30,
        ).item()
        print(f"{reduced_chi_squared_statistic=}")
