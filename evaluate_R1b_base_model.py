from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TypedDict

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from nautilus import Prior, Sampler

from calculate_RT import calculate_observed_transmission_spectrum
from model_statistics.calculate_statistics import calculate_log_likelihood
from model_statistics.error_inflation import inflate_errors_by_flux_scaling
from spectrum.bin import resample_spectral_quantity_to_new_wavelengths
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
    forward_model_module.precurated_crosssection_catalog
)

data: xr.Dataset = forward_model_module.data_dataset
reference_model_wavelengths: xr.DataArray = data.wavelength

fraction_of_reddest_fwhm_to_convolve_with: float = 0.01


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


class ModelFreeParameters(TypedDict):
    planet_radius_relative_to_earth: float
    uniform_ch4_log_abundance: float
    uniform_h2o_log_abundance: float
    flux_scaled_error_inflation_factor: float
    log10_constant_error_inflation_term: float


def prepare_model_for_likelihood_evaluation(
    free_parameters: ModelFreeParameters,
    resampling_fwhm_fraction: float = fraction_of_reddest_fwhm_to_convolve_with,
    data_dataset: xr.Dataset = data,
) -> xr.Dataset:
    # unpack dict of free parameters
    planet_radius_relative_to_earth: float = free_parameters[
        "planet_radius_relative_to_earth"
    ]
    uniform_log_abundances: dict[str, float] = {
        "h2o": free_parameters["uniform_h2o_log_abundance"],
        "ch4": free_parameters["uniform_ch4_log_abundance"],
    }
    flux_scaled_error_inflation_factor: float = free_parameters[
        "flux_scaled_error_inflation_factor"
    ]
    log10_constant_error_inflation_term: float = free_parameters[
        "log10_constant_error_inflation_term"
    ]

    forward_model_inputs: UserForwardModelInputs = (
        forward_model_module.build_uniform_mixing_ratio_forward_model_with_free_radius(
            planet_radius_relative_to_earth=planet_radius_relative_to_earth,
            uniform_log_abundances=uniform_log_abundances,
        )
    )

    transit_depths_sampled_to_data: xr.DataArray = calculate_transit_model(
        forward_model_inputs=forward_model_inputs,
        resampling_fwhm_fraction=resampling_fwhm_fraction,
    )

    inflated_data_errors: xr.DataArray = inflate_errors_by_flux_scaling(
        data_dataset.data,
        data_dataset.data_error,
        flux_scaled_error_inflation_factor,
        log10_constant_error_inflation_term,
    )

    return xr.Dataset(
        {
            "transit_model": transit_depths_sampled_to_data,
            "transit_data": data_dataset.data,
            "scaled_transit_data_error": inflated_data_errors,
        }
    )


def evaluate_log_likelihood_with_free_parameters(
    free_parameters: ModelFreeParameters,
    resampling_fwhm_fraction: float = fraction_of_reddest_fwhm_to_convolve_with,
    data_dataset: xr.Dataset = data,
) -> float:
    prepared_model: xr.Dataset = prepare_model_for_likelihood_evaluation(
        free_parameters=free_parameters,
        resampling_fwhm_fraction=resampling_fwhm_fraction,
        data_dataset=data_dataset,
    )

    transit_depths_sampled_to_data: xr.DataArray = prepared_model.transit_model
    transit_data: xr.DataArray = prepared_model.transit_data
    scaled_transit_data_errors: xr.DataArray = prepared_model.scaled_transit_data_error

    log_likelihood: float = calculate_log_likelihood(
        transit_depths_sampled_to_data, transit_data, scaled_transit_data_errors
    ).item()

    return log_likelihood


if __name__ == "__main__":
    test_model_parameters: ModelFreeParameters = ModelFreeParameters(
        planet_radius_relative_to_earth=2.90156,
        uniform_ch4_log_abundance=-3.318,
        uniform_h2o_log_abundance=-3.171,
        flux_scaled_error_inflation_factor=1e-4,
        log10_constant_error_inflation_term=-10.0,
    )

    test_log_likelihood: float = evaluate_log_likelihood_with_free_parameters(
        test_model_parameters
    )
    print(f"{test_log_likelihood=}")

    prior = Prior()
    prior.add_parameter("planet_radius_relative_to_earth", dist=(0.5, 20.0))
    prior.add_parameter("uniform_ch4_log_abundance", dist=(-12.0, -1.0))
    prior.add_parameter("uniform_h2o_log_abundance", dist=(-12.0, -1.0))
    prior.add_parameter("flux_scaled_error_inflation_factor", dist=(0.0, 1.0))
    prior.add_parameter("log10_constant_error_inflation_term", dist=(-100.0, 0.0))

    sampler = Sampler(
        prior,
        evaluate_log_likelihood_with_free_parameters,
        n_live=100,
        filepath=output_file_directory / f"{model_directory_label}_nautilus.hdf5",
    )
    sampler.run(verbose=True)

    points, log_w, log_l = sampler.posterior()
    log_z = sampler.log_z

    np.save(output_file_directory / f"{model_directory_label}_points.npy", points)
    np.save(output_file_directory / f"{model_directory_label}_log_w.npy", log_w)
    np.save(output_file_directory / f"{model_directory_label}_log_l.npy", log_l)
    np.save(output_file_directory / f"{model_directory_label}_log_z.npy", log_z)
