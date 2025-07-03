import os
from datetime import datetime
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TypedDict

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from mpi4py.futures import MPIPoolExecutor
from nautilus import Prior, Sampler
from scipy.stats import truncnorm

from calculate_RT import calculate_observed_fluxes_via_two_stream
from model_statistics.calculate_statistics import calculate_log_likelihood
from model_statistics.error_inflation import inflate_errors_by_flux_scaling
from spectrum.bin import resample_spectral_quantity_to_new_wavelengths
from user.input_structs import UserForwardModelInputs

os.environ["OMP_NUM_THREADS"] = "1"

model_directory_label: str = "2M2236b_G395H_logg-normal"

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

reference_model_wavelengths: xr.DataArray = (
    forward_model_module.reference_model_wavelengths
)

data_wavelengths: xr.DataArray = forward_model_module.reference_model_wavelengths
data_fluxes: xr.DataArray = forward_model_module.reference_model_flux_lambda
data_flux_errors: xr.DataArray = forward_model_module.reference_model_flux_lambda_errors


def calculate_emission_model(
    forward_model_inputs: UserForwardModelInputs,
    resampling_fwhm_fraction: float,
) -> float:
    emission_fluxes = calculate_observed_fluxes_via_two_stream(
        user_forward_model_inputs=forward_model_inputs,
        precalculated_crosssection_catalog=None,
    )

    reference_model_wavelengths: xr.DataArray = forward_model_inputs.output_wavelengths

    emission_fluxes_sampled_to_data: xr.DataArray = (
        resample_spectral_quantity_to_new_wavelengths(
            reference_model_wavelengths,
            emission_fluxes.wavelength,
            emission_fluxes,
            fwhm=resampling_fwhm_fraction
            * (
                reference_model_wavelengths.to_numpy()[-1]
                - reference_model_wavelengths.to_numpy()[-2]
            ),
        )
    ).rename("resampled_emission_flux")

    return emission_fluxes_sampled_to_data


class ModelFreeParameters(TypedDict):
    planet_radius_relative_to_earth: float
    planet_log_gravity: float
    uniform_h2o_log_abundance: float
    uniform_co_log_abundance: float
    uniform_co2_log_abundance: float
    uniform_ch4_log_abundance: float
    uniform_Lupu_alk_log_abundance: float
    uniform_h2s_log_abundance: float
    uniform_nh3_log_abundance: float
    photospheric_scaled_temperature: float
    shallow_scaled_temperature_0: float
    shallow_scaled_temperature_1: float
    shallow_scaled_temperature_2: float
    shallow_scaled_temperature_3: float
    shallow_scaled_temperature_4: float
    deep_scaled_temperature_0: float
    deep_scaled_temperature_1: float
    deep_scaled_temperature_2: float
    deep_scaled_temperature_3: float
    flux_scaled_error_inflation_factor: float
    log10_constant_error_inflation_term: float
    fraction_of_reddest_fwhm_to_convolve_with: float
    NRS1_flux_scale_factor: float
    NRS2_flux_scale_factor: float


def prepare_model_for_likelihood_evaluation(
    free_parameters: ModelFreeParameters,
    data_fluxes: xr.DataArray = data_fluxes,
    data_flux_errors: xr.DataArray = data_flux_errors,
) -> xr.Dataset:
    # unpack dict of free parameters
    planet_radius_relative_to_earth: float = free_parameters[
        "planet_radius_relative_to_earth"
    ]

    planet_log_gravity_in_cgs: float = free_parameters["planet_log_gravity"]

    uniform_log_abundances: dict[str, float] = {
        "h2o": free_parameters["uniform_h2o_log_abundance"],
        "co": free_parameters["uniform_co_log_abundance"],
        "co2": free_parameters["uniform_co2_log_abundance"],
        "ch4": free_parameters["uniform_ch4_log_abundance"],
        "Lupu_alk": free_parameters["uniform_Lupu_alk_log_abundance"],
        "h2s": free_parameters["uniform_h2s_log_abundance"],
        "nh3": free_parameters["uniform_nh3_log_abundance"],
    }

    initial_temp_sample: float = free_parameters["photospheric_scaled_temperature"]
    proportions_down: np.ndarray = np.array(
        [
            free_parameters["shallow_scaled_temperature_0"],
            free_parameters["shallow_scaled_temperature_1"],
            free_parameters["shallow_scaled_temperature_2"],
            free_parameters["shallow_scaled_temperature_3"],
            free_parameters["shallow_scaled_temperature_4"],
        ]
    )
    proportions_up: np.ndarray = np.array(
        [
            free_parameters["deep_scaled_temperature_0"],
            free_parameters["deep_scaled_temperature_1"],
            free_parameters["deep_scaled_temperature_2"],
            free_parameters["deep_scaled_temperature_3"],
        ]
    )

    flux_scaled_error_inflation_factor: float = free_parameters[
        "flux_scaled_error_inflation_factor"
    ]
    log10_constant_error_inflation_term: float = free_parameters[
        "log10_constant_error_inflation_term"
    ]

    forward_model_inputs: UserForwardModelInputs = (
        forward_model_module.build_uniform_mixing_ratio_forward_model(
            planet_radius_relative_to_earth=planet_radius_relative_to_earth,
            planet_log_gravity_in_cgs=planet_log_gravity_in_cgs,
            uniform_log_abundances=uniform_log_abundances,
            initial_temp_sample=initial_temp_sample,
            proportions_down=proportions_down,
            proportions_up=proportions_up,
        )
    )

    emission_fluxes_sampled_to_data: xr.DataArray = calculate_emission_model(
        forward_model_inputs=forward_model_inputs,
        resampling_fwhm_fraction=free_parameters[
            "fraction_of_reddest_fwhm_to_convolve_with"
        ],
    )

    inflated_data_errors: xr.DataArray = inflate_errors_by_flux_scaling(
        data_fluxes,
        data_flux_errors,
        flux_scaled_error_inflation_factor,
        log10_constant_error_inflation_term,
    )

    emission_fluxes_sampled_to_data: xr.DataArray = xr.where(
        emission_fluxes_sampled_to_data.wavelength < 4.1,
        emission_fluxes_sampled_to_data * free_parameters["NRS1_flux_scale_factor"],
        emission_fluxes_sampled_to_data * free_parameters["NRS2_flux_scale_factor"],
    )

    return xr.Dataset(
        {
            "emission_model": emission_fluxes_sampled_to_data,
            "emission_data": data_fluxes,
            "scaled_emission_data_error": inflated_data_errors,
        }
    )


def evaluate_log_likelihood_with_free_parameters(
    free_parameters: ModelFreeParameters,
    data_fluxes: xr.DataArray = data_fluxes,
    data_flux_errors: xr.DataArray = data_flux_errors,
) -> float:
    prepared_model: xr.Dataset = prepare_model_for_likelihood_evaluation(
        free_parameters=free_parameters,
        data_fluxes=data_fluxes,
        data_flux_errors=data_flux_errors,
    )

    emission_fluxes_sampled_to_data: xr.DataArray = prepared_model.emission_model
    emission_data: xr.DataArray = prepared_model.emission_data
    scaled_emission_data_errors: xr.DataArray = (
        prepared_model.scaled_emission_data_error
    )

    log_likelihood: float = calculate_log_likelihood(
        emission_fluxes_sampled_to_data, emission_data, scaled_emission_data_errors
    ).item()

    return log_likelihood


if __name__ == "__main__":
    test_model_parameters: ModelFreeParameters = ModelFreeParameters(
        planet_radius_relative_to_earth=7.776098627,
        planet_log_gravity=4.133962209,
        uniform_h2o_log_abundance=-5.940043768,
        uniform_co_log_abundance=-5.695578981,
        uniform_co2_log_abundance=-8.884468544,
        uniform_ch4_log_abundance=-7.663836048,
        uniform_Lupu_alk_log_abundance=-4.953393893,
        uniform_h2s_log_abundance=-11.42842546,
        uniform_nh3_log_abundance=-10.14099491,
        photospheric_scaled_temperature=0.2889458091719745,
        shallow_scaled_temperature_0=0.11159102,
        shallow_scaled_temperature_1=0.02182628,
        shallow_scaled_temperature_2=0.12510834,
        shallow_scaled_temperature_3=0.10768672,
        shallow_scaled_temperature_4=0.01539343,
        deep_scaled_temperature_0=0.02514635,
        deep_scaled_temperature_1=0.01982915,
        deep_scaled_temperature_2=0.06249186,
        deep_scaled_temperature_3=0.32445998,
        flux_scaled_error_inflation_factor=0.0,
        log10_constant_error_inflation_term=-50.0,
        fraction_of_reddest_fwhm_to_convolve_with=1.00,
        NRS1_flux_scale_factor=1.199143965,
        NRS2_flux_scale_factor=1.124040073,
    )

    test_log_likelihood: float = evaluate_log_likelihood_with_free_parameters(
        test_model_parameters
    )

    data_wavelengths: xr.DataArray = forward_model_module.reference_model_wavelengths
    data_fluxes: xr.DataArray = forward_model_module.reference_model_flux_lambda
    data_flux_errors: xr.DataArray = (
        forward_model_module.reference_model_flux_lambda_errors
    )

    # we're not askign
    run_retrieval: bool = True

    # run retrieval
    if run_retrieval:
        run_timestamp: str = datetime.now().strftime("%Y%b%d_%H:%M:%S")

        prior = Prior()
        prior.add_parameter("planet_radius_relative_to_earth", dist=(0.5, 20.0))

        gravity_mean: float = 4.33
        gravity_std: float = 0.12
        gravity_lower_bound: float = 4.33 - 3 * gravity_std
        gravity_upper_bound: float = 4.33 + 3 * gravity_std
        gravity_lower_bound_transformed, gravity_upper_bound_transformed = (
            (gravity_lower_bound - gravity_mean) / gravity_std,
            (gravity_upper_bound - gravity_mean) / gravity_std,
        )
        prior.add_parameter(
            "planet_log_gravity",
            dist=truncnorm(
                gravity_lower_bound_transformed,
                gravity_upper_bound_transformed,
                loc=gravity_mean,
                scale=gravity_std,
            ),
        )
        prior.add_parameter("uniform_h2o_log_abundance", dist=(-12.0, -1.0))
        prior.add_parameter("uniform_co_log_abundance", dist=(-12.0, -1.0))
        prior.add_parameter("uniform_co2_log_abundance", dist=(-12.0, -1.0))
        prior.add_parameter("uniform_ch4_log_abundance", dist=(-12.0, -1.0))
        prior.add_parameter("uniform_Lupu_alk_log_abundance", dist=(-12.0, -1.0))
        prior.add_parameter("uniform_h2s_log_abundance", dist=(-12.0, -1.0))
        prior.add_parameter("uniform_nh3_log_abundance", dist=(-12.0, -1.0))
        prior.add_parameter("photospheric_scaled_temperature", dist=(0.0, 1.0))
        prior.add_parameter("shallow_scaled_temperature_0", dist=(0.0, 1.0))
        prior.add_parameter("shallow_scaled_temperature_1", dist=(0.0, 1.0))
        prior.add_parameter("shallow_scaled_temperature_2", dist=(0.0, 1.0))
        prior.add_parameter("shallow_scaled_temperature_3", dist=(0.0, 1.0))
        prior.add_parameter("shallow_scaled_temperature_4", dist=(0.0, 1.0))
        prior.add_parameter("deep_scaled_temperature_0", dist=(0.0, 1.0))
        prior.add_parameter("deep_scaled_temperature_1", dist=(0.0, 1.0))
        prior.add_parameter("deep_scaled_temperature_2", dist=(0.0, 1.0))
        prior.add_parameter("deep_scaled_temperature_3", dist=(0.0, 1.0))
        prior.add_parameter("flux_scaled_error_inflation_factor", dist=(0.0, 3.0))
        prior.add_parameter("log10_constant_error_inflation_term", dist=(-100.0, 1.0))
        prior.add_parameter(
            "fraction_of_reddest_fwhm_to_convolve_with", dist=(0.01, 3.00)
        )
        prior.add_parameter("NRS1_flux_scale_factor", dist=(0.75, 1.25))
        prior.add_parameter("NRS2_flux_scale_factor", dist=(0.75, 1.25))

        run_output_file_prefix: str = f"{model_directory_label}_{run_timestamp}"

        sampler = Sampler(
            prior,
            evaluate_log_likelihood_with_free_parameters,
            n_live=1000,
            pool=MPIPoolExecutor(),
            filepath=output_file_directory / f"{run_output_file_prefix}_nautilus.hdf5",
        )
        sampler.run(verbose=True)

        points, log_w, log_l = sampler.posterior()
        log_z = sampler.log_z

        np.save(output_file_directory / f"{run_output_file_prefix}_points.npy", points)
        np.save(output_file_directory / f"{run_output_file_prefix}_log_w.npy", log_w)
        np.save(output_file_directory / f"{run_output_file_prefix}_log_l.npy", log_l)
        np.save(output_file_directory / f"{run_output_file_prefix}_log_z.npy", log_z)
