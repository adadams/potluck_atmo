import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TypedDict

import msgspec
import numpy as np
import xarray as xr
from nautilus import Prior, Sampler

sys.path.append(str(Path(__file__).parent.parent.parent))

from potluck.basic_types import LogMixingRatioValue, PositiveValue, TemperatureValue
from potluck.model_builders.default_builders import (
    DefaultFundamentalParameterInputs,
    UniformGasChemistryInputs,
    calculate_transmission_model,
)
from potluck.model_statistics.calculate_statistics import calculate_log_likelihood
from potluck.model_statistics.error_inflation import inflate_errors_by_flux_scaling
from potluck.temperature.models import IsothermalTemperatureModelArguments
from potluck.xarray_functional_wrappers import convert_units
from user.Planet0.build_model import ModelInputs, build_model_from_inputs

current_directory: Path = Path(__file__).parent


class ModelFreeParameters(TypedDict):
    planet_radius_in_meters: PositiveValue
    isothermal_temperature: TemperatureValue
    uniform_h2o_log_abundance: LogMixingRatioValue
    uniform_co_log_abundance: LogMixingRatioValue
    flux_scaled_error_inflation_factor: float
    log10_constant_error_inflation_term: float


def replace_inputs_with_free_parameters(
    inputs_from_toml_file: ModelInputs, free_parameters: ModelFreeParameters
):
    new_fundamental_parameters: DefaultFundamentalParameterInputs = (
        msgspec.structs.replace(
            inputs_from_toml_file.fundamental_parameters,
            planet_radius=free_parameters["planet_radius_in_meters"],
            radius_units="m",
        )
    )

    new_temperature_parameters: IsothermalTemperatureModelArguments = (
        msgspec.structs.replace(
            inputs_from_toml_file.temperature_profile,
            model_parameters=msgspec.structs.replace(
                inputs_from_toml_file.temperature_profile.model_parameters,
                temperature=free_parameters["isothermal_temperature"],
            ),
        )
    )

    new_gas_chemistry_inputs: UniformGasChemistryInputs = msgspec.structs.replace(
        inputs_from_toml_file.gas_chemistry,
        log_mixing_ratios={
            "h2o": free_parameters["uniform_h2o_log_abundance"],
            "co": free_parameters["uniform_co_log_abundance"],
        },
    )

    new_inputs_from_toml_file: ModelInputs = msgspec.structs.replace(
        inputs_from_toml_file,
        fundamental_parameters=new_fundamental_parameters,
        temperature_profile=new_temperature_parameters,
        gas_chemistry=new_gas_chemistry_inputs,
    )

    return new_inputs_from_toml_file


def evaluate_log_likelihood_with_free_parameters(
    free_parameters: ModelFreeParameters,
    inputs_from_toml_file: ModelInputs,
    data: xr.Dataset,
    resampling_fwhm_fraction: float,
) -> float:
    free_parameter_inputs: ModelInputs = replace_inputs_with_free_parameters(
        inputs_from_toml_file, free_parameters
    )

    free_parameter_forward_model: xr.DataTree = build_model_from_inputs(
        free_parameter_inputs
    )

    free_parameter_transmission_model: xr.DataArray = calculate_transmission_model(
        forward_model_inputs=free_parameter_forward_model,
        resampling_fwhm_fraction=resampling_fwhm_fraction,
    )

    inflated_data_errors: xr.DataArray = inflate_errors_by_flux_scaling(
        data.transit_depth,
        data.transit_depth_uncertainty,
        free_parameters["flux_scaled_error_inflation_factor"],
        free_parameters["log10_constant_error_inflation_term"],
    )

    log_likelihood: float = calculate_log_likelihood(
        free_parameter_transmission_model, data.transit_depth, inflated_data_errors
    ).item()

    return log_likelihood


if __name__ == "__main__":
    input_toml_filepath: Path = current_directory / "model_inputs.toml"

    with open(input_toml_filepath, "rb") as input_toml_file:
        inputs_from_toml_file: dict = msgspec.toml.decode(
            input_toml_file.read(), type=ModelInputs
        )

    model_ID: str = inputs_from_toml_file.metadata["model_ID"]
    run_date: str = inputs_from_toml_file.metadata["run_date"]

    save_name: str = f"{model_ID}_{run_date}"

    Planet0_data_filepath: Path = current_directory / "Planet0_data.nc"
    Planet0_data: xr.Dataset = xr.open_dataset(Planet0_data_filepath)
    Planet0_data_in_cgs: xr.Dataset = convert_units(Planet0_data, {"wavelength": "cm"})

    log_likelihood_from_original_parameters: float = (
        evaluate_log_likelihood_with_free_parameters(
            free_parameters={
                "planet_radius_in_meters": 7.9559e7,
                "isothermal_temperature": 1500.0,
                "uniform_h2o_log_abundance": -3.523,
                "uniform_co_log_abundance": -3.456,
                "flux_scaled_error_inflation_factor": 0.0,
                "log10_constant_error_inflation_term": -100.0,
            },
            inputs_from_toml_file=inputs_from_toml_file,
            data=Planet0_data_in_cgs,
            resampling_fwhm_fraction=0.1,
        )
    )
    print(f"{log_likelihood_from_original_parameters=}")

    run_retrieval = True

    if run_retrieval:
        prior = Prior()
        prior.add_parameter(
            "planet_radius_in_meters", dist=(1.787e7, 3.575e8)
        )  # 0.25 - 5 Rjup
        prior.add_parameter("isothermal_temperature", dist=(500.0, 3500.0))
        prior.add_parameter("uniform_h2o_log_abundance", dist=(-10.0, -0.3011))
        prior.add_parameter("uniform_co_log_abundance", dist=(-10.0, -0.3011))
        prior.add_parameter("flux_scaled_error_inflation_factor", dist=(0.0, 3.0))
        prior.add_parameter("log10_constant_error_inflation_term", dist=(-100.0, 1.0))

        likelihood_sampling_function: Callable[[ModelFreeParameters], float] = partial(
            evaluate_log_likelihood_with_free_parameters,
            inputs_from_toml_file=inputs_from_toml_file,
            data=Planet0_data_in_cgs,
            resampling_fwhm_fraction=0.1,
        )

        sampler = Sampler(
            prior,
            likelihood_sampling_function,
            n_live=500,
            # pool=12,
            filepath=f"{save_name}_nautilus.hdf5",
        )
        sampler.run(verbose=True)

        points, log_w, log_l = sampler.posterior()
        log_z = sampler.log_z

        np.save(current_directory / f"{save_name}_points.npy", points)
        np.save(current_directory / f"{save_name}_log_w.npy", log_w)
        np.save(current_directory / f"{save_name}_log_l.npy", log_l)
        np.save(current_directory / f"{save_name}_log_z.npy", log_z)
