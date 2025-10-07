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

from build_model import ModelInputs, evaluate_transmission_spectrum

from potluck.basic_types import LogMixingRatioValue, PositiveValue, TemperatureValue
from potluck.model_builders.default_builders import (
    DefaultFundamentalParameterInputs,
    EvenlyLogSpacedPressureProfileInputs,
    UniformGasChemistryInputs,
)
from potluck.model_statistics.calculate_statistics import calculate_log_likelihood
from potluck.temperature.models import IsothermalTemperatureModelArguments
from potluck.xarray_functional_wrappers import convert_units

current_directory: Path = Path(__file__).parent


class ModelFreeParameters(TypedDict):
    planet_radius_in_meters: PositiveValue
    isothermal_temperature: TemperatureValue
    deepest_log10_pressure: float
    uniform_he_log_abundance: LogMixingRatioValue
    uniform_h2o_log_abundance: LogMixingRatioValue
    uniform_ch4_log_abundance: LogMixingRatioValue


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

    new_pressure_profile: EvenlyLogSpacedPressureProfileInputs = (
        msgspec.structs.replace(
            inputs_from_toml_file.pressure_profile,
            deepest_log10_pressure=free_parameters["deepest_log10_pressure"],
            number_of_levels=int(
                inputs_from_toml_file.pressure_profile.number_of_levels
                * (
                    free_parameters["deepest_log10_pressure"]
                    - inputs_from_toml_file.pressure_profile.shallowest_log10_pressure
                )
                / (
                    inputs_from_toml_file.pressure_profile.deepest_log10_pressure
                    - inputs_from_toml_file.pressure_profile.shallowest_log10_pressure
                )
            ),
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
            "he": free_parameters["uniform_he_log_abundance"],
            "h2o": free_parameters["uniform_h2o_log_abundance"],
            "ch4": free_parameters["uniform_ch4_log_abundance"],
        },
    )

    new_inputs_from_toml_file: ModelInputs = msgspec.structs.replace(
        inputs_from_toml_file,
        pressure_profile=new_pressure_profile,
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

    free_parameter_transmission_model: xr.DataArray = evaluate_transmission_spectrum(
        inputs_from_toml_file=free_parameter_inputs,
        resampling_fwhm_fraction=resampling_fwhm_fraction,
    )

    log_likelihood: float = calculate_log_likelihood(
        free_parameter_transmission_model,
        data.transit_depth,
        data.transit_depth_uncertainty,
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

    T3B_data_filepath: Path = current_directory / "T3B_risotto_Full_data.nc"
    T3B_data: xr.Dataset = xr.open_dataset(T3B_data_filepath)
    T3B_data_in_cgs: xr.Dataset = convert_units(T3B_data, {"wavelength": "cm"})

    log_likelihood_from_original_parameters: float = (
        evaluate_log_likelihood_with_free_parameters(
            free_parameters={
                "planet_radius_in_meters": 7.9559e7,
                "isothermal_temperature": 800.0,
                "deepest_log10_pressure": 0.0,
                "uniform_he_log_abundance": -0.770,
                "uniform_h2o_log_abundance": -3.523,
                "uniform_ch4_log_abundance": -3.456,
            },
            inputs_from_toml_file=inputs_from_toml_file,
            data=T3B_data_in_cgs,
            resampling_fwhm_fraction=200.0,
        )
    )
    print(f"{log_likelihood_from_original_parameters=}")

    run_retrieval = True

    if run_retrieval:
        prior = Prior()
        radius_range = (10**6.804, 10**7.804)
        prior.add_parameter("planet_radius_in_meters", dist=radius_range)
        prior.add_parameter("isothermal_temperature", dist=(100.0, 1000.0))
        prior.add_parameter("deepest_log10_pressure", dist=(-5.0, 2.5))
        prior.add_parameter("uniform_he_log_abundance", dist=(-10.0, -0.3011))
        prior.add_parameter("uniform_h2o_log_abundance", dist=(-10.0, -1.0))
        prior.add_parameter("uniform_ch4_log_abundance", dist=(-10.0, -1.0))

        likelihood_sampling_function: Callable[[ModelFreeParameters], float] = partial(
            evaluate_log_likelihood_with_free_parameters,
            inputs_from_toml_file=inputs_from_toml_file,
            data=T3B_data_in_cgs,
            resampling_fwhm_fraction=200.0,
        )

        sampler = Sampler(
            prior,
            likelihood_sampling_function,
            n_live=1000,
            pool=4,
            filepath=f"{save_name}_nautilus.hdf5",
        )
        sampler.run(verbose=True)

        points, log_w, log_l = sampler.posterior()
        log_z = sampler.log_z

        np.save(current_directory / f"{save_name}_points.npy", points)
        np.save(current_directory / f"{save_name}_log_w.npy", log_w)
        np.save(current_directory / f"{save_name}_log_l.npy", log_l)
        np.save(current_directory / f"{save_name}_log_z.npy", log_z)
