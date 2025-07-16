from functools import partial
from pathlib import Path
from typing import Final

import msgspec
import xarray as xr

from model_builders.default_builders import (
    FundamentalParameterInputs,
    ObservableInputs,
    PressureProfileInputs,
    UniformGasChemistryInputs,
    build_default_fundamental_parameters,
    build_observable_inputs,
    build_pressure_profile_from_log_pressures,
    build_temperature_profile,
    build_uniform_gas_chemistry,
)
from temperature import models as temperature_models
from temperature.protocols import TemperatureModelConstructor
from xarray_serialization import SerializablePrimitiveType

current_directory: Path = Path(__file__).parent

PARSEC_TO_CM: Final[float] = 3.08567758128e18


class TemperatureModelArguments(msgspec.Struct, kw_only=True):
    model_constructor: str = "generate_piette_model"
    model_inputs: temperature_models.TemperatureBounds
    model_parameters: temperature_models.PietteTemperatureModelParameters


class TestModelInputs(msgspec.Struct):
    fundamental_parameters: FundamentalParameterInputs
    pressure_profile: PressureProfileInputs
    temperature_profile: TemperatureModelArguments
    gas_chemistry: UniformGasChemistryInputs
    observable_inputs: ObservableInputs
    metadata: dict[str, SerializablePrimitiveType]


input_toml_filepath: Path = current_directory / "test_model_inputs.toml"
with open(input_toml_filepath, "rb") as input_toml_file:
    inputs_from_toml_file: dict = msgspec.toml.decode(
        input_toml_file.read(), type=TestModelInputs
    )

fundamental_parameters: xr.Dataset = build_default_fundamental_parameters(
    inputs_from_toml_file.fundamental_parameters
)

pressure_profile: xr.Dataset = build_pressure_profile_from_log_pressures(
    inputs_from_toml_file.pressure_profile
)

temperature_model_constructor: TemperatureModelConstructor = partial(
    getattr(
        temperature_models,
        inputs_from_toml_file.temperature_profile.model_constructor,
    ),
    model_inputs=inputs_from_toml_file.temperature_profile.model_inputs,
)


temperature_profile: xr.Dataset = build_temperature_profile(
    temperature_model_constructor=temperature_model_constructor,
    temperature_model_parameters=inputs_from_toml_file.temperature_profile.model_parameters,
    pressure_profile=pressure_profile,
)

gas_chemistry: xr.Dataset = build_uniform_gas_chemistry(
    gas_chemistry_inputs=inputs_from_toml_file.gas_chemistry,
    pressure_profile=pressure_profile,
)

reference_model_filepath: Path = (
    current_directory / "test_reference_model_for_wavelengths.nc"
)
reference_model: xr.Dataset = xr.open_dataset(reference_model_filepath)
output_wavelengths: xr.DataArray = reference_model.wavelength.assign_attrs(
    units="micron"
)

observable_inputs: xr.Dataset = build_observable_inputs(
    observable_inputs=inputs_from_toml_file.observable_inputs,
    output_wavelengths=output_wavelengths,
)

opacity_catalog: str = "jwst50k"  # "jwst50k", "wide-jwst", "wide"

reference_data_directory: Path = Path("/media/gba8kj/Orange")
# reference_data_directory: Path = Path("/Volumes/Orange")
opacities_directory: Path = reference_data_directory / "Opacities_0v10" / "gases"
catalog_filepath: Path = opacities_directory / f"{opacity_catalog}.nc"

crosssection_catalog_dataset: xr.Dataset = xr.open_dataset(catalog_filepath)

model_metadata: dict[str, SerializablePrimitiveType] = inputs_from_toml_file.metadata
