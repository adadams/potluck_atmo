from pathlib import Path

import msgspec
import xarray as xr

from model_builders.default_builders import (
    DefaultFundamentalParameterInputs,
    DefaultObservableInputs,
    EvenlyLogSpacedPressureProfileInputs,
    UniformGasChemistryInputs,
    build_default_fundamental_parameters,
    build_default_observable_inputs,
    build_forward_model,
    build_pressure_profile_from_log_pressures,
    build_temperature_profile,
    build_uniform_gas_chemistry,
    calculate_emission_model,
    compile_vertical_structure,
)
from temperature.models import (
    PietteTemperatureModelParameters,
    TemperatureBounds,
    generate_piette_model,
)
from temperature.protocols import TemperatureModelConstructor
from xarray_serialization import SerializablePrimitiveType

current_directory: Path = Path(__file__).parent


class PietteTemperatureModelArguments(msgspec.Struct, kw_only=True):
    model_inputs: TemperatureBounds
    model_parameters: PietteTemperatureModelParameters


class TestModelInputs(msgspec.Struct):
    metadata: dict[str, SerializablePrimitiveType]
    fundamental_parameters: DefaultFundamentalParameterInputs
    pressure_profile: EvenlyLogSpacedPressureProfileInputs
    temperature_profile: PietteTemperatureModelArguments
    gas_chemistry: UniformGasChemistryInputs
    observable_inputs: DefaultObservableInputs
    reference_data: dict[str, SerializablePrimitiveType]


def build_model_from_inputs(inputs_from_file: TestModelInputs) -> xr.DataTree:
    fundamental_parameters: xr.Dataset = build_default_fundamental_parameters(
        inputs_from_file.fundamental_parameters
    )

    pressure_profile: xr.Dataset = build_pressure_profile_from_log_pressures(
        inputs_from_file.pressure_profile
    )

    temperature_model_constructor: TemperatureModelConstructor = generate_piette_model

    temperature_profile: xr.Dataset = build_temperature_profile(
        temperature_model_constructor=temperature_model_constructor,
        temperature_model_parameters=inputs_from_file.temperature_profile.model_parameters,
        pressure_profile=pressure_profile,
    )

    gas_chemistry: xr.Dataset = build_uniform_gas_chemistry(
        gas_chemistry_inputs=inputs_from_file.gas_chemistry,
        pressure_profile=pressure_profile,
    )

    model_metadata: dict[str, SerializablePrimitiveType] = inputs_from_file.metadata

    atmospheric_structure: xr.DataTree = compile_vertical_structure(
        fundamental_parameters=fundamental_parameters,
        pressure_profile=pressure_profile,
        temperature_profile=temperature_profile,
        gas_chemistry=gas_chemistry,
        additional_attributes=model_metadata,
    )

    reference_model_filepath: Path = inputs_from_file.reference_data[
        "reference_wavelength_filename"
    ]
    reference_model: xr.Dataset = xr.open_dataset(reference_model_filepath)
    output_wavelengths: xr.DataArray = reference_model.wavelength.assign_attrs(
        units="micron"
    )

    observable_inputs: xr.Dataset = build_default_observable_inputs(
        observable_inputs=inputs_from_file.observable_inputs,
        output_wavelengths=output_wavelengths,
    )

    gas_opacity_catalog_filepath: Path = Path(
        inputs_from_file.reference_data["gas_opacity_catalog_filepath"]
    )
    crosssection_catalog: xr.Dataset = xr.open_dataset(gas_opacity_catalog_filepath)

    forward_model_structure: xr.DataTree = build_forward_model(
        atmospheric_structure_by_layer=atmospheric_structure,
        temperature_profile=temperature_profile,
        crosssection_catalog=crosssection_catalog,
        observable_inputs=observable_inputs,
    )

    return forward_model_structure


if __name__ == "__main__":
    input_forward_model_structure_filepath: Path = (
        current_directory / "test_forward_model_structure.nc"
    )
    input_toml_filepath: Path = current_directory / "test_model_inputs.toml"

    # check if pre-compiled forward model structure exists
    if input_forward_model_structure_filepath.is_file():
        print(
            f"Using pre-compiled forward model structure at {input_forward_model_structure_filepath}."
        )
        test_forward_model_structure: xr.DataTree = xr.open_datatree(
            current_directory / "test_forward_model_structure.nc"
        )

    elif input_toml_filepath.is_file():
        print(
            f"Generating forward model structure from input toml file at {input_toml_filepath}."
        )
        with open(input_toml_filepath, "rb") as input_toml_file:
            inputs_from_toml_file: dict = msgspec.toml.decode(
                input_toml_file.read(), type=TestModelInputs
            )

        test_forward_model_structure: xr.DataTree = build_model_from_inputs(
            inputs_from_file=inputs_from_toml_file
        )

        test_forward_model_structure.to_netcdf(
            current_directory / "test_forward_model_structure.nc"
        )

    else:
        raise FileNotFoundError(
            f"Could not find test forward model structure at {input_forward_model_structure_filepath} "
            f"or input toml file to generate it at {input_toml_filepath}."
        )

    test_emission_model: xr.DataTree = calculate_emission_model(
        forward_model_inputs=test_forward_model_structure,
        resampling_fwhm_fraction=0.1,
    )

    test_emission_model.to_netcdf(current_directory / "test_emission_model.nc")
