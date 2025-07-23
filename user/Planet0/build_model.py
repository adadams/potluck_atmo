import sys
from pathlib import Path

import msgspec
import xarray as xr

sys.path.append(str(Path(__file__).parent.parent.parent))

from potluck.model_builders.default_builders import (
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
    calculate_transmission_model,
    compile_vertical_structure,
)
from potluck.temperature.models import (
    IsothermalTemperatureModelArguments,
    generate_isothermal_model,
)
from potluck.temperature.protocols import TemperatureModelConstructor
from potluck.xarray_serialization import AttrsType

current_directory: Path = Path(__file__).parent


class ModelInputs(msgspec.Struct):
    metadata: AttrsType
    fundamental_parameters: DefaultFundamentalParameterInputs
    pressure_profile: EvenlyLogSpacedPressureProfileInputs
    temperature_profile: IsothermalTemperatureModelArguments
    gas_chemistry: UniformGasChemistryInputs
    observable_inputs: DefaultObservableInputs
    reference_data: AttrsType


def build_model_from_inputs(inputs_from_file: ModelInputs) -> xr.DataTree:
    fundamental_parameters: xr.Dataset = build_default_fundamental_parameters(
        inputs_from_file.fundamental_parameters
    )

    pressure_profile: xr.Dataset = build_pressure_profile_from_log_pressures(
        inputs_from_file.pressure_profile
    )

    temperature_model_constructor: TemperatureModelConstructor = (
        generate_isothermal_model
    )

    temperature_profile: xr.Dataset = build_temperature_profile(
        temperature_model_constructor=temperature_model_constructor,
        temperature_model_parameters=inputs_from_file.temperature_profile.model_parameters,
        pressure_profile=pressure_profile,
    )

    gas_chemistry: xr.Dataset = build_uniform_gas_chemistry(
        gas_chemistry_inputs=inputs_from_file.gas_chemistry,
        pressure_profile=pressure_profile,
    )

    model_metadata: AttrsType = inputs_from_file.metadata

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


def evaluate_transmission_spectrum(
    inputs_from_toml_file: ModelInputs, resampling_fwhm_fraction: float
):
    forward_model: xr.DataTree = build_model_from_inputs(inputs_from_toml_file)

    transmission_model_spectrum: xr.DataArray = calculate_transmission_model(
        forward_model_inputs=forward_model,
        resampling_fwhm_fraction=resampling_fwhm_fraction,
    )

    return transmission_model_spectrum


if __name__ == "__main__":
    input_toml_filepath: Path = current_directory / "model_inputs.toml"

    if input_toml_filepath.is_file():
        with open(input_toml_filepath, "rb") as input_toml_file:
            inputs_from_toml_file: dict = msgspec.toml.decode(
                input_toml_file.read(), type=ModelInputs
            )

    model_ID: str = inputs_from_toml_file.metadata["model_ID"]

    forward_model_structure_filepath: Path = (
        current_directory / f"{model_ID}_forward_model_structure.nc"
    )

    # check if pre-compiled forward model structure exists
    if forward_model_structure_filepath.is_file():
        print(
            f"Using pre-compiled forward model structure at {forward_model_structure_filepath}."
        )
        forward_model_structure: xr.DataTree = xr.open_datatree(
            forward_model_structure_filepath
        )

    else:
        print(
            f"Generating forward model structure from input toml file at {input_toml_filepath}."
        )
        with open(input_toml_filepath, "rb") as input_toml_file:
            inputs_from_toml_file: dict = msgspec.toml.decode(
                input_toml_file.read(), type=ModelInputs
            )

        forward_model_structure: xr.DataTree = build_model_from_inputs(
            inputs_from_file=inputs_from_toml_file
        )

        forward_model_structure.to_netcdf(forward_model_structure_filepath)

    transmission_model: xr.DataTree = calculate_transmission_model(
        forward_model_inputs=forward_model_structure,
        resampling_fwhm_fraction=0.1,
    )

    radius_printout: str = f"{forward_model_structure['atmospheric_structure_by_layer']['vertical_structure']['planet_radius'].item():.3g}m"
    base_pressure_printout: str = f"{forward_model_structure['atmospheric_structure_by_layer']['vertical_structure']['pressure'].max().item():.3g}barye"

    transmission_model.to_netcdf(
        current_directory
        / f"{model_ID}_transmission_model_radius_{radius_printout}_base_pressure_{base_pressure_printout}.nc"
    )
