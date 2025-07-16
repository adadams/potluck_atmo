from importlib import import_module
from pathlib import Path
from types import ModuleType

import xarray as xr

from model_builders.default_builders import (
    build_forward_model,
    calculate_emission_model,
    compile_vertical_structure,
)

current_directory: Path = Path(__file__).parent

if __name__ == "__main__":
    # doing it this way instead of top-level import because it's essentially
    # a mock-up of what will become a toml file for inputs
    test_model_inputs: ModuleType = import_module("test_model_inputs")

    test_atmospheric_structure: xr.DataTree = compile_vertical_structure(
        fundamental_parameters=test_model_inputs.fundamental_parameters,
        pressure_profile=test_model_inputs.pressure_profile,
        temperature_profile=test_model_inputs.temperature_profile,
        gas_chemistry=test_model_inputs.gas_chemistry,
        additional_attributes=test_model_inputs.model_metadata,
    )

    test_forward_model_structure: xr.DataTree = build_forward_model(
        atmospheric_structure_by_layer=test_atmospheric_structure,
        temperature_profile=test_model_inputs.temperature_profile,
        crosssection_catalog_dataset=test_model_inputs.crosssection_catalog_dataset,
        observable_inputs=test_model_inputs.observable_inputs,
    )

    test_forward_model_structure.to_netcdf(
        current_directory / "test_forward_model_structure.nc"
    )

    test_emission_model: xr.DataTree = calculate_emission_model(
        forward_model_inputs=test_forward_model_structure,
        resampling_fwhm_fraction=0.1,
    )

    test_emission_model.to_netcdf(current_directory / "test_emission_model.nc")
