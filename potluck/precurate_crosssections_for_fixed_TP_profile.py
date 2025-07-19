from importlib import import_module
from pathlib import Path
from types import ModuleType

import xarray as xr

from compile_crosssection_data import curate_crosssection_catalog
from compile_vertical_structure import compile_vertical_structure_for_forward_model
from user.input_structs import UserForwardModelInputs

model_directory_label: str = "R1c_retrieval"

project_directory: Path = Path.cwd()
user_directory: Path = project_directory / "user"
model_directory: Path = user_directory / f"{model_directory_label}_model"
input_file_directory: Path = model_directory / "input_files"

parent_directory: str = "user"

forward_model_module: ModuleType = import_module(
    f"{parent_directory}.{model_directory_label}_model.input_files.{model_directory_label}_forward_model_inputs"
)
default_forward_model_inputs: UserForwardModelInputs = (
    forward_model_module.default_forward_model_inputs
)

data: xr.Dataset = forward_model_module.data_dataset
reference_model_wavelengths: xr.DataArray = data.wavelength

maximum_wavelength_needed_in_catalog: float = (
    reference_model_wavelengths.max()
    + 0.01 * (reference_model_wavelengths.max() - reference_model_wavelengths.min())
).item()

compiled_vertical_structure: xr.Dataset = compile_vertical_structure_for_forward_model(
    default_forward_model_inputs.vertical_inputs
).by_layer

precurated_crosssection_catalog: xr.Dataset = curate_crosssection_catalog(
    crosssection_catalog=default_forward_model_inputs.crosssection_catalog,
    temperatures_by_layer=compiled_vertical_structure.temperature,
    pressures_by_layer=compiled_vertical_structure.pressure,
    species_present_in_model=compiled_vertical_structure.number_density.species.values,
).sel(wavelength=slice(None, maximum_wavelength_needed_in_catalog))

precurated_crosssection_catalog_nonfixed_TP: xr.Dataset = (
    default_forward_model_inputs.crosssection_catalog.sel(
        wavelength=slice(None, maximum_wavelength_needed_in_catalog)
    ).get(compiled_vertical_structure.number_density.species.values)
)
print(f"{precurated_crosssection_catalog_nonfixed_TP=}")

precurated_crosssection_catalog.to_netcdf(
    input_file_directory / f"{model_directory_label}_precurated_crosssection_catalog.nc"
)

precurated_crosssection_catalog_nonfixed_TP.to_netcdf(
    input_file_directory
    / f"{model_directory_label}_precurated_crosssection_catalog_nonfixed_TP.nc"
)
