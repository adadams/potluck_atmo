from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import msgspec
import xarray as xr

from material.material import crosssections_to_optical_depths

current_directory: Path = Path(__file__).parent

molecular_reference_data_directory: Path = (
    current_directory / "molecular_crosssections" / "reference_data"
)

gas_properties_filepath: Path = (
    molecular_reference_data_directory / "gas_properties.toml"
)

GAS_PROPERTIES: dict[str, str] = msgspec.toml.decode(
    gas_properties_filepath.read_text()
)


def check_number_density_dataset_has_valid_variables(
    number_density_dataset: xr.Dataset,
    reference_gas_list: Sequence[str] = GAS_PROPERTIES.keys(),
) -> bool:
    # Note: data_vars won't return coordinate variables, e.g. pressure

    return all(
        variable_name in reference_gas_list
        for variable_name in number_density_dataset.data_vars
    )


class GasOpticalDepthInputs(TypedDict):
    crosssection_dataset: xr.Dataset
    number_density_dataset: xr.Dataset
    path_length_dataarray: xr.DataArray


def calculate_gas_optical_depths(
    crosssection_dataset: xr.Dataset,
    number_density_dataset: xr.Dataset,
    path_length_dataarray: xr.DataArray,
) -> xr.DataArray:
    return xr.Dataset(
        {
            variable_name: crosssections_to_optical_depths(
                crosssection_dataset.get(variable_name),
                number_density_dataset.get(variable_name),
                path_length_dataarray,
            )
            for variable_name in number_density_dataset.data_vars
        }
    )


def map_vertical_structure_to_optical_depth_inputs(
    crosssection_dataset: xr.Dataset, vertical_structure_datatree: xr.DataTree
) -> GasOpticalDepthInputs:
    number_density_dataset: xr.Dataset = vertical_structure_datatree[
        "molecular_inputs"
    ]["number_densities"]

    path_length_dataarray: xr.DataArray = vertical_structure_datatree[
        "vertical_structure"
    ]["altitude"]

    return GasOpticalDepthInputs(
        crosssection_dataset=crosssection_dataset,
        number_density_dataset=number_density_dataset,
        path_length_dataarray=path_length_dataarray,
    )


if __name__ == "__main__":
    test_opacity_catalog: str = "nir"
    test_species: str = "H2O"

    current_directory: Path = Path(__file__).parent
    opacity_data_directory: Path = (
        current_directory / "molecular_crosssections" / "reference_data"
    )

    catalog_filepath: Path = opacity_data_directory / f"{test_opacity_catalog}.nc"

    crosssection_catalog_dataset: xr.Dataset = xr.open_dataset(catalog_filepath)

    test_data_structure_directory: Path = (
        current_directory / "test_inputs" / "test_data_structures"
    )

    test_vertical_structure_dataset_path: Path = (
        test_data_structure_directory / "test_vertical_structure.nc"
    )

    test_vertical_structure_datatree: xr.DataTree = xr.open_datatree(
        test_vertical_structure_dataset_path
    )

    test_vertical_structure_dataset: xr.Dataset = test_vertical_structure_datatree[
        "vertical_structure"
    ]

    crosssection_catalog_dataset_interpolated_to_model: xr.Dataset = (
        crosssection_catalog_dataset.interp(
            temperature=test_vertical_structure_dataset.temperature,
            pressure=test_vertical_structure_dataset.pressure,
        )
    )

    gas_optical_depth_arguments: GasOpticalDepthInputs = (
        map_vertical_structure_to_optical_depth_inputs(
            crosssection_dataset=crosssection_catalog_dataset_interpolated_to_model,
            vertical_structure_datatree=test_vertical_structure_datatree,
        )
    )

    print(
        f"{check_number_density_dataset_has_valid_variables(gas_optical_depth_arguments['number_density_dataset'])=}"
    )

    gas_optical_depths: xr.Dataset = calculate_gas_optical_depths(
        crosssection_dataset=gas_optical_depth_arguments["crosssection_dataset"],
        number_density_dataset=gas_optical_depth_arguments["number_density_dataset"],
        path_length_dataarray=gas_optical_depth_arguments["path_length_dataarray"],
    )

    test_optical_depth: xr.DataArray = gas_optical_depths[test_species.lower()]

    gas_optical_depths.to_netcdf(
        test_data_structure_directory
        / f"test_gas_optical_depths_{test_opacity_catalog}.nc"
    )
