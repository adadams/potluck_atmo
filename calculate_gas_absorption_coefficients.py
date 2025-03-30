from collections.abc import Sequence
from pathlib import Path

import msgspec
import xarray as xr

from material.absorbing.from_crosssections import (
    crosssections_to_attenuation_coefficients,
)

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

    test_vertical_inputs_datatree: xr.DataTree = xr.open_datatree(
        test_vertical_structure_dataset_path
    )

    test_vertical_structure_dataset: xr.Dataset = test_vertical_inputs_datatree[
        "vertical_structure"
    ].to_dataset()

    test_molecular_inputs_datatree: xr.DataTree = test_vertical_inputs_datatree[
        "molecular_inputs"
    ]
    test_number_densities_dataset: xr.Dataset = test_molecular_inputs_datatree[
        "number_densities"
    ].to_dataset()

    crosssection_catalog_dataset_interpolated_to_model: xr.Dataset = (
        (
            crosssection_catalog_dataset.interp(
                temperature=test_vertical_structure_dataset.temperature,
                pressure=test_vertical_structure_dataset.pressure,
            )
        )
        .get(list(test_number_densities_dataset.data_vars))
        .to_array(dim="species", name="crosssections")
    )

    test_number_densities_dataarray: xr.DataArray = (
        test_number_densities_dataset.to_array(dim="species", name="number_density")
    )

    print(
        f"{check_number_density_dataset_has_valid_variables(test_number_densities_dataset)=}"
    )

    gas_absorption_coefficients = (
        crosssections_to_attenuation_coefficients(
            crosssection_catalog_dataset_interpolated_to_model,
            test_number_densities_dataarray,
        )
        .rename("gas_absorption_coefficient")
        .assign_attrs(units="cm^(-1)")
    )

    gas_absorption_coefficients.to_netcdf(
        test_data_structure_directory
        / f"test_gas_absorption_coefficients_{test_opacity_catalog}.nc"
    )
