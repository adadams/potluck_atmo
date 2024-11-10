from collections.abc import Iterable
from pathlib import Path
from typing import Final, Optional, TypedDict

import xarray as xr

from material.scattering.rayleigh import calculate_rayleigh_scattering_crosssection

MICRONS_TO_CM: Final[float] = 1e-4


class RayleighScatteringInputs(TypedDict):
    wavelength_dataarray: xr.DataArray
    crosssection_dataset: xr.Dataset


def calculate_rayleigh_scattering_crosssections(
    wavelength_dataarray: xr.DataArray,
    crosssection_dataset: xr.Dataset,
    list_of_species: Optional[Iterable[str]] = None,
) -> xr.Dataset:
    if list_of_species is None:
        list_of_species = crosssection_dataset.data_vars

    return xr.Dataset(
        {
            variable_name: calculate_rayleigh_scattering_crosssection(
                wavelength_dataarray,
                crosssection_dataset.get(variable_name),
            )
            for variable_name in list_of_species
        }
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

    test_vertical_structure_datatree: xr.Dataset = xr.open_datatree(
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

    test_wavelengths: xr.DataArray = (
        crosssection_catalog_dataset_interpolated_to_model.h2o.wavelength
    ) * MICRONS_TO_CM

    test_species_list: Iterable[str] = test_vertical_structure_datatree[
        "molecular_inputs"
    ]["number_densities"].data_vars

    test_rayleigh_crosssections: xr.Dataset = (
        calculate_rayleigh_scattering_crosssections(
            test_wavelengths,
            crosssection_catalog_dataset_interpolated_to_model,
            list_of_species=test_species_list,
        )
    )

    test_rayleigh_crosssections.to_netcdf(
        test_data_structure_directory / "test_rayleigh_crosssections.nc"
    )
