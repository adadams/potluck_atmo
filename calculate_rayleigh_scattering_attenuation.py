from pathlib import Path
from typing import Final

import xarray as xr

from material.scattering.rayleigh import calculate_rayleigh_scattering_crosssections

MICRONS_TO_CM: Final[float] = 1e-4


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

    test_wavelengths: xr.DataArray = (
        crosssection_catalog_dataset_interpolated_to_model.wavelength
    ) * MICRONS_TO_CM

    test_rayleigh_crosssections: xr.Dataset = (
        calculate_rayleigh_scattering_crosssections(
            test_wavelengths, crosssection_catalog_dataset_interpolated_to_model
        )
        .rename("rayleigh_scattering_crosssections")
        .assign_attrs(units="cm^2")
    )

    test_rayleigh_crosssections.to_netcdf(
        test_data_structure_directory
        / f"test_rayleigh_crosssections_{test_opacity_catalog}.nc"
    )
