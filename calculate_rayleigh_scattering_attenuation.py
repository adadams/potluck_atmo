from pathlib import Path

import xarray as xr

from material.scattering.rayleigh import calculate_rayleigh_scattering_crosssection

if __name__ == "__main__":
    test_opacity_catalog: str = "nir"
    test_species: str = "H2O"

    current_directory: Path = Path(__file__).parent
    opacity_data_directory: Path = (
        current_directory / "molecular_crosssections" / "reference_data"
    )

    catalog_filepath: Path = opacity_data_directory / f"{test_opacity_catalog}.nc"

    catalog_dataset: xr.Dataset = xr.open_dataset(catalog_filepath)

    print(f"{catalog_dataset[test_species.lower()]}")

    test_vertical_structure_dataset_path: Path = (
        current_directory
        / "test_inputs"
        / "test_data_structures"
        / "test_vertical_structure.nc"
    )

    test_number_density_dataset: xr.Dataset = xr.open_datatree(
        test_vertical_structure_dataset_path
    )["molecular_inputs"]["number_densities"]

    print(f"{test_number_density_dataset.data_vars=}")
