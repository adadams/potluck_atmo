from pathlib import Path

import xarray as xr

from material.scattering.rayleigh import calculate_rayleigh_scattering_crosssection

if __name__ == "__main__":
    test_opacity_catalog: str = "nir"
    test_species: str = "H2O"

    opacity_data_directory: Path = (
        Path(__file__).parent.parent.parent
        / "molecular_crosssections"
        / "reference_data"
    )

    catalog_filepath: Path = opacity_data_directory / f"{test_opacity_catalog}.nc"

    catalog_dataset: xr.Dataset = xr.open_dataset(catalog_filepath)

    print(f"{catalog_dataset[test_species.lower()]}")
