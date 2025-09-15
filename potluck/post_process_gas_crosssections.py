import sys
from pathlib import Path

import xarray as xr

sys.path.append(str(Path(__file__).parent.parent))

from potluck.basic_types import NormalizedValue
from potluck.material.scattering.rayleigh import (
    calculate_H2_rayleigh_scattering_crosssections,
)
from potluck.xarray_functional_wrappers import convert_units

current_directory: Path = Path(__file__).parent


def merge_two_species(
    species_A_crosssection: xr.DataArray,
    species_B_crosssection: xr.DataArray,
    relative_molecular_fraction_of_A: NormalizedValue,
) -> xr.DataArray:
    relative_molecular_fraction_of_B: NormalizedValue = (
        1 - relative_molecular_fraction_of_A
    )

    return (
        relative_molecular_fraction_of_A * species_A_crosssection
        + relative_molecular_fraction_of_B * species_B_crosssection
    )


if __name__ == "__main__":
    gas_crosssection_filepath: Path = (
        "/home/Research/Opacities_0v10/gases/nirfs30k-2025-resampled.nc"
    )

    gas_crosssection_dataset: xr.Dataset = xr.open_dataset(gas_crosssection_filepath)

    crosssection_wavelengths_in_angstroms: xr.DataArray = convert_units(
        gas_crosssection_dataset.wavelength, {"wavelength": "angstroms"}
    )

    H2_rayleigh_scattering_crosssections: xr.DataArray = (
        calculate_H2_rayleigh_scattering_crosssections(
            crosssection_wavelengths_in_angstroms
        )
    )

    H2_crosssections_with_rayleigh_scattering: xr.DataArray = (
        gas_crosssection_dataset.h2only + H2_rayleigh_scattering_crosssections
    ).assign_attrs(notes="Rayleigh scattering added")

    gas_crosssection_dataset: xr.Dataset = gas_crosssection_dataset.assign(
        {"h2": H2_crosssections_with_rayleigh_scattering}
    )

    gas_crosssection_dataset: xr.Dataset = gas_crosssection_dataset.assign(
        {
            "h2he": merge_two_species(
                gas_crosssection_dataset.h2,
                gas_crosssection_dataset.he,
                relative_molecular_fraction_of_A=0.83,
            )
        }
    ).get(["h2", "he", "h2he", "h2o", "co", "co2", "ch4", "k", "nh3", "h2s"])

    gas_crosssection_dataset.to_netcdf(
        "/home/Research/Opacities_0v10/gases/nirfs30k_2025_Ross458c.nc"
    )
