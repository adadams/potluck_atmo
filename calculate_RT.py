from pathlib import Path
from typing import Final

import msgspec
import xarray as xr

from material.absorbing.from_crosssections import (
    crosssections_to_attenutation_coefficients,
)
from material.scattering.rayleigh import calculate_two_stream_components
from material.scattering.types import TwoStreamScatteringCoefficients
from material.two_stream import compile_twostream_parameters
from material.types import TwoStreamParameters
from radiative_transfer.RT_Toon1989 import RT_Toon1989, RTToon1989Inputs

MICRONS_TO_CM: Final[float] = 1e-4


def compile_RT_inputs(
    wavelengths_in_cm: xr.DataArray,  # (wavelength,)
    temperatures_in_K: xr.DataArray,  # (pressure,)
    crosssections: xr.DataArray,  # (species, wavelength, pressure)
    number_density: xr.DataArray,  # (species, pressure)
    path_length: xr.DataArray,  # (pressure,)
) -> RTToon1989Inputs:
    scattering_coefficients: TwoStreamScatteringCoefficients = (
        calculate_two_stream_components(
            wavelengths_in_cm,
            crosssections,
            number_density,
        )
    )

    (
        cumulative_forward_scattering_coefficients,
        cumulative_backward_scattering_coefficients,
    ) = TwoStreamScatteringCoefficients(
        forward_scattering_coefficients=scattering_coefficients.forward_scattering_coefficients.sum(
            "species"
        ),
        backward_scattering_coefficients=scattering_coefficients.backward_scattering_coefficients.sum(
            "species"
        ),
    )

    absorption_coefficients: xr.Dataset = crosssections_to_attenutation_coefficients(
        crosssections, number_density
    )
    cumulative_absorption_coefficients: xr.DataArray = absorption_coefficients.sum(
        "species"
    )

    twostream_parameters: TwoStreamParameters = compile_twostream_parameters(
        forward_scattering_coefficients=cumulative_forward_scattering_coefficients,
        backward_scattering_coefficients=cumulative_backward_scattering_coefficients,
        absorption_coefficients=cumulative_absorption_coefficients,
        path_length=path_length,
    )

    return RTToon1989Inputs(
        wavelengths_in_cm=wavelengths_in_cm,
        temperatures_in_K=temperatures_in_K,
        **msgspec.structs.asdict(twostream_parameters),
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

    test_wavelengths: xr.DataArray = (
        crosssection_catalog_dataset_interpolated_to_model.wavelength
    ) * MICRONS_TO_CM

    test_temperatures: xr.DataArray = test_vertical_structure_dataset.temperature

    test_number_densities: xr.DataArray = test_number_densities_dataset.to_array(
        dim="species", name="number_density"
    )

    test_path_length: xr.DataArray = test_vertical_structure_dataset.altitude

    RT_inputs: RTToon1989Inputs = compile_RT_inputs(
        wavelengths_in_cm=test_wavelengths,
        temperatures_in_K=test_temperatures,
        crosssections=crosssection_catalog_dataset_interpolated_to_model,
        number_density=test_number_densities,
        path_length=test_path_length,
    )
