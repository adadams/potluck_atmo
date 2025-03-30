import sys
from collections.abc import Mapping
from glob import glob
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import xarray as xr

from material.gases.reference_data.picaso_lupu_temperature_levels import (
    PICASO_TP_LEVELS,
)
from material.gases.sanity_checks import check_if_all_headers_match
from spectrum.wavelength import (
    get_number_of_wavelengths,
    get_wavelengths_from_number_of_elements_and_resolution,
)

current_directory: Path = Path(__file__).parent


class CrossSectionTableHeader(msgspec.Struct):
    """
    Defines the header entries and properties that are needed for a collection
    of cross-section tables. Attributes follow the APOLLO-style header.
    """

    number_of_pressure_layers: int
    minimum_log_pressure: float
    maximum_log_pressure: float
    number_of_temperatures: int
    minimum_log_temperature: float
    maximum_log_temperature: float
    number_of_spectral_elements: int
    minimum_wavelength: float
    maximum_wavelength: float
    effective_resolution: float


expected_number_of_header_entries: int = len(CrossSectionTableHeader.__struct_fields__)


class MolecularCrossSectionMetadata(msgspec.Struct):
    scaopac: float
    mmw: float
    units: str = "cm^2"


molecular_metadata_filepath: Path = (
    current_directory / "material" / "gases" / "reference_data" / "gas_properties.toml"
)

molecular_metadata: dict[str, MolecularCrossSectionMetadata] = {
    species: MolecularCrossSectionMetadata(**molecular_original_metadata)
    for species, molecular_original_metadata in msgspec.toml.decode(
        molecular_metadata_filepath.read_text()
    ).items()
}


def get_file_headers(filepaths: Mapping[str, Path]) -> tuple[tuple[Any, ...], ...]:
    file_headers = {}
    for species, filepath in filepaths.items():
        with open(filepath, "r") as file:
            header_entries = file.readline().split()
            if len(header_entries) != expected_number_of_header_entries:
                print(
                    f"Header for {species} does not have {expected_number_of_header_entries} entries. Skipping."
                )
                continue

            header = tuple(
                [
                    float(entry) if "." in entry else int(entry)
                    for entry in header_entries
                ]
            )
        file_headers[species] = CrossSectionTableHeader(*header)

    return file_headers


def load_crosssections_into_array(
    number_of_pressure_layers: int,
    number_of_temperatures: int,
    number_of_spectral_elements: int,
    filepaths: dict[str, Path],
    excluded_species: list[str] = None,
    minimum_value: float = 1e-50,
    maximum_value: float = None,
):
    if excluded_species is None:
        excluded_species = []

    def load_array(
        filepath,
        loadtxt_kwargs=dict(skiprows=1),
        minimum_value: float = minimum_value,
        maximum_value: float = maximum_value,
    ):
        original_array = np.loadtxt(filepath, **loadtxt_kwargs)
        print(f"{filepath=}, {original_array.shape=}")

        return (
            original_array.clip(minimum_value, maximum_value)
            .reshape(
                number_of_temperatures,
                number_of_pressure_layers,
                number_of_spectral_elements,
            )
            .swapaxes(0, 1)
        )  # P T W is the ORIGINAL

    return {
        species: load_array(filepath)
        for species, filepath in filepaths.items()
        if species not in excluded_species
    }


if __name__ == "__main__":
    research_data_storage_directory: Path = Path("/Volumes/Orange")
    # research_data_storage_directory: Path = Path("/home/Research")

    test_opacity_directory: Path = (
        research_data_storage_directory / "Opacities_0v10" / "gases"
    )

    if not Path.exists(test_opacity_directory):
        raise ValueError(f"Directory {test_opacity_directory} does not exist.")

    opacity_catalog_name: str = "nir"

    all_opacity_filepaths: dict[str, Path] = {
        filepath[filepath.rfind("/") + 1 : filepath.index(".")]: Path(filepath)
        for filepath in sorted(
            glob(f"{test_opacity_directory}/*.{opacity_catalog_name}.dat")
        )
    }

    opacity_file_header_dict: dict[str, CrossSectionTableHeader] = get_file_headers(
        all_opacity_filepaths
    )

    opacity_file_headers: tuple[CrossSectionTableHeader] = tuple(
        opacity_file_header_dict.values()
    )

    opacity_filepaths: dict[str, Path] = {
        species: filepath
        for species, filepath in all_opacity_filepaths.items()
        if species in opacity_file_header_dict
    }

    check_if_all_headers_match(opacity_file_headers)
    fiducial_opacity_file_header: CrossSectionTableHeader = opacity_file_headers[-1]
    print(f"{fiducial_opacity_file_header=}")

    log_pressures: np.ndarray = np.linspace(
        fiducial_opacity_file_header.minimum_log_pressure,
        fiducial_opacity_file_header.maximum_log_pressure,
        fiducial_opacity_file_header.number_of_pressure_layers,
    )
    print(f"{log_pressures=}")

    # log_temperatures: np.ndarray = np.log10(PICASO_TP_LEVELS)
    log_temperatures: np.ndarray = np.linspace(
        fiducial_opacity_file_header.minimum_log_temperature,
        fiducial_opacity_file_header.maximum_log_temperature,
        fiducial_opacity_file_header.number_of_temperatures,
    )
    print(f"{10**log_temperatures=}")

    deltalogt = (fiducial_opacity_file_header.number_of_temperatures - 1) / (
        fiducial_opacity_file_header.maximum_log_temperature
        - fiducial_opacity_file_header.minimum_log_temperature
    )
    temps = 10**fiducial_opacity_file_header.minimum_log_temperature * np.power(
        10,
        np.arange(0, fiducial_opacity_file_header.number_of_temperatures) / deltalogt,
    )
    print(f"{temps=}")

    # for(int n=0; n<ntemp; n++){
    # logtemp[n] = log10(tmin*pow(10,n/deltalogt));
    # }

    number_of_wavelengths: int = get_number_of_wavelengths(
        fiducial_opacity_file_header.minimum_wavelength,
        fiducial_opacity_file_header.maximum_wavelength,
        fiducial_opacity_file_header.effective_resolution,
    )
    wavelengths: np.ndarray = get_wavelengths_from_number_of_elements_and_resolution(
        fiducial_opacity_file_header.minimum_wavelength,
        number_of_wavelengths,
        fiducial_opacity_file_header.effective_resolution,
    )
    sys.exit()

    opacity_data: dict[str, np.ndarray] = load_crosssections_into_array(
        fiducial_opacity_file_header.number_of_pressure_layers,
        fiducial_opacity_file_header.number_of_temperatures,
        fiducial_opacity_file_header.number_of_spectral_elements,
        opacity_filepaths,
    )

    shared_coordinates: dict[str, np.ndarray] = {
        "pressure": xr.Variable("pressure", 10**log_pressures, attrs={"units": "bar"}),
        "temperature": xr.Variable(
            "temperature", 10**log_temperatures, attrs={"units": "kelvin"}
        ),
        "wavelength": xr.Variable(
            "wavelength", wavelengths, attrs={"units": "microns"}
        ),
    }

    species_dataarrays: dict[str, xr.DataArray] = {
        species: xr.DataArray(
            dims=tuple(shared_coordinates.keys()),
            coords=shared_coordinates,
            data=crosssection_array,
            attrs=msgspec.structs.asdict(molecular_metadata[species])
            if species in molecular_metadata
            else {},
        )
        for species, crosssection_array in opacity_data.items()
    }

    species_dataset: xr.Dataset = xr.Dataset(
        data_vars=species_dataarrays,
        coords=shared_coordinates,
        attrs={"opacity_catalog": opacity_catalog_name},
    )

    species_dataset.to_netcdf(
        test_opacity_directory / f"{opacity_catalog_name}_test.nc"
    )
