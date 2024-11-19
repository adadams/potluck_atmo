from pathlib import Path
from typing import Final

import msgspec
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from spectres import spectres

from material.absorbing.from_crosssections import (
    crosssections_to_attenutation_coefficients,
)
from material.scattering.rayleigh import calculate_two_stream_components
from material.scattering.types import TwoStreamScatteringCoefficients
from material.two_stream import compile_two_stream_parameters
from material.types import TwoStreamParameters
from radiative_transfer.RT_one_stream import calculate_spectral_intensity_at_surface
from radiative_transfer.RT_Toon1989 import RT_Toon1989, RTToon1989Inputs
from temperature.thermal_intensity import calculate_thermal_intensity_by_layer

PARSEC_TO_CM: Final[float] = 3.08567758128e18
MICRONS_TO_CM: Final[float] = 1e-4


def compile_RT_inputs(
    wavelengths_in_cm: xr.DataArray,  # (wavelength,)
    thermal_intensity: xr.DataArray,  # (pressure, wavelength)
    delta_thermal_intensity: xr.DataArray,  # (pressure, wavelength)
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

    cumulative_absorption_coefficients: xr.DataArray = (
        absorption_coefficients.sum("species")
        + cumulative_forward_scattering_coefficients
        + cumulative_backward_scattering_coefficients
    )

    two_stream_parameters: TwoStreamParameters = compile_two_stream_parameters(
        cumulative_forward_scattering_coefficients,
        cumulative_backward_scattering_coefficients,
        cumulative_absorption_coefficients,
        path_length,
    )

    return RTToon1989Inputs(
        thermal_intensity=thermal_intensity,
        delta_thermal_intensity=delta_thermal_intensity,
        **msgspec.structs.asdict(two_stream_parameters),
    )


if __name__ == "__main__":
    test_opacity_catalog: str = "jwst50k"
    test_species: str = "H2O"

    current_directory: Path = Path(__file__).parent
    opacity_data_directory: Path = (
        Path(
            "/Volumes"
            # current_directory / "molecular_crosssections" / "reference_data"
        )
        / "ResearchStorage"
        / "Opacities_0v10"
        / "gases"
    )

    catalog_filepath: Path = opacity_data_directory / f"{test_opacity_catalog}.nc"

    crosssection_catalog_dataset: xr.Dataset = xr.open_dataset(catalog_filepath)

    crosssection_catalog_dataset["wavelength"] = crosssection_catalog_dataset.wavelength

    test_data_structure_directory: Path = (
        current_directory / "test_inputs" / "test_data_structures"
    )

    test_data_filepath: Path = (
        test_data_structure_directory / "2M2236b_NIRSpec_G395H_R500_APOLLO.nc"
    )

    test_data: xr.Dataset = xr.open_dataset(test_data_filepath)

    test_data_wavelengths: xr.DataArray = test_data.wavelength

    test_vertical_structure_dataset_path: Path = (
        test_data_structure_directory / "test_vertical_structure.nc"
    )

    test_vertical_inputs_datatree: xr.DataTree = xr.open_datatree(
        test_vertical_structure_dataset_path
    )

    test_vertical_structure_dataset: xr.Dataset = test_vertical_inputs_datatree[
        "vertical_structure"
    ].to_dataset()

    test_planet_radius_in_cm: float = test_vertical_structure_dataset.attrs[
        "planet_radius_in_cm"
    ]

    test_distance_in_cm: float = 63.0 * PARSEC_TO_CM

    test_path_length: xr.DataArray = -test_vertical_structure_dataset.altitude.diff(
        "pressure"
    )

    temperatures_by_level: xr.DataArray = test_vertical_structure_dataset.temperature

    midlayer_pressures: xr.DataArray = (
        test_vertical_structure_dataset.pressure.to_numpy()[1:]
        + test_vertical_structure_dataset.pressure.to_numpy()[:-1]
    ) / 2

    test_vertical_structure_dataset: xr.Dataset = (
        test_vertical_structure_dataset.interp(pressure=midlayer_pressures)
    )

    test_path_length = test_path_length.assign_coords(
        pressure=test_vertical_structure_dataset.pressure
    )

    test_molecular_inputs_datatree: xr.DataTree = test_vertical_inputs_datatree[
        "molecular_inputs"
    ]
    test_number_densities_dataset: xr.Dataset = (
        test_molecular_inputs_datatree["number_densities"]
        .to_dataset()
        .interp(pressure=midlayer_pressures)
    )

    crosssection_catalog_dataset_interpolated_to_model: xr.Dataset = (
        (
            crosssection_catalog_dataset.interp(
                temperature=test_vertical_structure_dataset.temperature,
                pressure=midlayer_pressures,
            )
        )
        .get(list(test_number_densities_dataset.data_vars))
        .to_array(dim="species", name="crosssections")
    )

    test_wavelengths: xr.DataArray = (
        crosssection_catalog_dataset_interpolated_to_model.wavelength
    )

    test_wavelengths_in_cm: xr.DataArray = test_wavelengths * MICRONS_TO_CM

    temperature_grid, wavelength_grid = np.meshgrid(
        temperatures_by_level, test_wavelengths_in_cm
    )

    test_thermal_intensity, test_delta_thermal_intensity = (
        calculate_thermal_intensity_by_layer(wavelength_grid, temperature_grid)
    )

    test_thermal_intensity = xr.DataArray(
        data=test_thermal_intensity,
        dims=["wavelength", "pressure"],
        coords={"pressure": midlayer_pressures, "wavelength": test_wavelengths},
        name="thermal_intensity",
        attrs={"units": "erg s^-1 cm^-3 sr^-1"},
    )

    test_delta_thermal_intensity = xr.DataArray(
        data=test_delta_thermal_intensity,
        dims=["wavelength", "pressure"],
        coords={"pressure": midlayer_pressures, "wavelength": test_wavelengths},
        name="delta_thermal_intensity",
        attrs={"units": "erg s^-1 cm^-3 sr^-1"},
    )

    test_temperatures: xr.DataArray = test_vertical_structure_dataset.temperature

    test_number_densities: xr.DataArray = test_number_densities_dataset.to_array(
        dim="species", name="number_density"
    )

    RT_inputs: RTToon1989Inputs = compile_RT_inputs(
        wavelengths_in_cm=test_wavelengths_in_cm,
        thermal_intensity=test_thermal_intensity,
        delta_thermal_intensity=test_delta_thermal_intensity,
        crosssections=crosssection_catalog_dataset_interpolated_to_model,
        number_density=test_number_densities,
        path_length=test_path_length,
    )

    RT_inputs["optical_depth"].to_netcdf(
        test_data_structure_directory / "test_RT_inputs_optical_depth.nc"
    )

    emitted_flux: xr.DataArray = (
        RT_Toon1989(*RT_inputs.values())
        .rename("emitted_flux")
        .assign_attrs(units="erg s^-1 cm^-3")
    )

    cumulative_optical_depth: xr.DataArray = (
        RT_inputs["optical_depth"].cumulative("pressure").sum()
    )

    emitted_onestream_flux: xr.DataArray = (
        calculate_spectral_intensity_at_surface(
            test_thermal_intensity,
            cumulative_optical_depth,
        )
        .rename("emitted_flux")
        .assign_attrs(units="erg s^-1 cm^-3")
    )

    observed_onestream_flux: xr.DataArray = (
        emitted_onestream_flux * (test_planet_radius_in_cm / test_distance_in_cm) ** 2
    ).rename("observed_flux")

    observed_twostream_flux: xr.DataArray = (
        emitted_flux * (test_planet_radius_in_cm / test_distance_in_cm) ** 2
    )

    resampled_onestream_flux: xr.DataArray = xr.apply_ufunc(
        spectres,
        test_data_wavelengths,
        observed_onestream_flux.wavelength,
        observed_onestream_flux,
        input_core_dims=[["wavelength"], ["wavelength"], ["wavelength"]],
        output_core_dims=[["wavelength"]],
        exclude_dims=set(("wavelength",)),
        keep_attrs=True,
    ).assign_coords(wavelength=test_data_wavelengths)

    resampled_twostream_flux: xr.DataArray = xr.apply_ufunc(
        spectres,
        test_data_wavelengths,
        observed_twostream_flux.wavelength,
        observed_twostream_flux,
        input_core_dims=[["wavelength"], ["wavelength"], ["wavelength"]],
        output_core_dims=[["wavelength"]],
        exclude_dims=set(("wavelength",)),
        keep_attrs=True,
    )

    test_apollo_model_spectrum_filepath: Path = (
        current_directory
        / "test_inputs"
        / "old_files_for_comparison"
        / (
            "2M2236.Piette.G395H.cloud-free.2024-02-27.continuation.retrieved.Spectrum.binned.dat"
        )
    )

    test_apollo_model_spectral_output = np.loadtxt(
        test_apollo_model_spectrum_filepath
    ).T
    tamso_wavelo, tamso_wavehi, tamso_flux, *_ = test_apollo_model_spectral_output
    tamso_wave = (tamso_wavelo + tamso_wavehi) / 2

    flux_factor = np.where(test_data_wavelengths.to_numpy() < 4.10, 1.133373, 0.904548)

    plt.plot(
        test_data_wavelengths, resampled_onestream_flux * flux_factor, label="onestream"
    )
    plt.plot(test_data_wavelengths, test_data.flux_lambda, label="test 2M2236b data")
    plt.plot(
        test_data_wavelengths, resampled_twostream_flux * flux_factor, label="twostream"
    )
    plt.plot(tamso_wave, tamso_flux, label="apollo model")
    plt.legend()
    plt.savefig(
        test_data_structure_directory / "test_RT_comparison.pdf", bbox_inches="tight"
    )

    resampled_onestream_flux.to_netcdf(
        test_data_structure_directory / "test_observed_onestream_flux.nc"
    )

    resampled_twostream_flux.to_netcdf(
        test_data_structure_directory / "test_observed_twostream_flux.nc"
    )
