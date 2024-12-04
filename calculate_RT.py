from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from compile_crosssection_data import curate_crosssection_catalog
from compile_vertical_structure import compile_vertical_structure_for_forward_model
from constants_and_conversions import MICRONS_TO_CM
from material.two_stream import compile_composite_two_stream_parameters
from material.types import TwoStreamParameters
from radiative_transfer.RT_one_stream import calculate_spectral_intensity_at_surface
from radiative_transfer.RT_Toon1989 import RT_Toon1989, RTToon1989Inputs
from temperature.thermal_intensity import calculate_thermal_intensity_by_layer
from user.input_importers import (
    UserForwardModelInputsPlusStuff,
    import_user_forward_model_inputs_plus_stuff,
)
from vertical.altitude import convert_dataset_by_pressure_levels_to_pressure_layers
from wavelength import resample_spectral_quantity_to_new_wavelengths
from xarray_functional_wrappers import rename_and_unitize

current_directory: Path = Path(__file__).parent


def convert_surface_quantity_to_observed_quantity(
    radius_in_solar_radii: float,
    distance_in_parsecs: float,
):
    distance_in_solar_radii: float = distance_in_parsecs * 4.435e7

    return (radius_in_solar_radii / distance_in_solar_radii) ** 2 / (4 * np.pi)


if __name__ == "__main__":
    model_directory_label: str = "test_Phillips"

    current_directory: Path = Path(__file__).parent
    user_directory: Path = current_directory / "user"
    model_directory: Path = user_directory / f"{model_directory_label}_model"
    intermediate_output_directory: Path = model_directory / "intermediate_outputs"
    output_file_directory: Path = model_directory / "output_files"

    user_model_inputs: UserForwardModelInputsPlusStuff = (
        import_user_forward_model_inputs_plus_stuff(
            model_directory_label=model_directory_label, parent_directory="user"
        )
    )

    reference_model: xr.Dataset = xr.open_dataset(
        user_model_inputs.reference_model_filepath
    )
    reference_model_wavelengths: xr.DataArray = reference_model.wavelength
    reference_model_flux_lambda: xr.DataArray = reference_model.flux_lambda

    compiled_vertical_dataset: xr.Dataset = compile_vertical_structure_for_forward_model(
        user_vertical_inputs=user_model_inputs.user_forward_model_inputs.vertical_inputs
    )

    vertical_inputs_datatree: xr.DataTree = xr.open_datatree(
        user_model_inputs.vertical_structure_datatree_path
    )

    vertical_structure_dataset: xr.Dataset = vertical_inputs_datatree[
        "vertical_structure"
    ].to_dataset()

    vertical_structure_dataset_by_layer: xr.Dataset = (
        convert_dataset_by_pressure_levels_to_pressure_layers(
            vertical_structure_dataset
        )
    )

    temperatures_by_layer: xr.DataArray = (
        vertical_structure_dataset_by_layer.temperature
    )
    pressures_by_layer: xr.DataArray = vertical_structure_dataset_by_layer.pressure

    molecular_inputs_datatree: xr.DataTree = vertical_inputs_datatree[
        "molecular_inputs"
    ]

    number_densities_by_level: xr.Dataset = molecular_inputs_datatree[
        "number_densities"
    ].to_dataset()

    number_densities_by_layer: xr.Dataset = (
        convert_dataset_by_pressure_levels_to_pressure_layers(number_densities_by_level)
    )

    species_present_in_model: list[str] = list(number_densities_by_layer.data_vars)

    crosssection_catalog_dataset_interpolated_to_model: xr.Dataset = curate_crosssection_catalog(
        crosssection_catalog_dataset=user_model_inputs.user_forward_model_inputs.crosssection_catalog,
        temperatures_by_layer=temperatures_by_layer,
        pressures_by_layer=pressures_by_layer,
        species_present_in_model=species_present_in_model,
    )

    model_wavelengths_in_microns: xr.DataArray = (
        crosssection_catalog_dataset_interpolated_to_model.wavelength
    )
    model_wavelengths_in_cm = model_wavelengths_in_microns * MICRONS_TO_CM

    temperature_grid, wavelength_grid = np.meshgrid(
        user_model_inputs.user_forward_model_inputs.vertical_inputs.temperatures_by_level,
        model_wavelengths_in_cm,
    )

    thermal_intensity, delta_thermal_intensity = calculate_thermal_intensity_by_layer(
        wavelength_grid, temperature_grid
    )

    shared_thermal_intensity_kwargs: dict[str, Any] = {
        "dims": ("wavelength", "pressure"),
        "coords": {
            "pressure": vertical_structure_dataset_by_layer.pressure,
            "wavelength": model_wavelengths_in_microns,
        },
        "attrs": {"units": "erg s^-1 cm^-3 sr^-1"},
    }

    thermal_intensity = xr.DataArray(
        data=thermal_intensity,
        name="thermal_intensity",
        **shared_thermal_intensity_kwargs,
    )

    delta_thermal_intensity = xr.DataArray(
        data=delta_thermal_intensity,
        name="delta_thermal_intensity",
        **shared_thermal_intensity_kwargs,
    )

    number_densities: xr.DataArray = number_densities_by_layer.to_array(
        dim="species", name="number_density"
    )

    two_stream_parameters: TwoStreamParameters = compile_composite_two_stream_parameters(
        wavelengths_in_cm=model_wavelengths_in_cm,
        crosssections=crosssection_catalog_dataset_interpolated_to_model,
        number_density=number_densities,
        path_lengths=user_model_inputs.user_forward_model_inputs.path_lengths_by_level,
    )

    RT_Toon1989_inputs: RTToon1989Inputs = RTToon1989Inputs(
        thermal_intensity=thermal_intensity,
        delta_thermal_intensity=delta_thermal_intensity,
        **msgspec.structs.asdict(two_stream_parameters),
    )

    RT_Toon1989_inputs.optical_depth.to_netcdf(
        output_file_directory / "test_RT_inputs_optical_depth.nc"
    )

    emitted_twostream_flux: xr.DataArray = rename_and_unitize(
        RT_Toon1989(*RT_Toon1989_inputs[:-2]),
        name="emitted_flux",
        units="erg s^-1 cm^-3",
    )

    cumulative_optical_depth: xr.DataArray = (
        RT_Toon1989_inputs.optical_depth.cumulative("pressure").sum()
    )

    emitted_onestream_flux: xr.DataArray = rename_and_unitize(
        calculate_spectral_intensity_at_surface(
            thermal_intensity,
            cumulative_optical_depth,
        ),
        name="emitted_flux",
        units="erg s^-1 cm^-3",
    )

    observed_onestream_flux: xr.DataArray = (
        emitted_onestream_flux
        * (
            user_model_inputs.user_forward_model_inputs.vertical_inputs.planet_radius_in_cm
            / user_model_inputs.user_forward_model_inputs.distance_to_system_in_cm
        )
        ** 2
    ).rename("observed_flux")

    observed_twostream_flux: xr.DataArray = (
        emitted_twostream_flux
        * (
            user_model_inputs.user_forward_model_inputs.vertical_inputs.planet_radius_in_cm
            / user_model_inputs.user_forward_model_inputs.distance_to_system_in_cm
        )
        ** 2
    ).rename("observed_flux")

    resampled_onestream_flux: xr.DataArray = (
        resample_spectral_quantity_to_new_wavelengths(
            reference_model_wavelengths,
            observed_onestream_flux.wavelength,
            observed_onestream_flux,
        )
    )

    resampled_twostream_flux: xr.DataArray = (
        resample_spectral_quantity_to_new_wavelengths(
            reference_model_wavelengths,
            observed_twostream_flux.wavelength,
            observed_twostream_flux,
        )
    )

    apollo_model_spectrum_filepath: Path = (
        current_directory
        / "user"
        / "old_files_for_comparison"
        / (
            "2M2236.Piette.G395H.cloud-free.2024-02-27.continuation.retrieved.Spectrum.binned.dat"
        )
    )

    apollo_model_spectral_output = np.loadtxt(apollo_model_spectrum_filepath).T
    tamso_wavelo, tamso_wavehi, tamso_flux, *_ = apollo_model_spectral_output
    tamso_wave = (tamso_wavelo + tamso_wavehi) / 2

    # flux_factor = np.where(data_wavelengths.to_numpy() < 4.10, 1.133373, 0.904548)

    control_picaso_dataset: xr.Dataset = xr.load_dataset(
        model_directory / "PICASO_model_control.nc"
    )

    control_picaso_dataset["flux_lambda"] = (
        control_picaso_dataset.flux_emission
        * convert_surface_quantity_to_observed_quantity(
            radius_in_solar_radii=0.116, distance_in_parsecs=64.5
        )
        * 10
    )

    # plt.plot(data_wavelengths, resampled_onestream_flux, label="onestream")
    # plt.plot(data_wavelengths, resampled_onestream_flux * flux_factor, label="onestream")
    plt.plot(
        reference_model_wavelengths,
        reference_model_flux_lambda * 1e9,
        label="Phillips model data",
    )
    plt.plot(reference_model_wavelengths, resampled_twostream_flux, label="twostream")
    plt.plot(
        control_picaso_dataset.wavelength,
        control_picaso_dataset.flux_lambda,
        label="PICASO model",
    )
    # plt.plot(data_wavelengths, resampled_twostream_flux * flux_factor, label="twostream")
    # plt.plot(tamso_wave, tamso_flux, label="apollo model")
    plt.legend()
    plt.savefig(
        output_file_directory / "Phillips_RT_comparison.pdf",
        bbox_inches="tight",
    )

    resampled_onestream_flux.to_netcdf(
        output_file_directory / "Phillips_observed_onestream_flux.nc"
    )

    resampled_twostream_flux.to_netcdf(
        output_file_directory / "Phillips_observed_twostream_flux.nc"
    )
