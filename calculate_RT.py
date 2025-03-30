from pathlib import Path
from typing import Final

import msgspec
import xarray as xr

from compile_crosssection_data import curate_crosssection_catalog
from compile_thermal_structure import compile_thermal_structure_for_forward_model
from compile_vertical_structure import (
    ForwardModelXarrayInputs,
    compile_vertical_structure_for_forward_model,
)
from constants_and_conversions import MICRONS_TO_CM
from material.two_stream import compile_composite_two_stream_parameters
from material.types import TwoStreamParameters
from radiative_transfer.RT_one_stream import calculate_spectral_intensity_at_surface
from radiative_transfer.RT_Toon1989 import RT_Toon1989, RTToon1989Inputs
from spectrum.bin import resample_spectral_quantity_to_new_wavelengths
from user.input_importers import UserForwardModelInputsPlusStuff
from xarray_functional_wrappers import (
    XarrayOutputs,
    rename_and_unitize,
    save_xarray_outputs_to_file,
)

current_directory: Path = Path(__file__).parent

METERS_TO_CENTIMETERS: Final[float] = 1e2


def calculate_observed_fluxes(
    user_model_inputs: UserForwardModelInputsPlusStuff,
) -> XarrayOutputs:
    compiled_vertical_dataset: ForwardModelXarrayInputs = compile_vertical_structure_for_forward_model(
        user_vertical_inputs=user_model_inputs.user_forward_model_inputs.vertical_inputs
    )
    vertical_structure_dataset_by_level: xr.Dataset = compiled_vertical_dataset.by_level
    vertical_structure_dataset_by_layer: xr.Dataset = compiled_vertical_dataset.by_layer

    temperatures_by_level: xr.DataArray = (
        vertical_structure_dataset_by_level.temperature
    )
    temperatures_by_layer: xr.DataArray = (
        vertical_structure_dataset_by_layer.temperature
    )
    pressures_by_layer: xr.DataArray = vertical_structure_dataset_by_layer.pressure

    number_densities_by_layer: xr.DataArray = vertical_structure_dataset_by_layer[
        "number_density"
    ]

    species_present_in_model: list[str] = number_densities_by_layer.species.values

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

    thermal_intensities_by_layer: xr.Dataset = (
        compile_thermal_structure_for_forward_model(
            temperatures_by_level=temperatures_by_level,
            pressures_by_layer=pressures_by_layer,
            model_wavelengths_in_microns=model_wavelengths_in_microns,
        )
    )

    path_lengths_in_cm: xr.DataArray = (
        user_model_inputs.user_forward_model_inputs.path_lengths_by_level
        * METERS_TO_CENTIMETERS
    )

    two_stream_parameters: TwoStreamParameters = (
        compile_composite_two_stream_parameters(
            wavelengths_in_cm=model_wavelengths_in_cm,
            crosssections=crosssection_catalog_dataset_interpolated_to_model,
            number_density=number_densities_by_layer,
            path_lengths=path_lengths_in_cm,
        )
    )

    RT_Toon1989_inputs: RTToon1989Inputs = RTToon1989Inputs(
        thermal_intensity=thermal_intensities_by_layer.thermal_intensity,
        delta_thermal_intensity=thermal_intensities_by_layer.delta_thermal_intensity,
        **msgspec.structs.asdict(two_stream_parameters),
    )

    RT_Toon1989_inputs.optical_depth.rename("optical_depth").to_netcdf(
        current_directory / "test_Wills_optical_depth.nc"
    )

    emitted_twostream_flux: xr.DataArray = rename_and_unitize(
        RT_Toon1989(*RT_Toon1989_inputs[:-2]),
        name="emitted_flux",
        units="erg s^-1 cm^-3",
    )

    cumulative_optical_depth: xr.DataArray = (
        RT_Toon1989_inputs.optical_depth.cumulative("pressure").sum()
    )

    cumulative_optical_depth.rename("optical_depth").to_netcdf(
        current_directory / "test_RT_inputs_optical_depth.nc"
    )

    emitted_onestream_flux: xr.DataArray = rename_and_unitize(
        calculate_spectral_intensity_at_surface(
            thermal_intensities_by_layer.thermal_intensity,
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

    return {
        "observed_onestream_flux": observed_onestream_flux,
        "observed_twostream_flux": observed_twostream_flux,
    }


@save_xarray_outputs_to_file
def resample_observed_fluxes(
    observed_fluxes: dict[str, xr.DataArray],
    reference_model_wavelengths: xr.DataArray,
) -> dict[str, xr.DataArray]:
    return {
        observed_flux_name.replace("observed", "resampled"): rename_and_unitize(
            resample_spectral_quantity_to_new_wavelengths(
                reference_model_wavelengths,
                observed_flux.wavelength,
                observed_flux,
            ),
            name="resampled_observed_flux",
            units="erg s^-1 cm^-3",
        )
        for observed_flux_name, observed_flux in observed_fluxes.items()
    }
