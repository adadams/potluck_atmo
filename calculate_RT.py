from dataclasses import asdict, astuple
from pathlib import Path
from typing import Optional

import xarray as xr

from compile_crosssection_data import curate_crosssection_catalog
from compile_thermal_structure import compile_thermal_structure_for_forward_model
from compile_vertical_structure import (
    ForwardModelXarrayInputs,
    compile_vertical_structure_for_forward_model,
)
from constants_and_conversions import MICRONS_TO_CM
from material.absorbing.from_crosssections import (
    attenuation_coefficients_to_optical_depths,
    crosssections_to_attenuation_coefficients,
)
from material.two_stream import compile_composite_two_stream_parameters
from material.types import TwoStreamParameters
from radiative_transfer.RT_in_transmission import calculate_transmission_spectrum
from radiative_transfer.RT_one_stream import (
    OneStreamRTInputs,
    calculate_spectral_intensity_at_surface,
)
from radiative_transfer.RT_Toon1989_jax import RT_Toon1989, RTToon1989Inputs
from spectrum.bin import resample_spectral_quantity_to_new_wavelengths
from user.input_structs import UserForwardModelInputs
from xarray_functional_wrappers import XarrayOutputs, convert_units

current_directory: Path = Path(__file__).parent


def calculate_observed_transmission_spectrum(
    user_forward_model_inputs: UserForwardModelInputs,
    precalculated_crosssection_catalog: Optional[xr.Dataset] = None,
) -> XarrayOutputs:
    compiled_vertical_dataset: ForwardModelXarrayInputs = (
        compile_vertical_structure_for_forward_model(
            user_vertical_inputs=user_forward_model_inputs.vertical_inputs
        )
    )
    vertical_structure_dataset_by_layer: xr.Dataset = compiled_vertical_dataset.by_layer

    number_densities_by_layer: xr.DataArray = (
        vertical_structure_dataset_by_layer.number_density
    )

    species_present_in_model: list[str] = number_densities_by_layer.species.values

    if precalculated_crosssection_catalog is None:
        crosssection_catalog_dataset_interpolated_to_model: xr.Dataset = curate_crosssection_catalog(
            crosssection_catalog_dataset=user_forward_model_inputs.crosssection_catalog,
            temperatures_by_layer=vertical_structure_dataset_by_layer.temperature,
            pressures_by_layer=vertical_structure_dataset_by_layer.pressure,
            species_present_in_model=species_present_in_model,
        )
    else:
        crosssection_catalog_dataset_interpolated_to_model = (
            precalculated_crosssection_catalog
        )

    absorption_coefficients: xr.Dataset = crosssections_to_attenuation_coefficients(
        crosssection_catalog_dataset_interpolated_to_model,
        vertical_structure_dataset_by_layer.number_density,
    )

    optical_depth: xr.DataArray = attenuation_coefficients_to_optical_depths(
        absorption_coefficients,
        user_forward_model_inputs.path_lengths_by_layer,
    )

    cumulative_optical_depth_by_layer: xr.DataArray = (
        optical_depth.cumulative("pressure").sum().sum("species")
    )

    transmission_flux: xr.DataArray = calculate_transmission_spectrum(
        cumulative_optical_depth_by_layer,
        user_forward_model_inputs.path_lengths_by_layer,
        user_forward_model_inputs.altitudes_by_layer,
        user_forward_model_inputs.stellar_radius_in_cm,
        user_forward_model_inputs.vertical_inputs.planet_radius_in_cm,
    )
    return transmission_flux


def calculate_observed_fluxes(
    user_forward_model_inputs: UserForwardModelInputs,
    precalculated_crosssection_catalog: Optional[xr.Dataset] = None,
) -> XarrayOutputs:
    compiled_vertical_dataset: ForwardModelXarrayInputs = (
        compile_vertical_structure_for_forward_model(
            user_vertical_inputs=user_forward_model_inputs.vertical_inputs
        )
    )
    vertical_structure_dataset_by_level: xr.Dataset = compiled_vertical_dataset.by_level
    vertical_structure_dataset_by_layer: xr.Dataset = compiled_vertical_dataset.by_layer

    temperatures_by_level: xr.DataArray = (
        vertical_structure_dataset_by_level.temperature
    )

    pressures_by_layer: xr.DataArray = vertical_structure_dataset_by_layer.pressure

    number_densities_by_layer: xr.DataArray = (
        vertical_structure_dataset_by_layer.number_density
    )

    if precalculated_crosssection_catalog is None:
        temperatures_by_layer: xr.DataArray = (
            vertical_structure_dataset_by_layer.temperature
        )

        species_present_in_model: list[str] = number_densities_by_layer.species.values

        crosssection_catalog_dataset_interpolated_to_model: xr.Dataset = curate_crosssection_catalog(
            crosssection_catalog_dataset=user_forward_model_inputs.crosssection_catalog,
            temperatures_by_layer=temperatures_by_layer,
            pressures_by_layer=pressures_by_layer,
            species_present_in_model=species_present_in_model,
        )

    else:
        crosssection_catalog_dataset_interpolated_to_model: xr.Dataset = (
            precalculated_crosssection_catalog
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

    path_lengths_in_cm: xr.DataArray = user_forward_model_inputs.path_lengths_by_layer

    altitudes_in_cm: xr.DataArray = user_forward_model_inputs.altitudes_by_layer

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
        **asdict(two_stream_parameters),
    )

    emitted_twostream_flux: xr.DataArray = RT_Toon1989(*astuple(RT_Toon1989_inputs))

    cumulative_optical_depth: xr.DataArray = (
        RT_Toon1989_inputs.optical_depth.cumulative("pressure").sum()
    )

    onestream_inputs: OneStreamRTInputs = OneStreamRTInputs(
        thermal_intensity=thermal_intensities_by_layer.thermal_intensity,
        cumulative_optical_depth_by_layer=cumulative_optical_depth,
    )

    emitted_onestream_flux: xr.DataArray = calculate_spectral_intensity_at_surface(
        *astuple(onestream_inputs)
    )

    transmission_flux: xr.DataArray = calculate_transmission_spectrum(
        cumulative_optical_depth,
        path_lengths_in_cm,
        altitudes_in_cm,
        user_forward_model_inputs.stellar_radius_in_cm,
        user_forward_model_inputs.vertical_inputs.planet_radius_in_cm,
    )

    observed_onestream_flux: xr.DataArray = (
        emitted_onestream_flux
        * (
            user_forward_model_inputs.vertical_inputs.planet_radius_in_cm
            / user_forward_model_inputs.distance_to_system_in_cm
        )
        ** 2
    ).rename("observed_onestream_flux")

    observed_twostream_flux: xr.DataArray = (
        emitted_twostream_flux
        * (
            user_forward_model_inputs.vertical_inputs.planet_radius_in_cm
            / user_forward_model_inputs.distance_to_system_in_cm
        )
        ** 2
    ).rename("observed_twostream_flux")

    return {
        "observed_onestream_flux": observed_onestream_flux,
        "observed_twostream_flux": observed_twostream_flux,
        "transmission_flux": transmission_flux,
    }


def calculate_observed_fluxes_via_two_stream(
    forward_model_inputs: xr.DataTree,
    precalculated_crosssection_catalog: Optional[xr.Dataset] = None,
) -> xr.DataArray:
    temperatures_by_level: xr.DataArray = forward_model_inputs[
        "temperature_profile_by_level"
    ].temperature

    atmospheric_structure_by_layer: xr.DataArray = forward_model_inputs[
        "atmospheric_structure_by_layer"
    ]
    vertical_structure_by_layer: xr.DataArray = atmospheric_structure_by_layer[
        "vertical_structure"
    ]

    pressures_by_layer: xr.DataArray = vertical_structure_by_layer.pressures_by_layer

    number_densities_by_layer: xr.DataArray = (
        atmospheric_structure_by_layer["gas_number_densities"]
        .to_dataset()
        .to_dataarray(dim="species")
    )

    if precalculated_crosssection_catalog is None:
        temperatures_by_layer: xr.DataArray = (
            vertical_structure_by_layer.temperatures_by_layer
        )

        species_present_in_model: list[str] = number_densities_by_layer.species.values

        crosssection_catalog_dataset_interpolated_to_model: xr.Dataset = (
            curate_crosssection_catalog(
                crosssection_catalog_dataset=forward_model_inputs["reference_data"][
                    "crosssection_catalog"
                ].to_dataset(),
                temperatures_by_layer=temperatures_by_layer,
                pressures_by_layer=pressures_by_layer,
                species_present_in_model=species_present_in_model,
            )
        )

    else:
        crosssection_catalog_dataset_interpolated_to_model: xr.Dataset = convert_units(
            precalculated_crosssection_catalog,
            {"wavelength": "cm", "pressure": "barye"},
        )

    model_wavelengths_in_cm: xr.DataArray = (
        crosssection_catalog_dataset_interpolated_to_model.wavelength
    )

    thermal_intensities: xr.Dataset = compile_thermal_structure_for_forward_model(
        temperatures_by_level=temperatures_by_level,
        model_wavelengths_in_cm=model_wavelengths_in_cm,
    )

    path_lengths_in_cm: xr.DataArray = vertical_structure_by_layer.path_lengths

    # TODO: there is probably a way to extend "set_result_name_and_units" so we can remove the tuple/class-like return structures
    two_stream_parameters: TwoStreamParameters = (
        compile_composite_two_stream_parameters(
            wavelengths_in_cm=model_wavelengths_in_cm,
            crosssections=crosssection_catalog_dataset_interpolated_to_model,
            number_density=number_densities_by_layer,
            path_lengths=path_lengths_in_cm,
        )
    )

    RT_Toon1989_inputs: RTToon1989Inputs = RTToon1989Inputs(
        thermal_intensity=thermal_intensities.thermal_intensity_by_layer,
        delta_thermal_intensity=thermal_intensities.delta_thermal_intensity_by_layer,
        **asdict(two_stream_parameters),
    )

    emitted_twostream_flux: xr.DataArray = RT_Toon1989(*astuple(RT_Toon1989_inputs))

    observed_twostream_flux: xr.DataArray = (
        emitted_twostream_flux
        * (
            vertical_structure_by_layer.planet_radius
            / forward_model_inputs["observable_parameters"].distance_to_system
        )
        ** 2
    ).rename("observed_twostream_flux")

    return observed_twostream_flux


def resample_observed_fluxes(
    observed_fluxes: dict[str, xr.DataArray],
    reference_model_wavelengths: xr.DataArray,
    **resampling_kwargs,
) -> dict[str, xr.DataArray]:
    return {
        observed_flux_name.replace(
            "observed", "resampled"
        ): resample_spectral_quantity_to_new_wavelengths(
            reference_model_wavelengths,
            observed_flux.wavelength,
            observed_flux,
            **resampling_kwargs,
        )
        for observed_flux_name, observed_flux in observed_fluxes.items()
    }
