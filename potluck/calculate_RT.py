from dataclasses import asdict, astuple
from pathlib import Path
from typing import Optional

import xarray as xr

from potluck.compile_crosssection_data import curate_gas_crosssection_catalog
from potluck.compile_thermal_structure import (
    compile_thermal_structure_for_forward_model,
)
from potluck.compile_vertical_structure import (
    ForwardModelXarrayInputs,
    compile_vertical_structure_for_forward_model,
)
from potluck.constants_and_conversions import MICRONS_TO_CM
from potluck.input_structs import UserForwardModelInputs
from potluck.material.absorbing.from_crosssections import (
    attenuation_coefficients_to_optical_depths,
    crosssections_to_attenuation_coefficients,
)
from potluck.material.two_stream import (
    compile_composite_two_stream_parameters,
    compile_composite_two_stream_parameters_with_gas_and_clouds,
    compile_composite_two_stream_parameters_with_gas_and_power_law_clouds,
    compile_composite_two_stream_parameters_with_gas_only,
)
from potluck.material.types import TwoStreamParameters
from potluck.radiative_transfer.RT_in_transmission import (
    calculate_transmission_spectrum,
)
from potluck.radiative_transfer.RT_one_stream import (
    OneStreamRTInputs,
    calculate_spectral_intensity_at_surface,
)
from potluck.radiative_transfer.RT_Toon1989_jax import RT_Toon1989, RTToon1989Inputs
from potluck.spectrum.bin import resample_spectral_quantity_to_new_wavelengths
from potluck.xarray_functional_wrappers import XarrayOutputs

current_directory: Path = Path(__file__).parent


# TODO: update with datatree input approach
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

        gas_crosssection_catalog_interpolated_to_model: xr.Dataset = (
            curate_gas_crosssection_catalog(
                crosssection_catalog=user_forward_model_inputs.gas_crosssection_catalog,
                temperatures_by_layer=temperatures_by_layer,
                pressures_by_layer=pressures_by_layer,
                species_present_in_model=species_present_in_model,
            )
        )

    else:
        gas_crosssection_catalog_interpolated_to_model: xr.Dataset = (
            precalculated_crosssection_catalog
        )

    model_wavelengths_in_microns: xr.DataArray = (
        gas_crosssection_catalog_interpolated_to_model.wavelength
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
            crosssections=gas_crosssection_catalog_interpolated_to_model,
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


def calculate_observed_transmission_spectrum(
    forward_model_inputs: xr.DataTree,
) -> XarrayOutputs:
    atmospheric_structure_by_layer: xr.DataArray = forward_model_inputs[
        "atmospheric_structure_by_layer"
    ]

    vertical_structure_by_layer: xr.DataArray = atmospheric_structure_by_layer[
        "vertical_structure"
    ]

    path_lengths_in_cm: xr.DataArray = vertical_structure_by_layer.path_lengths

    number_densities_by_layer: xr.DataArray = (
        atmospheric_structure_by_layer["gas_number_densities"]
        .to_dataset()
        .to_dataarray(dim="species")
    )

    gas_crosssection_catalog: xr.Dataset = forward_model_inputs["reference_data"][
        "gas_crosssection_catalog"
    ].to_dataset()

    gas_crosssection_catalog_as_dataarray: xr.DataArray = (
        gas_crosssection_catalog.to_array(dim="species", name="crosssections")
    )

    absorption_coefficients: xr.Dataset = crosssections_to_attenuation_coefficients(
        gas_crosssection_catalog_as_dataarray, number_densities_by_layer
    )

    optical_depth: xr.DataArray = attenuation_coefficients_to_optical_depths(
        absorption_coefficients, path_lengths_in_cm
    )

    cumulative_optical_depth_by_layer: xr.DataArray = (
        optical_depth.cumulative("pressure").sum().sum("species")
    )

    transmission_flux: xr.DataArray = calculate_transmission_spectrum(
        cumulative_optical_depth_by_layer,
        path_lengths_in_cm,
        vertical_structure_by_layer.altitudes_by_layer,
        forward_model_inputs["observable_parameters"].stellar_radius,
        vertical_structure_by_layer.planet_radius,
    )

    return transmission_flux


# TODO: ideally we can refine the interface for these sorts of functions.
# In other words, instead of providing the entire forward model input datatree,
# there could be a specific "pruning" of the tree to match the interface for a function
# like this. This would also be a helpful part of the goal of using xarray dataset/datatree
# protocols to prescribe what components are needed as arguments for a function. That sort of
# thing is currently obscured by passing an entire xarray structure as a composite argument.
def calculate_observed_fluxes_with_clouds_via_two_stream(
    forward_model_inputs: xr.DataTree,
):
    temperatures_by_level: xr.DataArray = forward_model_inputs[
        "temperature_profile_by_level"
    ].temperature

    atmospheric_structure_by_layer: xr.DataArray = forward_model_inputs[
        "atmospheric_structure_by_layer"
    ]

    vertical_structure_by_layer: xr.DataArray = atmospheric_structure_by_layer[
        "vertical_structure"
    ]

    number_densities_by_layer: xr.DataArray = (
        atmospheric_structure_by_layer["gas_number_densities"]
        .to_dataset()
        .to_dataarray(dim="species")
    )

    gas_crosssection_catalog: xr.Dataset = forward_model_inputs["reference_data"][
        "gas_crosssection_catalog"
    ].to_dataset()

    gas_crosssection_catalog_as_dataarray: xr.DataArray = (
        gas_crosssection_catalog.to_array(dim="species", name="crosssections")
    )

    model_wavelengths_in_cm: xr.DataArray = gas_crosssection_catalog.wavelength

    thermal_intensities: xr.Dataset = compile_thermal_structure_for_forward_model(
        temperatures_by_level=temperatures_by_level,
        model_wavelengths_in_cm=model_wavelengths_in_cm,
    )

    path_lengths_in_cm: xr.DataArray = vertical_structure_by_layer.path_lengths

    # TODO: there is probably a way to extend "set_result_name_and_units" so we can remove the tuple/class-like return structures
    two_stream_parameters: TwoStreamParameters = (
        compile_composite_two_stream_parameters_with_gas_and_clouds(
            wavelengths_in_cm=model_wavelengths_in_cm,
            gas_crosssections=gas_crosssection_catalog_as_dataarray,
            gas_number_density=number_densities_by_layer,
            cloud_crosssections=forward_model_inputs["reference_data"][
                "cloud_crosssection_catalog"
            ],
            cloud_number_densities=atmospheric_structure_by_layer[
                "cloud_number_densities"
            ],
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


def calculate_observed_fluxes_with_power_law_clouds_via_two_stream(
    forward_model_inputs: xr.DataTree,
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

    number_densities_by_layer: xr.DataArray = (
        atmospheric_structure_by_layer["gas_number_densities"]
        .to_dataset()
        .to_dataarray(dim="species")
    )

    gas_crosssection_catalog: xr.Dataset = forward_model_inputs["reference_data"][
        "gas_crosssection_catalog"
    ].to_dataset()

    gas_crosssection_catalog_as_dataarray: xr.DataArray = (
        gas_crosssection_catalog.to_array(dim="species", name="crosssections")
    )

    model_wavelengths_in_cm: xr.DataArray = gas_crosssection_catalog.wavelength

    thermal_intensities: xr.Dataset = compile_thermal_structure_for_forward_model(
        temperatures_by_level=temperatures_by_level,
        model_wavelengths_in_cm=model_wavelengths_in_cm,
    )

    path_lengths_in_cm: xr.DataArray = vertical_structure_by_layer.path_lengths

    power_law_cloud_profiles: xr.Dataset = forward_model_inputs[
        "atmospheric_structure_by_layer"
    ]["cloud_reference_optical_depths"].to_dataset()

    # TODO: there is probably a way to extend "set_result_name_and_units" so we can remove the tuple/class-like return structures
    two_stream_parameters: TwoStreamParameters = (
        compile_composite_two_stream_parameters_with_gas_and_power_law_clouds(
            wavelengths_in_cm=model_wavelengths_in_cm,
            gas_crosssections=gas_crosssection_catalog_as_dataarray,
            gas_number_density=number_densities_by_layer,
            path_lengths=path_lengths_in_cm,
            power_law_cloud_profiles=power_law_cloud_profiles,
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


def calculate_emitted_fluxes_without_clouds_via_two_stream(
    forward_model_inputs: xr.DataTree,
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

    number_densities_by_layer: xr.DataArray = (
        atmospheric_structure_by_layer["gas_number_densities"]
        .to_dataset()
        .to_dataarray(dim="species")
    )

    gas_crosssection_catalog: xr.Dataset = forward_model_inputs["reference_data"][
        "gas_crosssection_catalog"
    ].to_dataset()

    gas_crosssection_catalog_as_dataarray: xr.DataArray = (
        gas_crosssection_catalog.to_array(dim="species", name="crosssections")
    )

    model_wavelengths_in_cm: xr.DataArray = gas_crosssection_catalog.wavelength

    thermal_intensities: xr.Dataset = compile_thermal_structure_for_forward_model(
        temperatures_by_level=temperatures_by_level,
        model_wavelengths_in_cm=model_wavelengths_in_cm,
    )

    path_lengths_in_cm: xr.DataArray = vertical_structure_by_layer.path_lengths

    # TODO: there is probably a way to extend "set_result_name_and_units" so we can remove the tuple/class-like return structures
    two_stream_parameters: TwoStreamParameters = (
        compile_composite_two_stream_parameters_with_gas_only(
            wavelengths_in_cm=model_wavelengths_in_cm,
            crosssections=gas_crosssection_catalog_as_dataarray,
            number_density=number_densities_by_layer,
            path_lengths=path_lengths_in_cm,
        )
    )

    RT_Toon1989_inputs: RTToon1989Inputs = RTToon1989Inputs(
        thermal_intensity=thermal_intensities.thermal_intensity_by_layer,
        delta_thermal_intensity=thermal_intensities.delta_thermal_intensity_by_layer,
        **asdict(two_stream_parameters),
    )

    emitted_twostream_flux: xr.DataArray = RT_Toon1989(
        *astuple(RT_Toon1989_inputs)
    ).rename("emitted_twostream_flux")

    return emitted_twostream_flux


def calculate_observed_fluxes_without_clouds_via_two_stream(
    forward_model_inputs: xr.DataTree,
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

    number_densities_by_layer: xr.DataArray = (
        atmospheric_structure_by_layer["gas_number_densities"]
        .to_dataset()
        .to_dataarray(dim="species")
    )

    gas_crosssection_catalog: xr.Dataset = forward_model_inputs["reference_data"][
        "gas_crosssection_catalog"
    ].to_dataset()

    gas_crosssection_catalog_as_dataarray: xr.DataArray = (
        gas_crosssection_catalog.to_array(dim="species", name="crosssections")
    )

    model_wavelengths_in_cm: xr.DataArray = gas_crosssection_catalog.wavelength

    thermal_intensities: xr.Dataset = compile_thermal_structure_for_forward_model(
        temperatures_by_level=temperatures_by_level,
        model_wavelengths_in_cm=model_wavelengths_in_cm,
    )

    path_lengths_in_cm: xr.DataArray = vertical_structure_by_layer.path_lengths

    # TODO: there is probably a way to extend "set_result_name_and_units" so we can remove the tuple/class-like return structures
    two_stream_parameters: TwoStreamParameters = (
        compile_composite_two_stream_parameters_with_gas_only(
            wavelengths_in_cm=model_wavelengths_in_cm,
            crosssections=gas_crosssection_catalog_as_dataarray,
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


def calculate_observed_fluxes_without_clouds_via_one_stream(
    forward_model_inputs: xr.DataTree,
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

    number_densities_by_layer: xr.DataArray = (
        atmospheric_structure_by_layer["gas_number_densities"]
        .to_dataset()
        .to_dataarray(dim="species")
    )

    gas_crosssection_catalog: xr.Dataset = forward_model_inputs["reference_data"][
        "gas_crosssection_catalog"
    ].to_dataset()

    gas_crosssection_catalog_as_dataarray: xr.DataArray = (
        gas_crosssection_catalog.to_array(dim="species", name="crosssections")
    )

    model_wavelengths_in_cm: xr.DataArray = gas_crosssection_catalog.wavelength

    thermal_intensities: xr.Dataset = compile_thermal_structure_for_forward_model(
        temperatures_by_level=temperatures_by_level,
        model_wavelengths_in_cm=model_wavelengths_in_cm,
    )

    path_lengths_in_cm: xr.DataArray = vertical_structure_by_layer.path_lengths

    absorption_coefficients: xr.Dataset = crosssections_to_attenuation_coefficients(
        gas_crosssection_catalog_as_dataarray, number_densities_by_layer
    )

    optical_depth: xr.DataArray = attenuation_coefficients_to_optical_depths(
        absorption_coefficients, path_lengths_in_cm
    )

    cumulative_optical_depth_by_layer: xr.DataArray = (
        optical_depth.cumulative("pressure").sum().sum("species")
    )

    onestream_inputs: OneStreamRTInputs = OneStreamRTInputs(
        thermal_intensity=thermal_intensities.thermal_intensity_by_layer,
        cumulative_optical_depth_by_layer=cumulative_optical_depth_by_layer,
    )

    emitted_onestream_flux: xr.DataArray = calculate_spectral_intensity_at_surface(
        *astuple(onestream_inputs)
    )

    observed_onestream_flux: xr.DataArray = (
        emitted_onestream_flux
        * (
            vertical_structure_by_layer.planet_radius
            / forward_model_inputs["observable_parameters"].distance_to_system
        )
        ** 2
    ).rename("observed_onestream_flux")

    return observed_onestream_flux


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
