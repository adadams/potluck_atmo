import sys
from pathlib import Path
from time import time
from typing import Final

import jax
import numpy as np
import xarray as xr
from jax import numpy as jnp

from potluck.xarray_functional_wrappers import Dimensionalize, set_result_name_and_units

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from potluck.basic_types import (
    LogMixingRatioValue,
    NonnegativeValue,
    NormalizedValue,
    PressureDimension,
    WavelengthDimension,
)
from potluck.material.mixing_ratios import (
    calculate_uniform_mixing_ratios_in_slab_multi_layer,
    calculate_uniform_mixing_ratios_in_slab_single_layer,
)

jax.config.update("jax_enable_x64", True)

current_directory: Path = Path(__file__).parent

SMALL_VALUE: Final[float] = 1e-6


def convert_cloud_remaining_fraction_to_thickness(
    cloud_top_log10_pressure: float,
    cloud_remaining_fraction: float,
    maximum_log10_pressure: float,
) -> float:
    return cloud_remaining_fraction * (
        maximum_log10_pressure - cloud_top_log10_pressure
    )


def convert_cloud_top_pressure_and_thickness_to_top_and_bottom_level_indices(
    cloud_top_log10_pressure: float,
    cloud_log10_thickness: float,
    log10_pressures_by_level: np.ndarray,
) -> tuple[float, float]:
    cloud_bottom_log10_pressure: float = (
        cloud_top_log10_pressure + cloud_log10_thickness
    )

    integer_part_of_top_level_index, integer_part_of_bottom_level_index = (
        np.searchsorted(
            log10_pressures_by_level,
            [cloud_top_log10_pressure, cloud_bottom_log10_pressure],
        )
    )

    # if the bottom of the cloud layer is at the bottom of the atmosphere, then we attach it to the
    # previous layer
    if integer_part_of_bottom_level_index == len(log10_pressures_by_level) - 1:
        integer_part_of_bottom_level_index -= 1

    (
        left_hand_top_pressure,
        right_hand_top_pressure,
        left_hand_bottom_pressure,
        right_hand_bottom_pressure,
    ) = np.take(
        log10_pressures_by_level,
        [
            integer_part_of_top_level_index,
            integer_part_of_top_level_index + 1,
            integer_part_of_bottom_level_index,
            integer_part_of_bottom_level_index + 1,
        ],
    )

    fractional_part_of_top_level_index: NormalizedValue = (
        cloud_top_log10_pressure - left_hand_top_pressure
    ) / (right_hand_top_pressure - left_hand_top_pressure)

    fractional_part_of_bottom_level_index: NormalizedValue = np.clip(
        (cloud_bottom_log10_pressure - left_hand_bottom_pressure)
        / (right_hand_bottom_pressure - left_hand_bottom_pressure),
        0,
        1 - SMALL_VALUE,
    )  # this should only over be exactly one if the bottom of the cloud layer is at the bottom of the atmosphere

    return (
        integer_part_of_top_level_index + fractional_part_of_top_level_index,
        integer_part_of_bottom_level_index + fractional_part_of_bottom_level_index,
    )


def convert_fraction_of_remaining_pressures_to_level_index(
    cloud_fraction_of_remaining_pressures: float,
    cloud_top_level_fractional_index: float,
    number_of_pressure_layers: int,
) -> float:
    return cloud_top_level_fractional_index + cloud_fraction_of_remaining_pressures * (
        number_of_pressure_layers - cloud_top_level_fractional_index
    )


@set_result_name_and_units(
    result_names=(
        "forward_scattering_coefficients",
        "backward_scattering_coefficients",
        "absorption_coefficients",
    ),
    units=tuple(["cm^-1"] * 3),
)
@Dimensionalize(
    argument_dimensions=(
        (PressureDimension,),
        (PressureDimension,),
        (PressureDimension,),
        None,
        (WavelengthDimension,),
        None,
    ),
    result_dimensions=(
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
    ),
)
def calculate_power_law_cloud_two_stream_parameters_across_wavelengths(
    cloud_forward_scattering_coefficients_at_reference_wavelength_by_layer: np.ndarray,
    cloud_backward_scattering_coefficients_at_reference_wavelength_by_layer: np.ndarray,
    cloud_absorption_coefficients_at_reference_wavelength_by_layer: np.ndarray,
    power_law_exponent: float,
    wavelengths_in_cm: np.ndarray[NonnegativeValue],
    reference_wavelength_in_microns: NonnegativeValue = 1.00,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_wavelength_in_cm: float = reference_wavelength_in_microns * 1e-4

    power_law_multiplicative_factors: np.ndarray = (
        (wavelengths_in_cm / reference_wavelength_in_cm) ** power_law_exponent
    )[:, None]  # (n_wavelengths, n_layers)

    return (
        cloud_absorption_coefficients_at_reference_wavelength_by_layer
        * power_law_multiplicative_factors,
        cloud_forward_scattering_coefficients_at_reference_wavelength_by_layer
        * power_law_multiplicative_factors,
        cloud_backward_scattering_coefficients_at_reference_wavelength_by_layer
        * power_law_multiplicative_factors,
    )


@set_result_name_and_units(
    result_names=(
        "cloud_forward_scattering_coefficients",
        "cloud_backward_scattering_coefficients",
        "cloud_absorption_coefficients",
    ),
    units=tuple(["cm^-1"] * 3),
)
@Dimensionalize(
    argument_dimensions=(
        (PressureDimension,),
        (PressureDimension,),
        None,
        # None,
    ),
    result_dimensions=(
        (PressureDimension,),
        (PressureDimension,),
        (PressureDimension,),
    ),
)
def calculate_power_law_cloud_two_stream_parameters_at_reference_wavelength(
    cloud_optical_depths_at_reference_wavelength_by_layer: np.ndarray,
    path_lengths_by_layer: np.ndarray,
    uniform_single_scattering_albedo: NormalizedValue,
    uniform_asymmetry_parameter: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fraction_of_extinction_in_scattering: NormalizedValue = (
        uniform_single_scattering_albedo
    )
    fraction_of_extinction_in_absorption: NormalizedValue = (
        1 - uniform_single_scattering_albedo
    )

    # forward scattering, backward scattering, absorption
    cloud_extinction_coefficients_at_reference_wavelength_by_layer: np.ndarray = (
        cloud_optical_depths_at_reference_wavelength_by_layer / path_lengths_by_layer
    )

    cloud_absorption_coefficients_at_reference_wavelength_by_layer: np.ndarray = (
        cloud_extinction_coefficients_at_reference_wavelength_by_layer
        * fraction_of_extinction_in_absorption
    )

    cloud_total_scattering_coefficients_at_reference_wavelength_by_layer: np.ndarray = (
        cloud_extinction_coefficients_at_reference_wavelength_by_layer
        * fraction_of_extinction_in_scattering
    )

    fraction_of_scattering_in_forward_direction: NormalizedValue = (
        1 + uniform_asymmetry_parameter
    ) / 2
    fraction_of_scattering_in_backward_direction: NormalizedValue = (
        1 - uniform_asymmetry_parameter
    ) / 2

    cloud_forward_scattering_coefficients_at_reference_wavelength_by_layer: np.ndarray = (
        fraction_of_scattering_in_forward_direction
        * cloud_total_scattering_coefficients_at_reference_wavelength_by_layer
    )

    cloud_backward_scattering_coefficients_at_reference_wavelength_by_layer: np.ndarray = (
        fraction_of_scattering_in_backward_direction
        * cloud_total_scattering_coefficients_at_reference_wavelength_by_layer
    )

    return (
        cloud_forward_scattering_coefficients_at_reference_wavelength_by_layer,
        cloud_backward_scattering_coefficients_at_reference_wavelength_by_layer,
        cloud_absorption_coefficients_at_reference_wavelength_by_layer,
    )


def calculate_cloud_mixing_ratios_by_layer(
    log10_uniform_cloud_mixing_ratio: LogMixingRatioValue,
    cloud_top_log10_pressure: float,
    cloud_log10_thickness: float,
    log10_pressures_by_level: np.ndarray,
):
    number_of_pressure_levels: int = len(log10_pressures_by_level)
    number_of_pressure_layers: int = number_of_pressure_levels - 1

    uniform_cloud_mixing_ratio: float = 10**log10_uniform_cloud_mixing_ratio

    cloud_top_level_fractional_index, cloud_bottom_level_fractional_index = (
        convert_cloud_top_pressure_and_thickness_to_top_and_bottom_level_indices(
            cloud_top_log10_pressure=cloud_top_log10_pressure,
            cloud_log10_thickness=cloud_log10_thickness,
            log10_pressures_by_level=log10_pressures_by_level,
        )
    )

    cloud_top_index: int = int(cloud_top_level_fractional_index)
    cloud_bottom_index: int = int(cloud_bottom_level_fractional_index)

    cloud_mixing_ratios_by_layer: np.ndarray = (
        calculate_uniform_mixing_ratios_in_slab_multi_layer(
            slab_top_level_fractional_index=cloud_top_level_fractional_index,
            slab_top_level_index=cloud_top_index,
            slab_bottom_level_fractional_index=cloud_bottom_level_fractional_index,
            slab_bottom_level_index=cloud_bottom_index,
            uniform_mixing_ratio=uniform_cloud_mixing_ratio,
            number_of_pressure_layers=number_of_pressure_layers,
        )
        if cloud_top_index != cloud_bottom_index
        else (
            calculate_uniform_mixing_ratios_in_slab_single_layer(
                slab_top_level_fractional_index=cloud_top_level_fractional_index,
                slab_bottom_level_fractional_index=cloud_bottom_level_fractional_index,
                uniform_mixing_ratio=uniform_cloud_mixing_ratio,
                number_of_pressure_layers=number_of_pressure_layers,
            )
        )
    )

    return cloud_mixing_ratios_by_layer


def calculate_cloud_number_densities_by_layer(
    log10_uniform_cloud_mixing_ratio: LogMixingRatioValue,
    cloud_top_log10_pressure: float,
    cloud_log10_thickness: float,
    log10_pressures_by_level: np.ndarray,
    total_gas_number_density_by_layer: np.ndarray,
):
    cloud_mixing_ratios_by_layer: np.ndarray = calculate_cloud_mixing_ratios_by_layer(
        log10_uniform_cloud_mixing_ratio=log10_uniform_cloud_mixing_ratio,
        cloud_top_log10_pressure=cloud_top_log10_pressure,
        cloud_log10_thickness=cloud_log10_thickness,
        log10_pressures_by_level=log10_pressures_by_level,
    )

    cloud_number_density_by_layer: np.ndarray = (
        cloud_mixing_ratios_by_layer / (1 - cloud_mixing_ratios_by_layer)
    ) * total_gas_number_density_by_layer

    return cloud_number_density_by_layer


@jax.jit
def compute_cloud_log_normal_particle_distribution_opacities(
    cloud_particle_number_densities,
    cloud_particles_mean_radii,
    log_cloud_particles_distribution_std,  # assuming same units as mean radius
    cloud_particles_radii_bin_widths,
    cloud_particles_radii,
    clouds_absorption_opacities,
    clouds_scattering_opacities,
    clouds_particles_asymmetry_parameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""This function reimplements calc_cloud_opas from fortran_radtrans_core.f90 in JAX.
    This is the petitRADTRANS implementation with minor changes to use
    number densities instead of mass densities.
    This function integrates the cloud opacity through the different layers of the atmosphere to get the total
    optical depth, scattering and anisotropic fraction.
    # TODO optical depth or opacity?

    author: Francois Rozet

    Args:
        atmosphere_densities:
            Density of the atmosphere at each of its layer # ADA: mass density
        clouds_particles_densities:
            Density of each cloud particles # ADA: mass density
        clouds_mass_fractions:
            Mass fractions of each cloud at each atmospheric layer
        cloud_particles_mean_radii:
            Mean radius of each cloud particles at each atmospheric layer
        cloud_particles_distribution_std:
            Standard deviation of the log-normal cloud particles distribution
        cloud_particles_radii_bins:
            Bins of the particles cloud radii grid
        cloud_particles_radii:
            Particles cloud radii grid
        clouds_absorption_opacities:
            Cloud absorption opacities (radius grid, wavelength grid, clouds)
        clouds_scattering_opacities:
            Cloud scattering opacities (radius grid, wavelength grid, clouds)
        clouds_particles_asymmetry_parameters:
            Cloud particles asymmetry parameters (radius grid, wavelength grid, clouds)

    Returns:

    """

    diff = jnp.log(cloud_particles_radii[:, None, None] / cloud_particles_mean_radii)

    dn_dr = (  # (n_radii, n_layers, n_clouds)
        cloud_particle_number_densities[None, :, None]
        / (
            cloud_particles_radii[:, None, None]
            * jnp.sqrt(2.0 * jnp.pi)
            * log_cloud_particles_distribution_std
        )
        * jnp.exp(-(diff**2) / (2.0 * log_cloud_particles_distribution_std**2))
    )

    # (n_radii, n_layers, n_clouds, n_wavelengths)
    integrand_absorption = (
        dn_dr[:, :, :, None] * clouds_absorption_opacities[:, None, None, :]
    )
    integrand_scattering = (
        dn_dr[:, :, :, None] * clouds_scattering_opacities[:, None, None, :]
    )

    forward_scattering_fractions = (1 + clouds_particles_asymmetry_parameters) / 2
    integrand_forward_scattering = (
        integrand_scattering * forward_scattering_fractions[:, None, None, :]
    )

    backward_scattering_fractions = (1 - clouds_particles_asymmetry_parameters) / 2
    integrand_backward_scattering = (
        integrand_scattering * backward_scattering_fractions[:, None, None, :]
    )

    widths = cloud_particles_radii_bin_widths[:, None, None, None]

    # ADA: sum over axes 0 (n_radii) and 3 (n_clouds)
    cloud_absorption_opacities = jnp.sum(
        integrand_absorption * widths, axis=(0, 2)
    )  # (n_wavelengths, n_layers)
    cloud_forward_scattering_opacities = jnp.sum(
        integrand_forward_scattering * widths, axis=(0, 2)
    )  # (n_wavelengths, n_layers)
    cloud_backward_scattering_opacities = jnp.sum(
        integrand_backward_scattering * widths, axis=(0, 2)
    )  # (n_wavelengths, n_layers)

    return (
        cloud_forward_scattering_opacities,
        cloud_backward_scattering_opacities,
        cloud_absorption_opacities,
    )


if __name__ == "__main__":
    test_forward_model_datatree_filepath: Path = (
        current_directory / "Ross458c_no_clouds_retrieved_forward_model_structure.nc"
    )

    test_forward_model_datatree: xr.DataTree = xr.open_datatree(
        test_forward_model_datatree_filepath
    )

    clouds_directory: Path = current_directory / "reference_data"
    crystalline_Na2S_filepath: Path = (
        clouds_directory / "crystalline_sodium_sulfide_for_potluck.nc"
    )

    cloud_dataset: xr.Dataset = xr.open_dataset(crystalline_Na2S_filepath)

    test_forward_model_gas_number_densities: xr.DataArray = (
        test_forward_model_datatree["atmospheric_structure_by_layer"][
            "gas_number_densities"
        ]
        .to_dataset()
        .to_dataarray(dim="species")
    )

    pressures_by_layer: xr.DataArray = test_forward_model_datatree[
        "atmospheric_structure_by_layer"
    ]["vertical_structure"].pressures_by_layer

    log10_pressures_by_layer: xr.DataArray = test_forward_model_datatree[
        "atmospheric_structure_by_layer"
    ]["vertical_structure"].log_pressures_by_layer

    log10_pressures_by_level: xr.DataArray = np.log10(
        test_forward_model_datatree["temperature_profile_by_level"].pressure
    )

    cloud_log10_mixing_ratio: float = -4.00
    cloud_mixing_ratio: float = 10**cloud_log10_mixing_ratio
    cloud_top_log10_pressure: float = -3.00 + 6  # barye = 3
    cloud_base_log10_pressure: float = 0.50 + 6  # barye = 6.5

    cloud_top_level_fractional_index: float = 5.3846
    cloud_fraction_of_remaining_pressures: float = 0.63636363

    number_of_pressure_layers: int = len(pressures_by_layer)

    for i in range(10):
        start_time: float = time()

        test_cloud_mixing_ratio: float = cloud_mixing_ratio * (1 + 0.01 * i)
        test_cloud_fraction_of_remaining_pressures: float = (
            cloud_fraction_of_remaining_pressures + (0.01 * i)
        )
        test_cloud_top_level_fractional_index: float = (
            cloud_top_level_fractional_index + (0.01 * i)
        )

        cloud_bottom_level_fractional_index: float = (
            cloud_top_level_fractional_index
            + cloud_fraction_of_remaining_pressures
            * (number_of_pressure_layers - cloud_top_level_fractional_index)
        )

        cloud_top_index: int = int(cloud_top_level_fractional_index)
        cloud_bottom_index: int = int(cloud_bottom_level_fractional_index)

        if cloud_bottom_index == cloud_top_index:
            test_cloud_mixing_ratio_array: np.ndarray = calculate_uniform_mixing_ratios_in_slab_single_layer(
                slab_top_level_fractional_index=test_cloud_top_level_fractional_index,
                slab_bottom_level_fractional_index=cloud_bottom_level_fractional_index,
                uniform_mixing_ratio=test_cloud_mixing_ratio,
                number_of_pressure_layers=number_of_pressure_layers,
            )

        else:
            test_cloud_mixing_ratio_array: np.ndarray = calculate_uniform_mixing_ratios_in_slab_multi_layer(
                slab_top_level_fractional_index=test_cloud_top_level_fractional_index,
                slab_top_level_index=cloud_top_index,
                slab_bottom_level_fractional_index=cloud_bottom_level_fractional_index,
                slab_bottom_level_index=cloud_bottom_index,
                uniform_mixing_ratio=test_cloud_mixing_ratio,
                number_of_pressure_layers=number_of_pressure_layers,
            )

        test_cloud_mixing_ratios_by_layer: xr.DataArray = xr.DataArray(
            test_cloud_mixing_ratio_array,
            dims=log10_pressures_by_layer.dims,
            coords=log10_pressures_by_layer.coords,
        )

        test_cloud_number_density: xr.DataArray = (
            test_cloud_mixing_ratios_by_layer / (1 - test_cloud_mixing_ratios_by_layer)
        ) * test_forward_model_gas_number_densities.sum("species")

        lognormal_start_time: float = time()
        (
            cloud_forward_scattering_opacities,
            cloud_backward_scattering_opacities,
            cloud_absorption_opacities,
        ) = compute_cloud_log_normal_particle_distribution_opacities(
            cloud_particle_number_densities=test_cloud_number_density.to_numpy(),
            cloud_particles_densities=cloud_dataset.cloud_material_density.item(),
            cloud_particles_mean_radii=1e-5,  # [cm]
            log_cloud_particles_distribution_std=0.5 + 0.01 * i,
            cloud_particles_radii_bin_widths=cloud_dataset.cloud_particles_radii_bin_widths.to_numpy(),
            cloud_particles_radii=cloud_dataset.particle_radius.to_numpy(),
            clouds_absorption_opacities=cloud_dataset.cloud_absorption_opacities.to_numpy(),
            clouds_scattering_opacities=cloud_dataset.cloud_scattering_opacities.to_numpy(),
            clouds_particles_asymmetry_parameters=cloud_dataset.cloud_scattering_asymmetry_parameters.to_numpy(),
        )
        end_time: float = time()
        print(
            f"Time to compute cloud opacities overall: {end_time - start_time}, "
            f"time_just_for_lognormal: {end_time - lognormal_start_time}"
        )

    print(f"{cloud_absorption_opacities=}")
    print(f"{cloud_forward_scattering_opacities=}")
    print(f"{cloud_backward_scattering_opacities=}")

    cloud_absorption_opacity_dataarray: xr.DataArray = xr.DataArray(
        cloud_absorption_opacities,
        dims=("pressure", "wavelength"),
        coords={"pressure": pressures_by_layer, "wavelength": cloud_dataset.wavelength},
    )

    cloud_absorption_opacity_dataarray.to_netcdf(
        current_directory / "cloud_absorption_opacities.nc"
    )

    cloud_forward_scattering_opacity_dataarray: xr.DataArray = xr.DataArray(
        cloud_forward_scattering_opacities,
        dims=("pressure", "wavelength"),
        coords={"pressure": pressures_by_layer, "wavelength": cloud_dataset.wavelength},
    )

    cloud_forward_scattering_opacity_dataarray.to_netcdf(
        current_directory / "cloud_forward_scattering_opacities.nc"
    )

    cloud_backward_scattering_opacity_dataarray: xr.DataArray = xr.DataArray(
        cloud_backward_scattering_opacities,
        dims=("pressure", "wavelength"),
        coords={"pressure": pressures_by_layer, "wavelength": cloud_dataset.wavelength},
    )

    cloud_backward_scattering_opacity_dataarray.to_netcdf(
        current_directory / "cloud_backward_scattering_opacities.nc"
    )
