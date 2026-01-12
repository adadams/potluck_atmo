from collections.abc import Iterable
from typing import TypeAlias

import xarray as xr

from potluck.material.absorbing.from_crosssections import (
    attenuation_coefficients_to_optical_depths,
    crosssections_to_attenuation_coefficients,
)
from potluck.material.clouds.cloud_metrics import (
    calculate_power_law_cloud_two_stream_parameters_across_wavelengths,
    calculate_power_law_cloud_two_stream_parameters_at_reference_wavelength,
    compute_cloud_log_normal_particle_distribution_opacities,
)
from potluck.material.scattering.rayleigh import (
    calculate_two_stream_scattering_components,
)
from potluck.material.scattering.scattering_types import TwoStreamScatteringCoefficients
from potluck.material.scattering.two_stream import (
    calculate_two_stream_scattering_parameters,
)
from potluck.material.types import TwoStreamParameters, TwoStreamScatteringParameters

# @dataclass
# class TwoStreamInputs:
#    forward_scattering_coefficients: xr.DataArray
#    backward_scattering_coefficients: xr.DataArray
#    absorption_coefficients: xr.DataArray
#    # path_length: xr.DataArray

TwoStreamInputs: TypeAlias = xr.Dataset


def compile_two_stream_parameters(
    forward_scattering_coefficients: xr.DataArray,
    backward_scattering_coefficients: xr.DataArray,
    absorption_coefficients: xr.DataArray,
    path_lengths: xr.DataArray,
) -> TwoStreamParameters:
    two_stream_scattering_parameters: TwoStreamScatteringParameters = (
        TwoStreamScatteringParameters(
            *calculate_two_stream_scattering_parameters(
                forward_scattering_coefficients,
                backward_scattering_coefficients,
                absorption_coefficients,
            )
        )
    )

    optical_depth: xr.DataArray = attenuation_coefficients_to_optical_depths(
        absorption_coefficients, path_lengths
    )

    return TwoStreamParameters(
        scattering_asymmetry_parameter=two_stream_scattering_parameters.scattering_asymmetry_parameter,
        single_scattering_albedo=two_stream_scattering_parameters.single_scattering_albedo,
        optical_depth=optical_depth,
    )


def compile_gas_two_stream_inputs_by_species(
    wavelengths_in_cm: xr.DataArray,  # (wavelength,)
    crosssections: xr.DataArray,  # (species, wavelength, pressure)
    number_density: xr.DataArray,  # (species, pressure)
) -> xr.Dataset:
    scattering_coefficients: TwoStreamScatteringCoefficients = (
        calculate_two_stream_scattering_components(wavelengths_in_cm, number_density)
    )

    absorption_coefficients: xr.Dataset = crosssections_to_attenuation_coefficients(
        crosssections, number_density
    )

    (
        cumulative_forward_scattering_coefficients,
        cumulative_backward_scattering_coefficients,
    ) = (
        scattering_coefficients.forward_scattering_coefficients,
        scattering_coefficients.backward_scattering_coefficients,
    )

    cumulative_absorption_coefficients: xr.DataArray = (
        absorption_coefficients
        # + cumulative_forward_scattering_coefficients
        # + cumulative_backward_scattering_coefficients
    )

    return xr.Dataset(
        {
            "forward_scattering_coefficients": cumulative_forward_scattering_coefficients,
            "backward_scattering_coefficients": cumulative_backward_scattering_coefficients,
            "absorption_coefficients": cumulative_absorption_coefficients,
        }
    )


def compile_gas_two_stream_inputs(
    wavelengths_in_cm: xr.DataArray,  # (wavelength,)
    crosssections: xr.DataArray,  # (species, wavelength, pressure)
    number_density: xr.DataArray,  # (species, pressure)
) -> xr.Dataset:
    gas_two_stream_inputs_by_species: xr.Dataset = (
        compile_gas_two_stream_inputs_by_species(
            wavelengths_in_cm, crosssections, number_density
        )
    )

    return gas_two_stream_inputs_by_species.sum("species")


"""
def compile_gas_two_stream_inputs(
    wavelengths_in_cm: xr.DataArray,  # (wavelength,)
    crosssections: xr.DataArray,  # (species, wavelength, pressure)
    number_density: xr.DataArray,  # (species, pressure)
) -> xr.Dataset:
    scattering_coefficients: TwoStreamScatteringCoefficients = (
        calculate_two_stream_scattering_components(wavelengths_in_cm, number_density)
    )

    absorption_coefficients: xr.Dataset = crosssections_to_attenuation_coefficients(
        crosssections, number_density
    )

    (
        cumulative_forward_scattering_coefficients,
        cumulative_backward_scattering_coefficients,
    ) = (
        scattering_coefficients.forward_scattering_coefficients.sum("species"),
        scattering_coefficients.backward_scattering_coefficients.sum("species"),
    )

    cumulative_absorption_coefficients: xr.DataArray = (
        absorption_coefficients.sum("species")
        # + cumulative_forward_scattering_coefficients
        # + cumulative_backward_scattering_coefficients
    )

    return xr.Dataset(
        {
            "forward_scattering_coefficients": cumulative_forward_scattering_coefficients,
            "backward_scattering_coefficients": cumulative_backward_scattering_coefficients,
            "absorption_coefficients": cumulative_absorption_coefficients,
        }
    )
"""


def compile_composite_two_stream_parameters_with_gas_only(
    wavelengths_in_cm: xr.DataArray,  # (wavelength,)
    crosssections: xr.DataArray,  # (species, wavelength, pressure)
    number_density: xr.DataArray,  # (species, pressure)
    path_lengths: xr.DataArray,  # (pressure,)
) -> TwoStreamParameters:
    two_stream_inputs: xr.Dataset = compile_gas_two_stream_inputs(
        wavelengths_in_cm, crosssections, number_density
    )

    return compile_two_stream_parameters(
        forward_scattering_coefficients=two_stream_inputs.forward_scattering_coefficients,
        backward_scattering_coefficients=two_stream_inputs.backward_scattering_coefficients,
        absorption_coefficients=two_stream_inputs.absorption_coefficients,
        path_lengths=path_lengths,
    )


def compile_composite_two_stream_parameters_with_gas_and_power_law_clouds(
    wavelengths_in_cm: xr.DataArray,  # (wavelength,)
    gas_crosssections: xr.DataArray,  # (species, wavelength, pressure)
    gas_number_density: xr.DataArray,  # (species, pressure))
    path_lengths: xr.DataArray,  # (pressure,)
    power_law_cloud_profiles: xr.Dataset,  # each array is a cloud profile, with (pressure,)
) -> TwoStreamParameters:
    gas_two_stream_inputs: xr.Dataset = compile_gas_two_stream_inputs(
        wavelengths_in_cm, gas_crosssections, gas_number_density
    )

    cloud_two_stream_parameters_by_slab: list[xr.Dataset] = [
        xr.merge(
            calculate_power_law_cloud_two_stream_parameters_at_reference_wavelength(
                cloud_profile,
                path_lengths,
                cloud_profile.attrs["single_scattering_albedo"],
            )
        )
        for cloud_profile in power_law_cloud_profiles.data_vars.values()
    ]

    cumulative_cloud_two_stream_inputs_by_slab: xr.Dataset = [
        xr.merge(
            calculate_power_law_cloud_two_stream_parameters_across_wavelengths(
                cloud_two_stream_parameters.cloud_forward_scattering_coefficients,
                cloud_two_stream_parameters.cloud_backward_scattering_coefficients,
                cloud_two_stream_parameters.cloud_absorption_coefficients,
                cloud_profile.attrs["power_law_exponent"],
                wavelengths_in_cm,
                cloud_profile.attrs["reference_wavelength_in_microns"],
            )
        )
        for cloud_profile, cloud_two_stream_parameters in zip(
            power_law_cloud_profiles.data_vars.values(),
            cloud_two_stream_parameters_by_slab,
        )
    ]

    cumulative_cloud_two_stream_inputs: xr.Dataset = xr.concat(
        cumulative_cloud_two_stream_inputs_by_slab, dim="cloud_slab"
    ).sum("cloud_slab")
    # cumulative_cloud_two_stream_inputs.to_netcdf("power_law_cloud_two_stream_inputs_at_reference_wavelength.nc")

    return compile_composite_two_stream_parameters(
        [gas_two_stream_inputs, cumulative_cloud_two_stream_inputs], path_lengths
    )


def compile_composite_two_stream_parameters_with_gas_and_clouds(
    wavelengths_in_cm: xr.DataArray,  # (wavelength,)
    gas_crosssections: xr.DataArray,  # (species, wavelength, pressure)
    gas_number_density: xr.DataArray,  # (species, pressure)
    cloud_crosssections: xr.DataTree,
    cloud_number_densities: xr.Dataset,
    path_lengths: xr.DataArray,  # (pressure,)
) -> TwoStreamParameters:
    gas_two_stream_inputs: xr.Dataset = compile_gas_two_stream_inputs(
        wavelengths_in_cm, gas_crosssections, gas_number_density
    )
    # gas_two_stream_inputs.to_netcdf("gas_two_stream_inputs.nc")

    for cloud_species in cloud_crosssections.children:
        cloud_crosssections_for_species: xr.DataArray = cloud_crosssections[
            cloud_species
        ]

        cloud_number_densities_for_species: xr.DataArray = cloud_number_densities[
            cloud_species
        ]

        (
            cloud_forward_scattering_coefficients,
            cloud_backward_scattering_coefficients,
            cloud_absorption_coefficients,
        ) = compute_cloud_log_normal_particle_distribution_opacities(
            cloud_particle_number_densities=cloud_number_densities_for_species.to_numpy(),
            cloud_particles_mean_radii=cloud_number_densities_for_species.attrs[
                "mean_particle_radius"
            ],
            log_cloud_particles_distribution_std=cloud_number_densities_for_species.attrs[
                "log10_particle_distribution_standard_deviation"
            ],
            cloud_particles_radii_bin_widths=cloud_crosssections_for_species.particle_radius_bin_widths.to_numpy(),
            cloud_particles_radii=cloud_crosssections_for_species.particle_radius.to_numpy(),
            clouds_absorption_opacities=cloud_crosssections_for_species.cloud_absorption_opacities.to_numpy(),
            clouds_scattering_opacities=cloud_crosssections_for_species.cloud_scattering_opacities.to_numpy(),
            clouds_particles_asymmetry_parameters=cloud_crosssections_for_species.cloud_scattering_asymmetry_parameters.to_numpy(),
        )

        cloud_forward_scattering_dataarray: xr.DataArray = xr.DataArray(
            name="forward_scattering_coefficients",
            data=cloud_forward_scattering_coefficients,
            dims=("pressure", "wavelength"),
            coords={
                "pressure": cloud_number_densities_for_species.pressure,
                "wavelength": cloud_crosssections_for_species.wavelength,
            },
        ).interp(wavelength=wavelengths_in_cm)

        cloud_backward_scattering_dataarray: xr.DataArray = xr.DataArray(
            name="backward_scattering_coefficients",
            data=cloud_backward_scattering_coefficients,
            dims=("pressure", "wavelength"),
            coords={
                "pressure": cloud_number_densities_for_species.pressure,
                "wavelength": cloud_crosssections_for_species.wavelength,
            },
        ).interp(wavelength=wavelengths_in_cm)

        cloud_absorption_dataarray: xr.DataArray = xr.DataArray(
            name="absorption_coefficients",
            data=cloud_absorption_coefficients,
            dims=("pressure", "wavelength"),
            coords={
                "pressure": cloud_number_densities_for_species.pressure,
                "wavelength": cloud_crosssections_for_species.wavelength,
            },
        ).interp(wavelength=wavelengths_in_cm)

        cloud_two_stream_inputs: xr.Dataset = xr.Dataset(
            {
                "forward_scattering_coefficients": cloud_forward_scattering_dataarray,
                "backward_scattering_coefficients": cloud_backward_scattering_dataarray,
                "absorption_coefficients": cloud_absorption_dataarray,
            }
        )
        # cloud_two_stream_inputs.to_netcdf("cloud_two_stream_inputs.nc")

    return compile_composite_two_stream_parameters(
        [gas_two_stream_inputs, cloud_two_stream_inputs], path_lengths
    )


def compile_composite_two_stream_parameters(
    two_stream_inputs: Iterable[TwoStreamInputs],
    path_lengths: xr.DataArray,  # (pressure,)
) -> TwoStreamParameters:
    concatenated_two_stream_inputs: xr.Dataset = xr.concat(
        two_stream_inputs, dim="source"
    ).sum("source")

    composite_two_stream_parameters: TwoStreamParameters = compile_two_stream_parameters(
        forward_scattering_coefficients=concatenated_two_stream_inputs.forward_scattering_coefficients,
        backward_scattering_coefficients=concatenated_two_stream_inputs.backward_scattering_coefficients,
        absorption_coefficients=concatenated_two_stream_inputs.absorption_coefficients,
        path_lengths=path_lengths,
    )

    return composite_two_stream_parameters
