from dataclasses import dataclass

import xarray as xr

from potluck.material.absorbing.from_crosssections import (
    attenuation_coefficients_to_optical_depths,
    crosssections_to_attenuation_coefficients,
)
from potluck.material.scattering.rayleigh import (
    calculate_two_stream_scattering_components,
)
from potluck.material.scattering.two_stream import (
    calculate_two_stream_scattering_parameters,
)
from potluck.material.scattering.types import TwoStreamScatteringCoefficients
from potluck.material.types import TwoStreamParameters, TwoStreamScatteringParameters


@dataclass
class TwoStreamInputs:
    forward_scattering_coefficients: xr.DataArray
    backward_scattering_coefficients: xr.DataArray
    absorption_coefficients: xr.DataArray
    # path_length: xr.DataArray


def compile_two_stream_parameters(
    forward_scattering_coefficients: xr.DataArray,
    backward_scattering_coefficients: xr.DataArray,
    absorption_coefficients: xr.DataArray,
    path_length: xr.DataArray,
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
        absorption_coefficients, path_length
    )

    return TwoStreamParameters(
        scattering_asymmetry_parameter=two_stream_scattering_parameters.scattering_asymmetry_parameter,
        single_scattering_albedo=two_stream_scattering_parameters.single_scattering_albedo,
        optical_depth=optical_depth,
    )


def compile_composite_two_stream_parameters(
    wavelengths_in_cm: xr.DataArray,  # (wavelength,)
    crosssections: xr.DataArray,  # (species, wavelength, pressure)
    number_density: xr.DataArray,  # (species, pressure)
    path_lengths: xr.DataArray,  # (pressure,)
) -> TwoStreamParameters:
    scattering_coefficients: TwoStreamScatteringCoefficients = (
        calculate_two_stream_scattering_components(
            wavelengths_in_cm, crosssections, number_density
        )
    )

    absorption_coefficients: xr.Dataset = crosssections_to_attenuation_coefficients(
        crosssections, number_density
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

    cumulative_absorption_coefficients: xr.DataArray = (
        absorption_coefficients.sum("species")
        + cumulative_forward_scattering_coefficients
        + cumulative_backward_scattering_coefficients
    )

    return compile_two_stream_parameters(
        forward_scattering_coefficients=cumulative_forward_scattering_coefficients,
        backward_scattering_coefficients=cumulative_backward_scattering_coefficients,
        absorption_coefficients=cumulative_absorption_coefficients,
        path_length=path_lengths,
    )
