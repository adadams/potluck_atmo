from typing import TypedDict

import xarray as xr

from material.absorbing.from_crosssections import (
    attenuation_coefficients_to_optical_depths,
)
from material.scattering.two_stream import calculate_two_stream_scattering_parameters
from material.types import TwoStreamParameters, TwoStreamScatteringParameters


class TwoStreamInputs(TypedDict):
    forward_scattering_coefficients: xr.DataArray
    backward_scattering_coefficients: xr.DataArray
    absorption_coefficients: xr.DataArray
    path_length: xr.DataArray


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
