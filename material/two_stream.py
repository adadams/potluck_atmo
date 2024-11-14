from typing import TypedDict

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from material.absorbing.from_crosssections import (
    attenuation_coefficients_to_optical_depths,
)
from material.scattering.two_stream import calculate_twostream_scattering_parameters
from material.types import TwoStreamParameters, TwoStreamScatteringParameters


class TwoStreamInputs(TypedDict):
    forward_scattering_coefficients: xr.DataArray
    backward_scattering_coefficients: xr.DataArray
    absorption_coefficients: xr.DataArray
    path_length: xr.DataArray


def compile_twostream_parameters(
    forward_scattering_coefficients: xr.DataArray,
    backward_scattering_coefficients: xr.DataArray,
    absorption_coefficients: xr.DataArray,
    path_length: xr.DataArray,
) -> TwoStreamParameters:
    two_stream_scattering_parameters: TwoStreamScatteringParameters = (
        calculate_twostream_scattering_parameters(
            forward_scattering_coefficients=forward_scattering_coefficients,
            backward_scattering_coefficients=backward_scattering_coefficients,
        )
    )

    optical_depth: NDArray[np.float64] = attenuation_coefficients_to_optical_depths(
        attenuation_coefficients=absorption_coefficients, path_length=path_length
    )

    return TwoStreamParameters(
        scattering_asymmetry_parameter=two_stream_scattering_parameters.scattering_asymmetry_parameter,
        single_scattering_albedo=two_stream_scattering_parameters.single_scattering_albedo,
        optical_depth=optical_depth,
    )
