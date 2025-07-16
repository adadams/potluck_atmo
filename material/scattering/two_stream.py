from dataclasses import astuple

import numpy as np

from basic_types import PressureDimension, WavelengthDimension
from material.types import TwoStreamScatteringParameters
from xarray_functional_wrappers import Dimensionalize


@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
    ),
    result_dimensions=(
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
    ),
)
def calculate_two_stream_scattering_parameters(
    forward_scattering_coefficients: np.ndarray[np.float64],
    backward_scattering_coefficients: np.ndarray[np.float64],
    absorption_coefficients: np.ndarray[np.float64],
) -> TwoStreamScatteringParameters:
    total_scattering_coefficient: np.ndarray[np.float64] = (
        forward_scattering_coefficients + backward_scattering_coefficients
    )

    total_extinction_coefficient: np.ndarray[np.float64] = (
        absorption_coefficients + total_scattering_coefficient
    )

    scattering_asymmetry_parameter: np.ndarray[np.float64] = (
        forward_scattering_coefficients - backward_scattering_coefficients
    ) / total_scattering_coefficient

    single_scattering_albedo: np.ndarray[np.float64] = (
        total_scattering_coefficient / total_extinction_coefficient
    )

    return astuple(
        TwoStreamScatteringParameters(
            scattering_asymmetry_parameter=scattering_asymmetry_parameter,
            single_scattering_albedo=single_scattering_albedo,
        )
    )
