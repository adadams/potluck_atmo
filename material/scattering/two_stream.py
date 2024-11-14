import numpy as np
from numpy.typing import NDArray

from material.types import TwoStreamScatteringParameters
from xarray_functional_wrappers import Dimensionalize
from xarray_serialization import PressureType, WavelengthType


@Dimensionalize(
    argument_dimensions=(
        (WavelengthType, PressureType),
        (WavelengthType, PressureType),
        (WavelengthType, PressureType),
    ),
    result_dimensions=((WavelengthType, PressureType), (WavelengthType, PressureType)),
)
def calculate_two_stream_scattering_parameters(
    forward_scattering_coefficients: NDArray[np.float64],
    backward_scattering_coefficients: NDArray[np.float64],
    absorption_coefficients: NDArray[np.float64],
) -> TwoStreamScatteringParameters:
    total_scattering_coefficient: NDArray[np.float64] = (
        forward_scattering_coefficients + backward_scattering_coefficients
    )

    total_extinction_coefficient: NDArray[np.float64] = (
        absorption_coefficients + total_scattering_coefficient
    )

    scattering_asymmetry_parameter: NDArray[np.float64] = (
        forward_scattering_coefficients - backward_scattering_coefficients
    ) / total_scattering_coefficient

    single_scattering_albedo: NDArray[np.float64] = (
        total_scattering_coefficient / total_extinction_coefficient
    )

    return TwoStreamScatteringParameters(
        scattering_asymmetry_parameter=scattering_asymmetry_parameter,
        single_scattering_albedo=single_scattering_albedo,
    )
