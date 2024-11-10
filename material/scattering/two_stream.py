import numpy as np
from numpy.typing import NDArray

from material.scattering.types import ScatteringParameters


def calculate_scattering_parameters(
    forward_scattering_coefficient: NDArray[np.float64],
    backward_scattering_coefficient: NDArray[np.float64],
    absorption_coefficient: NDArray[np.float64],
) -> ScatteringParameters:
    total_scattering_coefficient: NDArray[np.float64] = (
        forward_scattering_coefficient + backward_scattering_coefficient
    )

    total_extinction_coefficient: NDArray[np.float64] = (
        absorption_coefficient + total_scattering_coefficient
    )

    g: NDArray[np.float64] = (
        forward_scattering_coefficient - backward_scattering_coefficient
    ) / total_scattering_coefficient
    w0: NDArray[np.float64] = (
        total_scattering_coefficient / total_extinction_coefficient
    )

    return ScatteringParameters(g=g, w0=w0)
