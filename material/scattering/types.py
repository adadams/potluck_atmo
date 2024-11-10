import msgspec
import numpy as np
from numpy.typing import NDArray


class TwoStreamScatteringCrossSections(msgspec.Struct):
    forward_scattering_crosssections: NDArray[np.float64]
    backward_scattering_crosssections: NDArray[np.float64]


class TwoStreamScatteringCoefficients(msgspec.Struct):
    forward_scattering_coefficients: NDArray[np.float64]
    backward_scattering_coefficients: NDArray[np.float64]


class TwoStreamMaterial(msgspec.Struct):
    name: str
    scattering_coefficients: TwoStreamScatteringCoefficients
    absorption_coefficients: NDArray[np.float64]


class ScatteringParameters(msgspec.Struct):
    g: NDArray[np.float64]
    w0: NDArray[np.float64]
