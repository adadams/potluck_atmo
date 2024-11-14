from typing import NamedTuple

import msgspec
import numpy as np
from numpy.typing import NDArray

from material.scattering.types import TwoStreamScatteringCoefficients


class TwoStreamMaterial(msgspec.Struct):
    name: str
    scattering_coefficients: TwoStreamScatteringCoefficients
    absorption_coefficients: NDArray[np.float64]


# NOTE: these are meant as a cumulative struct, encompassing contributions from multiple materials.
class TwoStreamScatteringParameters(NamedTuple):
    scattering_asymmetry_parameter: NDArray[np.float64]  # g
    single_scattering_albedo: NDArray[np.float64]  # w0


class TwoStreamParameters(msgspec.Struct):
    scattering_asymmetry_parameter: NDArray[np.float64]  # g
    single_scattering_albedo: NDArray[np.float64]  # w0
    optical_depth: NDArray[np.float64]  # tau
