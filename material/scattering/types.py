from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class TwoStreamScatteringCoefficients(NamedTuple):
    forward_scattering_coefficients: NDArray[np.float64]
    backward_scattering_coefficients: NDArray[np.float64]
