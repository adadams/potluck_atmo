from typing import NamedTuple

import numpy as np


class TwoStreamScatteringCoefficients(NamedTuple):
    forward_scattering_coefficients: np.ndarray[np.float64]
    backward_scattering_coefficients: np.ndarray[np.float64]
