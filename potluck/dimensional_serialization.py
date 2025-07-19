from enum import ReprEnum

import numpy as np


class TwoStreamScatteringAngles(float, ReprEnum):
    FORWARD = 0.0
    BACKWARD = np.pi
