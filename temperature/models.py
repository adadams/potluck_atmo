from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator as monotonic_interpolation
from scipy.ndimage import gaussian_filter1d as gaussian_smoothing


def general_piette_function(
    *temperatures: Sequence[float],
    log_pressure_nodes: NDArray[np.float64],
    log_pressures: NDArray[np.float64],
    smoothing_parameter: float,
):
    interpolated_function = monotonic_interpolation(log_pressure_nodes, temperatures)

    TP_profile = gaussian_smoothing(
        interpolated_function(log_pressures), sigma=smoothing_parameter
    )
    return TP_profile


def intentionally_wrong_piette(
    T_m4: float,
    T_m3: float,
    T_m2: float,
    T_m1: float,
    T_0: float,
    T_0p5: float,
    T_1: float,
    T_1p5: float,
    T_2: float,
    T_2p5: float,
    pressures: NDArray[np.float64],
) -> NDArray[np.float64]:
    log_pressure_nodes = np.array([-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2])

    return general_piette_function(
        T_m4,
        T_m3,
        T_m2,
        T_m1,
        T_0,
        T_0p5,
        T_1,
        T_1p5,
        T_2,
        log_pressure_nodes=log_pressure_nodes,
        log_pressures=pressures,
        smoothing_parameter=0.3,
    )
