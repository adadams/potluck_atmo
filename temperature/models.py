from collections.abc import Sequence

import numba
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


def piette(
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
    log_pressures: NDArray[np.float64],
) -> NDArray[np.float64]:
    log_pressure_nodes = np.array([-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5])

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
        T_2p5,
        log_pressure_nodes=log_pressure_nodes,
        log_pressures=log_pressures,
        smoothing_parameter=0.3,
    )


def proper_piette(
    T_0p5: float,
    delta_T_2p5_T_2: float,
    delta_T_2_T_1p5: float,
    delta_T_1p5_T_1: float,
    delta_T_1_T_0p5: float,
    delta_T_0p5_T_0: float,
    delta_T_0_T_m1: float,
    delta_T_m1_T_m2: float,
    delta_T_m2_T_m3: float,
    delta_T_m3_T_m4: float,
    log_pressures: NDArray[np.float64],
):
    log_pressure_nodes = np.array([-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5])

    # start from log P = 0.5, move down to -4, then stitch that onto an array that starts at 0.5 and moves up
    temperatures: np.ndarray = np.array(
        [
            T_0p5
            - delta_T_0p5_T_0
            - delta_T_0_T_m1
            - delta_T_m1_T_m2
            - delta_T_m2_T_m3
            - delta_T_m3_T_m4,
            T_0p5
            - delta_T_0p5_T_0
            - delta_T_0_T_m1
            - delta_T_m1_T_m2
            - delta_T_m2_T_m3,
            T_0p5 - delta_T_0p5_T_0 - delta_T_0_T_m1 - delta_T_m1_T_m2,
            T_0p5 - delta_T_0p5_T_0 - delta_T_0_T_m1,
            T_0p5,
            T_0p5 + delta_T_1_T_0p5,
            T_0p5 + delta_T_1_T_0p5 + delta_T_1p5_T_1,
            T_0p5 + delta_T_1_T_0p5 + delta_T_1p5_T_1 + delta_T_2_T_1p5,
            T_0p5
            + delta_T_1_T_0p5
            + delta_T_1p5_T_1
            + delta_T_2_T_1p5
            + delta_T_2p5_T_2,
        ]
    )

    return general_piette_function(
        *temperatures,
        log_pressure_nodes=log_pressure_nodes,
        log_pressures=log_pressures,
        smoothing_parameter=0.3,
    )


@numba.jit(nopython=True)
def create_monotonic_temperature_profile_from_samples_numba(
    lower_bound,
    upper_bound,
    reference_index,
    initial_temp_sample,
    proportions_down,
    proportions_up,
):
    """
    Creates a monotonically increasing temperature profile within the given bounds
    using pre-sampled uniform values.

    Args:
        lower_bound (float): The lower bound of the temperature range.
        upper_bound (float): The upper bound of the temperature range.
        reference_index (int): The index of the reference temperature (0-based).
        initial_temp_sample (float): A uniform sample for the reference temperature [0, 1].
        proportions_down (np.ndarray): Uniform samples [0, 1] for proportions below the reference.
        proportions_up (np.ndarray): Uniform samples [0, 1] for proportions above the reference.

    Returns:
        numpy.ndarray: An array of 10 temperature values corresponding to the
                       ordered pressure points (low_5 to high_1).
    """
    num_points = 10
    temperatures = np.empty(num_points, dtype=np.float64)

    # 1. Determine the reference temperature
    reference_temp = lower_bound + initial_temp_sample * (upper_bound - lower_bound)
    temperatures[reference_index] = reference_temp

    # 2. Sample downwards from the reference temperature
    current_temp = reference_temp
    remaining_downward_range = current_temp - lower_bound
    for i in range(reference_index - 1, -1, -1):
        if remaining_downward_range <= 0:
            decrement = 0.0
        else:
            proportion = proportions_down[reference_index - 1 - i]
            decrement = proportion * remaining_downward_range
            remaining_downward_range -= decrement
        temperatures[i] = current_temp - decrement
        current_temp = temperatures[i]

    # 3. Sample upwards from the reference temperature
    current_temp = reference_temp
    remaining_upward_range = upper_bound - current_temp
    for i in range(reference_index + 1, num_points):
        if remaining_upward_range <= 0:
            increment = 0.0
        else:
            proportion = proportions_up[i - (reference_index + 1)]
            increment = proportion * remaining_upward_range
            remaining_upward_range -= increment
        temperatures[i] = current_temp + increment
        current_temp = temperatures[i]

    return temperatures


if __name__ == "__main__":
    lower_temp = 75.0
    upper_temp = 4000.0
    reference_index = 5
    num_samples = 10

    for i in range(num_samples):
        # Generate separate uniform samples
        initial_temp_sample = np.random.uniform(0, 1)
        proportions_down = np.random.uniform(0, 1, size=reference_index)
        proportions_up = np.random.uniform(
            0, 1, size=(num_samples - 1) - reference_index
        )

        profile = create_monotonic_temperature_profile_from_samples_numba(
            lower_temp,
            upper_temp,
            reference_index,
            initial_temp_sample,
            proportions_down,
            proportions_up,
        )
        print(f"Sample {i + 1}: {profile}")

        is_monotonic = all(
            profile[j] <= profile[j + 1] for j in range(len(profile) - 1)
        )
        within_bounds = all(lower_temp <= temp <= upper_temp for temp in profile)
        print(f"   Monotonic: {is_monotonic}, Within Bounds: {within_bounds}\n")
