import time

import numba
import numpy as np
from scipy.interpolate import PchipInterpolator as monotonic_interpolation
from scipy.ndimage import gaussian_filter1d as gaussian_smoothing


@numba.jit(nopython=True, cache=True)
def create_monotonic_temperature_profile_from_samples_numba(
    initial_temp_sample,
    proportions_down,
    proportions_up,
    lower_bound,
    upper_bound,
    reference_index,
):
    """
    Creates a monotonically increasing temperature profile within the given bounds
    using pre-sampled uniform values.  Numba-accelerated.
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


def general_piette_function(
    temperature_nodes: np.ndarray,
    log_pressures: np.ndarray,  # Use np.ndarray directly as type hint
    smoothing_parameter: float,
):
    """
    Performs interpolation and smoothing.  Smoothing is done in Python.
    """
    # Assuming log_pressure_nodes_array is always the same for this function's purpose
    log_pressure_nodes_array = np.array([-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5])

    interpolated_function = monotonic_interpolation(
        log_pressure_nodes_array, temperature_nodes
    )

    TP_profile = gaussian_smoothing(
        interpolated_function(log_pressures), sigma=smoothing_parameter
    )
    return TP_profile


def piette(
    initial_temp_sample: float,
    proportions_down: np.ndarray,  # Use np.ndarray directly as type hint
    proportions_up: np.ndarray,  # Use np.ndarray directly as type hint
    log_pressures: np.ndarray,  # Use np.ndarray directly as type hint
    lower_bound: float,
    upper_bound: float,
    reference_index: int = 5,
):
    """
    Combines node generation (Numba) and profile generation (Python/SciPy).
    """

    temperature_nodes = create_monotonic_temperature_profile_from_samples_numba(
        initial_temp_sample,
        proportions_down,
        proportions_up,
        lower_bound,
        upper_bound,
        reference_index,
    )

    return general_piette_function(
        temperature_nodes,
        log_pressures=log_pressures,
        smoothing_parameter=0.3,
    )


if __name__ == "__main__":
    lower_temp = 75.0
    upper_temp = 4000.0
    reference_index = 5
    num_samples = 100
    log_pressure_nodes_array = np.array([-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5])
    log_pressures = np.linspace(
        log_pressure_nodes_array.min(), log_pressure_nodes_array.max(), 100
    )
    num_nodes = 10

    # --- Timing ---
    start_time = time.time()
    for _ in range(num_samples):
        # Generate all uniform samples
        initial_temp_sample = np.random.uniform(0, 1)
        proportions_down = np.random.uniform(0, 1, size=reference_index)
        proportions_up = np.random.uniform(0, 1, size=num_nodes - 1 - reference_index)

        start_time_inner = time.time()
        profile = piette(
            initial_temp_sample,
            proportions_down,
            proportions_up,
            log_pressures,
            lower_temp,
            upper_temp,
            reference_index,
        )
        end_time_inner = time.time()
        total_time_inner = end_time_inner - start_time_inner
        print(f"Time for sample: {total_time_inner:.4f} seconds")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Time for {num_samples} samples: {total_time:.4f} seconds")

    # --- Basic Checks on a single sample ---
    initial_temp_sample = np.random.uniform(0, 1)
    proportions_down = np.random.uniform(0, 1, size=reference_index)
    proportions_up = np.random.uniform(0, 1, size=num_nodes - 1 - reference_index)
    profile = piette(
        initial_temp_sample,
        proportions_down,
        proportions_up,
        log_pressures,
        lower_temp,
        upper_temp,
        reference_index,
    )
    print(
        f"\nSample profile: {profile[:5]}...{profile[-5:]}"
    )  # Print both ends of the profile for better check

    print(f"{profile=}")
    # Check for NaN values
    if np.any(np.isnan(profile)):
        print("WARNING: Profile contains NaN values!")
    else:
        print("Profile does not contain NaN values.")

    # Check bounds (approximate - interpolation can slightly overshoot)
    if np.all((profile >= lower_temp - 1) & (profile <= upper_temp + 1)):
        print("Profile stays approximately within bounds.")
    else:
        print(
            f"WARNING: Profile exceeds bounds! Min: {profile.min()}, Max: {profile.max()}"
        )
    print(
        f"Log Pressures range used for interpolation: {log_pressures.min()} to {log_pressures.max()}"
    )
