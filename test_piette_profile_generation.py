from time import time

import numpy as np

from basic_types import TemperatureValue
from temperature.models import (
    PietteTemperatureModelInputs,
    generate_piette_model,
)
from temperature.protocols import TemperatureModel

if __name__ == "__main__":
    lower_temp: TemperatureValue = 75.0
    upper_temp: TemperatureValue = 3975.0
    num_samples: int = 100
    log_pressure_nodes_array: np.ndarray[np.float64] = np.array(
        [-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5]
    )
    num_nodes: int = len(log_pressure_nodes_array)
    reference_index: int = 5  # Index of the reference photospheric pressure, ~3 bar
    log_pressures: np.ndarray[np.float64] = np.linspace(
        log_pressure_nodes_array.min(), log_pressure_nodes_array.max(), 100
    )

    # --- Timing ---
    start_time = time()  # noqa: F821
    for _ in range(num_samples):
        photospheric_scaled_3bar_temperature = np.random.uniform(0, 1)
        proportions_down = np.random.uniform(0, 1, size=reference_index)
        proportions_up = np.random.uniform(0, 1, size=num_nodes - 1 - reference_index)

        piette_model_inputs: PietteTemperatureModelInputs = (
            PietteTemperatureModelInputs(
                photospheric_scaled_3bar_temperature, *proportions_down, *proportions_up
            )
        )

        piette: TemperatureModel = generate_piette_model(
            piette_model_inputs, lower_temp, upper_temp
        )

        start_time_inner = time()
        profile = piette(profile_log_pressures=log_pressures)
        end_time_inner = time()
        total_time_inner = end_time_inner - start_time_inner
        print(f"Time for sample: {total_time_inner:.4f} seconds")
    end_time = time()
    total_time = end_time - start_time
    print(f"Time for {num_samples} samples: {total_time:.4f} seconds")

    # --- Basic Checks on a single sample ---
    photospheric_scaled_3bar_temperature = np.random.uniform(0, 1)
    proportions_down = np.random.uniform(0, 1, size=reference_index)
    proportions_up = np.random.uniform(0, 1, size=num_nodes - 1 - reference_index)

    piette_model_inputs: PietteTemperatureModelInputs = PietteTemperatureModelInputs(
        photospheric_scaled_3bar_temperature, *proportions_down, *proportions_up
    )

    piette: TemperatureModel = generate_piette_model(
        piette_model_inputs, lower_temp, upper_temp
    )

    profile = piette(profile_log_pressures=log_pressures)
    print(
        f"\nSample profile: {profile[:reference_index]}...{profile[-reference_index:]}"
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
        "Log Pressures range used for interpolation: "
        f"{log_pressures.min()} to {log_pressures.max()}"
    )
