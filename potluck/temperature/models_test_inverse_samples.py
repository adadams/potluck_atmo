import numpy as np
from numba import njit


@njit(cache=True)
def get_uniform_samples_from_monotonic_nodes(
    temperature_nodes: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    reference_index: int = 5,
):
    """
    Given a set of 10 monotonic temperature nodes within bounds,
    returns the uniform samples (initial_temp_sample, proportions_down, proportions_up)
    that would yield these nodes using the generation logic.

    Args:
        temperature_nodes (np.ndarray): An array of 10 monotonically increasing
                                        temperature values.
        lower_bound (float): The lower bound of the temperature range.
        upper_bound (float): The upper bound of the temperature range.
        reference_index (int): The index of the reference temperature (0-based).

    Returns:
        tuple: A tuple containing:
            - initial_temp_sample (float): The uniform sample for the reference temperature [0, 1].
            - proportions_down (np.ndarray): Uniform samples [0, 1] for proportions below the reference.
            - proportions_up (np.ndarray): Uniform samples [0, 1] for proportions above the reference.

    Raises:
        ValueError: If input temperature_nodes are not monotonic or outside bounds,
                    or if calculations lead to division by zero unexpectedly.
    """
    num_points = len(temperature_nodes)
    if num_points != 10:
        raise ValueError("temperature_nodes must contain exactly 10 points.")
    if not np.all(
        (temperature_nodes >= lower_bound - 1e-9)
        & (temperature_nodes <= upper_bound + 1e-9)
    ):
        # Allow for tiny floating point errors
        raise ValueError(
            f"Temperature nodes are not within bounds [{lower_bound}, {upper_bound}]. Found: {temperature_nodes}"
        )
    if not np.all(temperature_nodes[:-1] <= temperature_nodes[1:] + 1e-9):
        # Allow for tiny floating point errors
        raise ValueError("Input temperature_nodes are not monotonically increasing.")

    # 1. Reconstruct initial_temp_sample from reference_temp
    reference_temp = temperature_nodes[reference_index]
    total_range = upper_bound - lower_bound
    if total_range <= 1e-9:  # Handle case where upper_bound == lower_bound
        initial_temp_sample = 0.0 if reference_temp <= lower_bound else 1.0
    else:
        initial_temp_sample = (reference_temp - lower_bound) / total_range
    initial_temp_sample = max(
        0.0, min(1.0, initial_temp_sample)
    )  # Clamp to [0, 1] for robustness

    # 2. Reconstruct proportions_down
    proportions_down = np.empty(reference_index, dtype=np.float64)
    current_temp = reference_temp
    remaining_downward_range = current_temp - lower_bound
    for k in range(reference_index):  # k is the index in proportions_down array
        node_idx = reference_index - k  # The higher temp node for this step
        prev_node_idx = node_idx - 1  # The lower temp node for this step

        decrement = current_temp - temperature_nodes[prev_node_idx]

        if remaining_downward_range <= 1e-9:  # If no more range to drop
            proportion = 0.0  # No further drop is possible
        else:
            proportion = decrement / remaining_downward_range

        proportions_down[k] = max(0.0, min(1.0, proportion))  # Clamp to [0, 1]

        current_temp = temperature_nodes[
            prev_node_idx
        ]  # Update current_temp for next iteration
        remaining_downward_range = current_temp - lower_bound  # Update remaining range

    # 3. Reconstruct proportions_up
    proportions_up = np.empty(num_points - 1 - reference_index, dtype=np.float64)
    current_temp = reference_temp
    remaining_upward_range = upper_bound - current_temp
    for k in range(
        num_points - 1 - reference_index
    ):  # k is the index in proportions_up array
        node_idx = reference_index + k  # The lower temp node for this step
        next_node_idx = node_idx + 1  # The higher temp node for this step

        increment = temperature_nodes[next_node_idx] - current_temp

        if remaining_upward_range <= 1e-9:  # If no more range to rise
            proportion = 0.0  # No further rise is possible
        else:
            proportion = increment / remaining_upward_range

        proportions_up[k] = max(0.0, min(1.0, proportion))  # Clamp to [0, 1]

        current_temp = temperature_nodes[
            next_node_idx
        ]  # Update current_temp for next iteration
        remaining_upward_range = upper_bound - current_temp  # Update remaining range

    return reference_temp, proportions_down, proportions_up


# --- Main execution block for testing ---
if __name__ == "__main__":
    lower_temp = 75.0
    upper_temp = 4000.0
    reference_index = 5
    num_nodes = 10
    num_tests = 5

    # Import the generation function for verification
    # (assuming it's in the same script or imported correctly)
    @njit
    def create_monotonic_temperature_nodes_from_samples_numba(
        lower_bound,
        upper_bound,
        reference_index,
        initial_temp_sample,
        proportions_down,
        proportions_up,
    ):
        num_points = 10
        temperatures = np.empty(num_points, dtype=np.float64)
        reference_temp = lower_bound + initial_temp_sample * (upper_bound - lower_bound)
        temperatures[reference_index] = reference_temp
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

    print("--- Testing Inverse Function ---")
    for i in range(num_tests):
        print(f"\n--- Test Case {i + 1} ---")
        # 1. Generate random samples
        orig_initial_temp_sample = np.random.uniform(0, 1)
        orig_proportions_down = np.random.uniform(0, 1, size=reference_index)
        orig_proportions_up = np.random.uniform(
            0, 1, size=num_nodes - 1 - reference_index
        )

        print(f"Original initial_temp_sample: {orig_initial_temp_sample:.4f}")
        print(f"Original proportions_down: {orig_proportions_down}")
        print(f"Original proportions_up: {orig_proportions_up}")

        # 2. Generate a temperature profile using the original function
        generated_nodes = create_monotonic_temperature_nodes_from_samples_numba(
            lower_temp,
            upper_temp,
            reference_index,
            orig_initial_temp_sample,
            orig_proportions_down,
            orig_proportions_up,
        )
        print(f"Generated Nodes: {generated_nodes}")

        # 3. Use the inverse function to get the samples back
        (
            reconstructed_initial_temp_sample,
            reconstructed_proportions_down,
            reconstructed_proportions_up,
        ) = get_uniform_samples_from_monotonic_nodes(
            generated_nodes,
            lower_temp,
            upper_temp,
            reference_index,
        )

        print(
            f"Reconstructed initial_temp_sample: {reconstructed_initial_temp_sample:.4f}"
        )
        print(f"Reconstructed proportions_down: {reconstructed_proportions_down}")
        print(f"Reconstructed proportions_up: {reconstructed_proportions_up}")

        # 4. Verify (allowing for small floating point differences)
        initial_temp_match = np.isclose(
            orig_initial_temp_sample, reconstructed_initial_temp_sample, atol=1e-7
        )
        down_match = np.allclose(
            orig_proportions_down, reconstructed_proportions_down, atol=1e-7
        )
        up_match = np.allclose(
            orig_proportions_up, reconstructed_proportions_up, atol=1e-7
        )

        print(f"Initial temp sample match: {initial_temp_match}")
        print(f"Proportions down match: {down_match}")
        print(f"Proportions up match: {up_match}")

        if initial_temp_match and down_match and up_match:
            print("--- Reconstruction SUCCESSFUL! ---")
        else:
            print("--- Reconstruction FAILED! ---")

    # Test with edge case: All temperatures at lower_bound
    print("\n--- Test Case: All at lower_bound ---")
    nodes_at_lower_bound = np.full(10, lower_temp)
    initial_temp_s, p_d, p_u = get_uniform_samples_from_monotonic_nodes(
        nodes_at_lower_bound, lower_temp, upper_temp, reference_index
    )
    print(f"Nodes: {nodes_at_lower_bound}")
    print(f"Reconstructed initial_temp_sample: {initial_temp_s}")
    print(f"Reconstructed proportions_down: {p_d}")
    print(f"Reconstructed proportions_up: {p_u}")
    print(
        f"Matches expected (0, [0...], [0...]): {np.isclose(initial_temp_s, 0.0) and np.all(p_d == 0.0) and np.all(p_u == 0.0)}"
    )

    # Test with edge case: All temperatures at upper_bound
    print("\n--- Test Case: All at upper_bound ---")
    nodes_at_upper_bound = np.full(10, upper_temp)
    initial_temp_s, p_d, p_u = get_uniform_samples_from_monotonic_nodes(
        nodes_at_upper_bound, lower_temp, upper_temp, reference_index
    )
    print(f"Nodes: {nodes_at_upper_bound}")
    print(f"Reconstructed initial_temp_sample: {initial_temp_s}")
    print(f"Reconstructed proportions_down: {p_d}")
    print(f"Reconstructed proportions_up: {p_u}")
    print(
        f"Matches expected (1, [0...], [0...]): {np.isclose(initial_temp_s, 1.0) and np.all(p_d == 0.0) and np.all(p_u == 0.0)}"
    )

    # Test with constant temperature (not at bounds)
    print("\n--- Test Case: Constant temp (middle) ---")
    constant_temp = (lower_temp + upper_temp) / 2
    nodes_constant = np.full(10, constant_temp)
    initial_temp_s, p_d, p_u = get_uniform_samples_from_monotonic_nodes(
        nodes_constant, lower_temp, upper_temp, reference_index
    )
    print(f"Nodes: {nodes_constant}")
    print(f"Reconstructed initial_temp_sample: {initial_temp_s}")
    print(f"Reconstructed proportions_down: {p_d}")
    print(f"Reconstructed proportions_up: {p_u}")
    print(
        f"Matches expected (0.5, [0...], [0...]): {np.isclose(initial_temp_s, 0.5) and np.all(p_d == 0.0) and np.all(p_u == 0.0)}"
    )

    # Test with a slight non-monotonicity (should raise error)
    print("\n--- Test Case: Non-monotonic (should error) ---")
    non_monotonic_nodes = np.array([75, 80, 78, 90, 100, 150, 200, 250, 300, 350])
    try:
        get_uniform_samples_from_monotonic_nodes(
            non_monotonic_nodes, lower_temp, upper_temp, reference_index
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test with out of bounds (should raise error)
    print("\n--- Test Case: Out of bounds (should error) ---")
    out_of_bounds_nodes = np.array([70, 80, 90, 100, 150, 200, 250, 300, 350, 4050])
    try:
        get_uniform_samples_from_monotonic_nodes(
            out_of_bounds_nodes, lower_temp, upper_temp, reference_index
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")

    apollo_temperature_node_dict: dict[str, float] = dict(
        T_m4=832.564035,
        T_m3=844.4078632,
        T_m2=937.2620338,
        T_m1=1060.564358,
        T_0=1082.55555,
        T_0p5=1209.112301,
        T_1=1279.292947,
        T_1p5=1333.242248,
        T_2=1499.892897,
        T_2p5=2311.077596,
    )
    apollo_temperature_nodes_as_array: np.ndarray = np.array(
        list(apollo_temperature_node_dict.values())
    )
    print(f"Apollo Nodes: {apollo_temperature_nodes_as_array}")
    reconstructed_sample_from_apollo = get_uniform_samples_from_monotonic_nodes(
        apollo_temperature_nodes_as_array, lower_temp, upper_temp, reference_index
    )
    print(f"Reconstructed sample from Apollo: {reconstructed_sample_from_apollo}")
