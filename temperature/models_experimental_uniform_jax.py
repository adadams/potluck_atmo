import time
from functools import partial

import jax
import jax.numpy as jnp

# Original Numba version (from your file, for comparison)
import numba
import numpy as np
from scipy.interpolate import PchipInterpolator as monotonic_interpolation_scipy
from scipy.ndimage import gaussian_filter1d as gaussian_smoothing_scipy


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


# CORRECTED: Added parentheses for static_argnums
jax_jit_with_5_static_argnums = partial(jax.jit, static_argnums=(5,))


@jax_jit_with_5_static_argnums  # reference_index is the 6th argument (0-indexed)
def create_monotonic_temperature_profile_from_samples_jax(
    initial_temp_sample: jnp.ndarray,
    proportions_down: jnp.ndarray,
    proportions_up: jnp.ndarray,
    lower_bound: jnp.ndarray,
    upper_bound: jnp.ndarray,
    reference_index: int,  # This will be treated as a static argument
) -> jnp.ndarray:
    """
    Creates a monotonically increasing temperature profile within the given bounds
    using pre-sampled uniform values. JAX-accelerated.
    """
    num_points = 10
    temperatures = jnp.zeros(num_points, dtype=jnp.float64)

    # 1. Determine the reference temperature
    reference_temp = lower_bound + initial_temp_sample * (upper_bound - lower_bound)
    temperatures = temperatures.at[reference_index].set(reference_temp)

    # 2. Sample downwards from the reference temperature using lax.scan
    def scan_down_fn(carry, proportion):
        current_temp, remaining_downward_range = carry
        decrement = jnp.where(
            remaining_downward_range <= 0, 0.0, proportion * remaining_downward_range
        )
        new_remaining_downward_range = remaining_downward_range - decrement
        new_temp = current_temp - decrement
        return (new_temp, new_remaining_downward_range), new_temp

    proportions_down_reversed = proportions_down[::-1]
    initial_carry_down = (reference_temp, reference_temp - lower_bound)
    _, temps_down = jax.lax.scan(
        scan_down_fn, initial_carry_down, proportions_down_reversed
    )

    # Assign scanned temperatures to the correct positions in the profile
    temperatures = temperatures.at[:reference_index].set(temps_down[::-1])

    # 3. Sample upwards from the reference temperature using lax.scan
    def scan_up_fn(carry, proportion):
        current_temp, remaining_upward_range = carry
        increment = jnp.where(
            remaining_upward_range <= 0, 0.0, proportion * remaining_upward_range
        )
        new_remaining_upward_range = remaining_upward_range - increment
        new_temp = current_temp + increment
        return (new_temp, new_remaining_upward_range), new_temp

    initial_carry_up = (reference_temp, upper_bound - reference_temp)
    _, temps_up = jax.lax.scan(scan_up_fn, initial_carry_up, proportions_up)

    # Assign scanned temperatures to the correct positions in the profile
    temperatures = temperatures.at[reference_index + 1 :].set(temps_up)

    return temperatures


# This is the actual CPU-bound function that will be called by jax.pure_callback
def _general_piette_function_numpy_impl(
    temperature_nodes_np: np.ndarray,
    log_pressures_np: np.ndarray,
    smoothing_parameter: float,
) -> np.ndarray:
    """
    Performs interpolation and smoothing using SciPy (on CPU).
    This function should only receive NumPy arrays.
    """
    log_pressure_nodes_array = np.array(
        [-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5], dtype=np.float64
    )

    # PchipInterpolator (SciPy)
    interpolated_function = monotonic_interpolation_scipy(
        log_pressure_nodes_array, temperature_nodes_np
    )
    interpolated_temps_np = interpolated_function(log_pressures_np)

    # Gaussian Smoothing (SciPy)
    TP_profile_np = gaussian_smoothing_scipy(
        interpolated_temps_np, sigma=smoothing_parameter
    )
    return TP_profile_np


# This is the JAX-facing wrapper for the CPU-bound operations
def general_piette_function_cpu_ops(
    temperature_nodes: jnp.ndarray,
    log_pressures: jnp.ndarray,
    smoothing_parameter: float,
) -> jnp.ndarray:
    """
    Wraps _general_piette_function_numpy_impl using jax.pure_callback
    to bridge between JAX and NumPy/SciPy.
    """
    # Define the output shape and dtype that the callback will return
    # This must be static and match the actual output of the numpy function
    output_shape = log_pressures.shape
    output_dtype = (
        temperature_nodes.dtype
    )  # Assuming output dtype matches input temperature dtype

    # jax.pure_callback requires the Python function to accept NumPy arrays.
    # It also requires you to specify the output shape and dtype.
    # CORRECTED: Changed 'result_shape_dtype' to 'result_shape_dtypes'
    return jax.pure_callback(
        _general_piette_function_numpy_impl,
        result_shape_dtypes=jax.ShapeDtypeStruct(output_shape, output_dtype),
        temperature_nodes_np=temperature_nodes,
        log_pressures_np=log_pressures,
        smoothing_parameter=smoothing_parameter,
    )


jax_jit_with_6_static_argnums = partial(jax.jit, static_argnums=(6,))


@jax_jit_with_6_static_argnums  # reference_index is the 7th argument (0-indexed)
def piette_jax(
    initial_temp_sample: jnp.ndarray,
    proportions_down: jnp.ndarray,
    proportions_up: jnp.ndarray,
    log_pressures: jnp.ndarray,
    lower_bound: jnp.ndarray,
    upper_bound: jnp.ndarray,
    reference_index: int = 5,  # This will be treated as a static argument
) -> jnp.ndarray:
    """
    Combines node generation (JAX) and profile generation (SciPy for interpolation/smoothing).
    Note: Interpolation and smoothing steps will be run on CPU via jax.pure_callback.
    """
    temperature_nodes = create_monotonic_temperature_profile_from_samples_jax(
        initial_temp_sample,
        proportions_down,
        proportions_up,
        lower_bound,
        upper_bound,
        reference_index,  # Pass reference_index as a static argument
    )

    # Call the CPU-bound function for interpolation and smoothing via the JAX wrapper
    return general_piette_function_cpu_ops(
        temperature_nodes,
        log_pressures=log_pressures,
        smoothing_parameter=0.3,
    )


# Original SciPy based functions for comparison
def general_piette_function_scipy(
    temperature_nodes: np.ndarray,
    log_pressures: np.ndarray,
    smoothing_parameter: float,
):
    """
    Performs interpolation and smoothing. Smoothing is done in Python (SciPy).
    """
    log_pressure_nodes_array = np.array([-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5])

    interpolated_function = monotonic_interpolation_scipy(
        log_pressure_nodes_array, temperature_nodes
    )

    TP_profile = gaussian_smoothing_scipy(
        interpolated_function(log_pressures), sigma=smoothing_parameter
    )
    return TP_profile


def piette_numba_scipy(
    initial_temp_sample: float,
    proportions_down: np.ndarray,
    proportions_up: np.ndarray,
    log_pressures: np.ndarray,
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

    return general_piette_function_scipy(
        temperature_nodes,
        log_pressures=log_pressures,
        smoothing_parameter=0.3,
    )


if __name__ == "__main__":
    lower_temp = 75.0
    upper_temp = 4000.0
    reference_index = 5  # This value is now treated as static by JAX
    num_samples = 100
    log_pressure_nodes_array = np.array([-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5])
    log_pressures = np.linspace(
        log_pressure_nodes_array.min(), log_pressure_nodes_array.max(), 100
    )
    num_nodes = 10

    print("--- Running Numba + SciPy Version ---")
    start_time_numba = time.time()
    for _ in range(num_samples):
        initial_temp_sample_np = np.random.uniform(0, 1)
        proportions_down_np = np.random.uniform(0, 1, size=reference_index)
        proportions_up_np = np.random.uniform(
            0, 1, size=num_nodes - 1 - reference_index
        )

        _ = piette_numba_scipy(
            initial_temp_sample_np,
            proportions_down_np,
            proportions_up_np,
            log_pressures,
            lower_temp,
            upper_temp,
            reference_index,
        )
    end_time_numba = time.time()
    total_time_numba = end_time_numba - start_time_numba
    print(
        f"Time for {num_samples} samples (Numba+SciPy): {total_time_numba:.4f} seconds"
    )

    print("\n--- Running JAX Version (PCHIP and Gaussian on CPU via pure_callback) ---")
    key = jax.random.PRNGKey(0)  # Initialize a JAX random key

    start_time_jax = time.time()
    for i in range(num_samples):
        key, subkey_initial = jax.random.split(key)
        key, subkey_down = jax.random.split(key)
        key, subkey_up = jax.random.split(key)

        initial_temp_sample_jax = jax.random.uniform(
            subkey_initial, shape=(), dtype=jnp.float64
        )
        proportions_down_jax = jax.random.uniform(
            subkey_down, shape=(reference_index,), dtype=jnp.float64
        )
        proportions_up_jax = jax.random.uniform(
            subkey_up, shape=(num_nodes - 1 - reference_index,), dtype=jnp.float64
        )

        lower_temp_jax = jnp.array(lower_temp, dtype=jnp.float64)
        upper_temp_jax = jnp.array(upper_temp, dtype=jnp.float64)
        log_pressures_jax = jnp.array(log_pressures, dtype=jnp.float64)

        _ = piette_jax(
            initial_temp_sample_jax,
            proportions_down_jax,
            proportions_up_jax,
            log_pressures_jax,
            lower_temp_jax,
            upper_temp_jax,
            reference_index,  # Pass reference_index here
        ).block_until_ready()  # Ensure computation finishes before timing

    end_time_jax = time.time()
    total_time_jax = end_time_jax - start_time_jax
    print(
        f"Time for {num_samples} samples (JAX, PCHIP and Gaussian on CPU): {total_time_jax:.4f} seconds"
    )

    # --- Basic Checks on a single sample (JAX version) ---
    print("\n--- Basic Checks (JAX Version, PCHIP and Gaussian on CPU) ---")
    key, subkey_initial = jax.random.split(key)
    key, subkey_down = jax.random.split(key)
    key, subkey_up = jax.random.split(key)

    initial_temp_sample_jax = jax.random.uniform(
        subkey_initial, shape=(), dtype=jnp.float64
    )
    proportions_down_jax = jax.random.uniform(
        subkey_down, shape=(reference_index,), dtype=jnp.float64
    )
    proportions_up_jax = jax.random.uniform(
        subkey_up, shape=(num_nodes - 1 - reference_index,), dtype=jnp.float64
    )

    lower_temp_jax = jnp.array(lower_temp, dtype=jnp.float64)
    upper_temp_jax = jnp.array(upper_temp, dtype=jnp.float64)
    log_pressures_jax = jnp.array(log_pressures, dtype=jnp.float64)

    profile_jax = piette_jax(
        initial_temp_sample_jax,
        proportions_down_jax,
        proportions_up_jax,
        log_pressures_jax,
        lower_temp_jax,
        upper_temp_jax,
        reference_index,
    )
    print(f"\nSample profile (JAX): {profile_jax[:5]}...{profile_jax[-5:]}")

    print(f"{profile_jax=}")
    # Check for NaN values
    if jnp.any(jnp.isnan(profile_jax)):
        print("WARNING: JAX Profile contains NaN values!")
    else:
        print("JAX Profile does not contain NaN values.")

    # Check bounds (approximate - interpolation can slightly overshoot)
    if jnp.all(
        (profile_jax >= lower_temp_jax - 1) & (profile_jax <= upper_temp_jax + 1)
    ):
        print("JAX Profile stays approximately within bounds.")
    else:
        print(
            f"WARNING: JAX Profile exceeds bounds! Min: {profile_jax.min()}, Max: {profile_jax.max()}"
        )
    print(
        f"Log Pressures range used for interpolation: {log_pressures_jax.min()} to {log_pressures_jax.max()}"
    )

    # --- Compare Numba and JAX outputs for a single set of inputs ---
    print("\n--- Comparison of Numba and JAX outputs ---")
    np.random.seed(42)  # For reproducibility
    initial_temp_sample_common = np.random.uniform(0, 1)
    proportions_down_common = np.random.uniform(0, 1, size=reference_index)
    proportions_up_common = np.random.uniform(
        0, 1, size=num_nodes - 1 - reference_index
    )

    # Numba version
    profile_numba_scipy_compare = piette_numba_scipy(
        initial_temp_sample_common,
        proportions_down_common,
        proportions_up_common,
        log_pressures,
        lower_temp,
        upper_temp,
        reference_index,
    )

    # JAX version (convert inputs to JAX arrays)
    initial_temp_sample_jax_compare = jnp.array(
        initial_temp_sample_common, dtype=jnp.float64
    )
    proportions_down_jax_compare = jnp.array(proportions_down_common, dtype=jnp.float64)
    proportions_up_jax_compare = jnp.array(proportions_up_common, dtype=jnp.float64)
    log_pressures_jax_compare = jnp.array(log_pressures, dtype=jnp.float64)
    lower_temp_jax_compare = jnp.array(lower_temp, dtype=jnp.float64)
    upper_temp_jax_compare = jnp.array(upper_temp, dtype=jnp.float64)

    profile_jax_compare = piette_jax(
        initial_temp_sample_jax_compare,
        proportions_down_jax_compare,
        proportions_up_jax_compare,
        log_pressures_jax_compare,
        lower_temp_jax_compare,
        upper_temp_jax_compare,
        reference_index,
    )

    profile_jax_compare_np = np.array(profile_jax_compare)

    are_close = np.allclose(
        profile_numba_scipy_compare, profile_jax_compare_np, rtol=1e-5, atol=1e-8
    )
    print(f"Are Numba+SciPy and JAX profiles approximately equal? {are_close}")

    if not are_close:
        diff = np.abs(profile_numba_scipy_compare - profile_jax_compare_np)
        print(f"Maximum absolute difference: {diff.max()}")
        print(f"Mean absolute difference: {diff.mean()}")
