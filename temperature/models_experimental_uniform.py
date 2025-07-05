from collections.abc import Callable
from functools import partial

import numpy as np
from msgspec.structs import astuple
from numba import njit
from scipy.interpolate import PchipInterpolator as monotonic_interpolation
from scipy.ndimage import gaussian_filter1d as gaussian_smoothing

from basic_types import NormalizedValue, Shape, TemperatureValue
from temperature.protocols import TemperatureModelInputs


class PietteTemperatureModelInputs(TemperatureModelInputs):
    photospheric_scaled_3bar_temperature: NormalizedValue
    scaled_1bar_temperature: NormalizedValue
    scaled_0p1bar_temperature: NormalizedValue
    scaled_0p01bar_temperature: NormalizedValue
    scaled_0p001bar_temperature: NormalizedValue
    scaled_0p0001bar_temperature: NormalizedValue
    scaled_10bar_temperature: NormalizedValue
    scaled_30bar_temperature: NormalizedValue
    scaled_100bar_temperature: NormalizedValue
    scaled_300bar_temperature: NormalizedValue


@njit(cache=True)
def create_monotonic_temperature_profile_from_samples(
    photospheric_scaled_3bar_temperature: NormalizedValue,
    scaled_1bar_temperature: NormalizedValue,
    scaled_0p1bar_temperature: NormalizedValue,
    scaled_0p01bar_temperature: NormalizedValue,
    scaled_0p001bar_temperature: NormalizedValue,
    scaled_0p0001bar_temperature: NormalizedValue,
    scaled_10bar_temperature: NormalizedValue,
    scaled_30bar_temperature: NormalizedValue,
    scaled_100bar_temperature: NormalizedValue,
    scaled_300bar_temperature: NormalizedValue,
    lower_bound: TemperatureValue,
    upper_bound: TemperatureValue,
):
    reference_index: int = 5
    number_of_pressure_nodes: int = 10

    proportions_down: np.ndarray[
        (number_of_pressure_nodes - reference_index,), NormalizedValue
    ] = np.array(
        [
            scaled_1bar_temperature,
            scaled_0p1bar_temperature,
            scaled_0p01bar_temperature,
            scaled_0p001bar_temperature,
            scaled_0p0001bar_temperature,
        ]
    )
    proportions_up: np.ndarray[
        ((number_of_pressure_nodes - reference_index) - 1,), NormalizedValue
    ] = np.array(
        [
            scaled_10bar_temperature,
            scaled_30bar_temperature,
            scaled_100bar_temperature,
            scaled_300bar_temperature,
        ]
    )

    number_of_pressure_nodes: int = len(proportions_down) + len(proportions_up) + 1

    temperatures: np.ndarray[np.float64] = np.empty(
        number_of_pressure_nodes, dtype=np.float64
    )

    # 1. Determine the reference temperature
    reference_temp = lower_bound + photospheric_scaled_3bar_temperature * (
        upper_bound - lower_bound
    )
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
    for i in range(reference_index + 1, number_of_pressure_nodes):
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
    log_pressure_nodes: np.ndarray[Shape, np.float64],
    temperature_nodes: np.ndarray[Shape, TemperatureValue],
    profile_log_pressures: np.ndarray[np.float64],
    smoothing_parameter: float = 0.3,
):
    interpolated_function: Callable[
        [np.ndarray[Shape, np.float64], float], np.ndarray[Shape, TemperatureValue]
    ] = monotonic_interpolation(log_pressure_nodes, temperature_nodes)

    TP_profile = gaussian_smoothing(
        interpolated_function(profile_log_pressures), sigma=smoothing_parameter
    )

    return TP_profile


# @njit(cache=True)
def generate_piette_model(
    piette_parameters: PietteTemperatureModelInputs,
    lower_bound: TemperatureValue,
    upper_bound: TemperatureValue,
):
    log_pressure_nodes: np.ndarray[(10,), np.float64] = np.array(
        [-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5]
    )
    number_of_pressure_nodes: int = len(log_pressure_nodes)

    temperature_nodes: np.ndarray[(number_of_pressure_nodes,), TemperatureValue] = (
        create_monotonic_temperature_profile_from_samples(
            *astuple(piette_parameters), lower_bound, upper_bound
        )
    )

    return partial(
        general_piette_function,
        log_pressure_nodes=log_pressure_nodes,
        temperature_nodes=temperature_nodes,
    )
