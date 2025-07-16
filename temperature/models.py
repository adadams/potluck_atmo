from collections.abc import Callable
from functools import partial
from typing import Tuple, TypeVar

import msgspec
import numpy as np
from msgspec.structs import astuple
from numba import njit
from scipy.interpolate import PchipInterpolator as monotonic_interpolation
from scipy.ndimage import gaussian_filter1d as gaussian_smoothing

from basic_types import (
    LogPressureValue,
    NormalizedValue,
    PositiveValue,
    TemperatureValue,
)
from temperature.protocols import TemperatureModel, TemperatureModelParameters

NumberofNodes = TypeVar("NumberofNodes", bound=Tuple[int, ...])  # numpy ndarray shape
NumberofModelPressures = TypeVar("NumberofModelPressures", bound=Tuple[int, ...])


class TemperatureBounds(msgspec.Struct):
    lower_temperature_bound: TemperatureValue
    upper_temperature_bound: TemperatureValue


class PietteTemperatureModelParameters(TemperatureModelParameters):
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
def create_monotonic_temperature_nodes_from_samples(
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
    lower_temperature_bound: TemperatureValue,  # usually fixed, set by e.g. opacity temperature range
    upper_temperature_bound: TemperatureValue,  # usually fixed, set by e.g. opacity temperature range
) -> np.ndarray[(10,), TemperatureValue]:
    """
    Enforces a temperature profile that does not decrease with increasing pressure.
    """

    number_of_pressure_nodes: int = 10
    photospheric_index: int = 5

    proportions_shallower: np.ndarray[
        (number_of_pressure_nodes - photospheric_index,), NormalizedValue
    ] = np.array(
        [
            scaled_1bar_temperature,
            scaled_0p1bar_temperature,
            scaled_0p01bar_temperature,
            scaled_0p001bar_temperature,
            scaled_0p0001bar_temperature,
        ]
    )
    proportions_deeper: np.ndarray[
        ((number_of_pressure_nodes - photospheric_index) - 1,), NormalizedValue
    ] = np.array(
        [
            scaled_10bar_temperature,
            scaled_30bar_temperature,
            scaled_100bar_temperature,
            scaled_300bar_temperature,
        ]
    )

    temperatures: np.ndarray[(number_of_pressure_nodes,), TemperatureValue] = np.empty(
        number_of_pressure_nodes, dtype=np.float64
    )

    # Back out the physical temperature at 3 bars, which is the fiducial photosphere,
    # given its fractional position between the lower and upper bound of the valid temperatures.
    photospheric_temperature: TemperatureValue = (
        lower_temperature_bound
        + photospheric_scaled_3bar_temperature
        * (upper_temperature_bound - lower_temperature_bound)
    )
    temperatures[photospheric_index] = photospheric_temperature

    current_temperature: TemperatureValue = photospheric_temperature
    remaining_shallower_range: TemperatureValue = (
        current_temperature - lower_temperature_bound
    )

    # Sample from the photospheric temperature to the top of the (model) atmosphere
    for i in range(photospheric_index - 1, -1, -1):
        if remaining_shallower_range <= 0:
            decrement: float = 0.0

        else:
            proportion: NormalizedValue = proportions_shallower[
                photospheric_index - 1 - i
            ]
            decrement: TemperatureValue = proportion * remaining_shallower_range
            remaining_shallower_range -= decrement

        temperatures[i] = current_temperature - decrement

        current_temperature: TemperatureValue = temperatures[i]

    # Sample from the photospheric temperature to the bottom of the (model) atmosphere
    current_temperature: TemperatureValue = photospheric_temperature
    remaining_upward_range: TemperatureValue = (
        upper_temperature_bound - current_temperature
    )

    for i in range(photospheric_index + 1, number_of_pressure_nodes):
        if remaining_upward_range <= 0:
            increment = 0.0

        else:
            proportion: NormalizedValue = proportions_deeper[
                i - (photospheric_index + 1)
            ]
            increment: TemperatureValue = proportion * remaining_upward_range
            remaining_upward_range -= increment

        temperatures[i] = current_temperature + increment

        current_temperature: TemperatureValue = temperatures[i]

    return temperatures


def general_piette_function(
    profile_log_pressures: np.ndarray[(NumberofModelPressures,), LogPressureValue],
    log_pressure_nodes: np.ndarray[(NumberofNodes,), LogPressureValue],
    temperature_nodes: np.ndarray[(NumberofNodes,), TemperatureValue],
    smoothing_parameter: PositiveValue = 0.3,
) -> np.ndarray[(NumberofModelPressures,), TemperatureValue]:
    interpolated_function: Callable[
        [np.ndarray[(NumberofNodes,), LogPressureValue], PositiveValue],
        np.ndarray[(NumberofModelPressures,), TemperatureValue],
    ] = monotonic_interpolation(log_pressure_nodes, temperature_nodes)

    TP_profile = gaussian_smoothing(
        interpolated_function(profile_log_pressures), sigma=smoothing_parameter
    )

    return TP_profile


def generate_piette_model(
    model_parameters: PietteTemperatureModelParameters,
    model_inputs: TemperatureBounds,
) -> TemperatureModel:
    log_pressure_nodes: np.ndarray[(10,), np.float64] = np.array(
        [2, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 8.5]
    )  # cgs units (1 bar = 1e6 cgs)
    number_of_pressure_nodes: int = len(log_pressure_nodes)

    temperature_nodes: np.ndarray[(number_of_pressure_nodes,), TemperatureValue] = (
        create_monotonic_temperature_nodes_from_samples(
            *astuple(model_parameters),
            *astuple(model_inputs),
        )
    )

    return partial(
        general_piette_function,
        log_pressure_nodes=log_pressure_nodes,
        temperature_nodes=temperature_nodes,
    )


piette: TemperatureModel = generate_piette_model  # alias


# ADA: the closest representation of the original Piette et. al. (2020) prescription.
# Reference: https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.5136P
def proper_piette_model(
    T_0p5: TemperatureValue,  # i.e. T(10**0.5 bar)
    delta_T_2p5_T_2: TemperatureValue,
    delta_T_2_T_1p5: TemperatureValue,
    delta_T_1p5_T_1: TemperatureValue,
    delta_T_1_T_0p5: TemperatureValue,
    delta_T_0p5_T_0: TemperatureValue,
    delta_T_0_T_m1: TemperatureValue,
    delta_T_m1_T_m2: TemperatureValue,
    delta_T_m2_T_m3: TemperatureValue,
    delta_T_m3_T_m4: TemperatureValue,
    log_pressures: np.ndarray[(NumberofModelPressures,), LogPressureValue],
) -> np.ndarray[(NumberofModelPressures,), TemperatureValue]:
    number_of_pressure_nodes: int = 10

    log_pressure_nodes: np.ndarray[(number_of_pressure_nodes,), np.float64] = np.array(
        [-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 2.5]
    )

    temperature_nodes: np.ndarray[(number_of_pressure_nodes,), TemperatureValue] = (
        np.array(
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
    )

    return general_piette_function(
        *temperature_nodes,
        log_pressure_nodes=log_pressure_nodes,
        log_pressures=log_pressures,
        smoothing_parameter=0.3,
    )
