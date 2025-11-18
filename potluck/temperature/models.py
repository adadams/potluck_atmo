import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TypeVar

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
from msgspec.structs import astuple
from scipy.interpolate import PchipInterpolator as monotonic_interpolation
from scipy.ndimage import gaussian_filter1d as gaussian_smoothing

from potluck.basic_types import (
    LogPressureValue,
    NormalizedValue,
    PositiveValue,
    TemperatureValue,
)
from potluck.temperature.protocols import (
    TemperatureModel,
    TemperatureModelArguments,
    TemperatureModelInputs,
    TemperatureModelParameters,
    TemperatureModelSamples,
)

NumberofNodes = TypeVar("NumberofNodes", bound=tuple[int, ...])  # numpy ndarray shape
NumberofModelPressures = TypeVar("NumberofModelPressures", bound=tuple[int, ...])


class TemperatureBounds(TemperatureModelInputs):
    lower_temperature_bound: TemperatureValue
    upper_temperature_bound: TemperatureValue


class PietteTemperatureModelSamples(TemperatureModelSamples):
    photospheric_temperature_3bar: TemperatureValue
    scaled_temperature_1bar: NormalizedValue
    scaled_temperature_0p1bar: NormalizedValue
    scaled_temperature_0p01bar: NormalizedValue
    scaled_temperature_0p001bar: NormalizedValue
    scaled_temperature_0p0001bar: NormalizedValue
    scaled_temperature_10bar: NormalizedValue
    scaled_temperature_30bar: NormalizedValue
    scaled_temperature_100bar: NormalizedValue
    scaled_temperature_300bar: NormalizedValue


class PietteRossTemperatureModelSamples(TemperatureModelSamples):
    photospheric_temperature_3bar: TemperatureValue
    scaled_temperature_1bar: NormalizedValue
    scaled_temperature_0p1bar: NormalizedValue
    scaled_temperature_0p01bar: NormalizedValue
    scaled_temperature_0p001bar: NormalizedValue
    scaled_temperature_10bar: NormalizedValue
    scaled_temperature_30bar: NormalizedValue
    scaled_temperature_100bar: NormalizedValue


# @njit
def create_monotonic_temperature_nodes_from_samples(
    photospheric_temperature_3bar: TemperatureValue,
    scaled_temperature_1bar: NormalizedValue,
    scaled_temperature_0p1bar: NormalizedValue,
    scaled_temperature_0p01bar: NormalizedValue,
    scaled_temperature_0p001bar: NormalizedValue,
    scaled_temperature_0p0001bar: NormalizedValue,
    scaled_temperature_10bar: NormalizedValue,
    scaled_temperature_30bar: NormalizedValue,
    scaled_temperature_100bar: NormalizedValue,
    scaled_temperature_300bar: NormalizedValue,
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
            scaled_temperature_1bar,
            scaled_temperature_0p1bar,
            scaled_temperature_0p01bar,
            scaled_temperature_0p001bar,
            scaled_temperature_0p0001bar,
        ]
    )
    proportions_deeper: np.ndarray[
        ((number_of_pressure_nodes - photospheric_index) - 1,), NormalizedValue
    ] = np.array(
        [
            scaled_temperature_10bar,
            scaled_temperature_30bar,
            scaled_temperature_100bar,
            scaled_temperature_300bar,
        ]
    )

    temperatures: np.ndarray[(number_of_pressure_nodes,), TemperatureValue] = np.empty(
        (
            number_of_pressure_nodes,
            *np.shape(photospheric_temperature_3bar),
        ),
        dtype=np.float64,
    )

    # Back out the physical temperature at 3 bars, which is the fiducial photosphere,
    # given its fractional position between the lower and upper bound of the valid temperatures.
    temperatures[photospheric_index] = photospheric_temperature_3bar

    current_temperature: TemperatureValue = photospheric_temperature_3bar
    remaining_shallower_range: TemperatureValue = (
        current_temperature - lower_temperature_bound
    )

    # Sample from the photospheric temperature to the top of the (model) atmosphere
    for i in range(photospheric_index - 1, -1, -1):
        proportion: NormalizedValue = proportions_shallower[photospheric_index - 1 - i]
        decrement: TemperatureValue = np.where(
            remaining_shallower_range > 0, proportion * remaining_shallower_range, 0
        )
        remaining_shallower_range -= decrement

        temperatures[i] = current_temperature - decrement

        current_temperature: TemperatureValue = temperatures[i]

    # Sample from the photospheric temperature to the bottom of the (model) atmosphere
    current_temperature: TemperatureValue = photospheric_temperature_3bar
    remaining_upward_range: TemperatureValue = (
        upper_temperature_bound - current_temperature
    )

    for i in range(photospheric_index + 1, number_of_pressure_nodes):
        proportion: NormalizedValue = proportions_deeper[i - (photospheric_index + 1)]
        increment: TemperatureValue = np.where(
            remaining_upward_range > 0, proportion * remaining_upward_range, 0
        )
        remaining_upward_range -= increment

        temperatures[i] = current_temperature + increment

        current_temperature: TemperatureValue = temperatures[i]

    return temperatures


def create_monotonic_temperature_nodes_from_Ross_samples(
    photospheric_temperature_3bar: TemperatureValue,
    scaled_temperature_1bar: NormalizedValue,
    scaled_temperature_0p1bar: NormalizedValue,
    scaled_temperature_0p01bar: NormalizedValue,
    scaled_temperature_0p001bar: NormalizedValue,
    scaled_temperature_10bar: NormalizedValue,
    scaled_temperature_30bar: NormalizedValue,
    scaled_temperature_100bar: NormalizedValue,
    lower_temperature_bound: TemperatureValue,  # usually fixed, set by e.g. opacity temperature range
    upper_temperature_bound: TemperatureValue,  # usually fixed, set by e.g. opacity temperature range
) -> np.ndarray[(8,), TemperatureValue]:
    """
    Enforces a temperature profile that does not decrease with increasing pressure.
    """

    number_of_pressure_nodes: int = 8
    photospheric_index: int = 4

    proportions_shallower: np.ndarray[
        (number_of_pressure_nodes - photospheric_index,), NormalizedValue
    ] = np.array(
        [
            scaled_temperature_1bar,
            scaled_temperature_0p1bar,
            scaled_temperature_0p01bar,
            scaled_temperature_0p001bar,
        ]
    )
    proportions_deeper: np.ndarray[
        ((number_of_pressure_nodes - photospheric_index) - 1,), NormalizedValue
    ] = np.array(
        [scaled_temperature_10bar, scaled_temperature_30bar, scaled_temperature_100bar]
    )

    temperatures: np.ndarray[(number_of_pressure_nodes,), TemperatureValue] = np.empty(
        (
            number_of_pressure_nodes,
            *np.shape(photospheric_temperature_3bar),
        ),
        dtype=np.float64,
    )

    # Back out the physical temperature at 3 bars, which is the fiducial photosphere,
    # given its fractional position between the lower and upper bound of the valid temperatures.
    temperatures[photospheric_index] = photospheric_temperature_3bar

    current_temperature: TemperatureValue = photospheric_temperature_3bar
    remaining_shallower_range: TemperatureValue = (
        current_temperature - lower_temperature_bound
    )

    # Sample from the photospheric temperature to the top of the (model) atmosphere
    for i in range(photospheric_index - 1, -1, -1):
        proportion: NormalizedValue = proportions_shallower[photospheric_index - 1 - i]
        decrement: TemperatureValue = np.where(
            remaining_shallower_range > 0, proportion * remaining_shallower_range, 0
        )
        remaining_shallower_range -= decrement

        temperatures[i] = current_temperature - decrement

        current_temperature: TemperatureValue = temperatures[i]

    # Sample from the photospheric temperature to the bottom of the (model) atmosphere
    current_temperature: TemperatureValue = photospheric_temperature_3bar
    remaining_upward_range: TemperatureValue = (
        upper_temperature_bound - current_temperature
    )

    for i in range(photospheric_index + 1, number_of_pressure_nodes):
        proportion: NormalizedValue = proportions_deeper[i - (photospheric_index + 1)]
        increment: TemperatureValue = np.where(
            remaining_upward_range > 0, proportion * remaining_upward_range, 0
        )
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

    test_smoothing_parameter: float = round(
        smoothing_parameter
        / (
            (np.max(log_pressure_nodes) - np.min(log_pressure_nodes))
            / len(profile_log_pressures)
        )
    )

    TP_profile = gaussian_smoothing(
        interpolated_function(profile_log_pressures),
        sigma=test_smoothing_parameter,
        mode="nearest",
    )

    return TP_profile


class PietteRossTemperatureModelParameters(TemperatureModelParameters):
    temperature_0p001bar: TemperatureValue
    temperature_0p01bar: TemperatureValue
    temperature_0p1bar: TemperatureValue
    temperature_1bar: TemperatureValue
    photospheric_temperature_3bar: TemperatureValue
    temperature_10bar: TemperatureValue
    temperature_30bar: TemperatureValue
    temperature_100bar: TemperatureValue


class PietteTemperatureModelParameters(TemperatureModelParameters):
    temperature_0p0001bar: TemperatureValue
    temperature_0p001bar: TemperatureValue
    temperature_0p01bar: TemperatureValue
    temperature_0p1bar: TemperatureValue
    temperature_1bar: TemperatureValue
    photospheric_temperature_3bar: TemperatureValue
    temperature_10bar: TemperatureValue
    temperature_30bar: TemperatureValue
    temperature_100bar: TemperatureValue
    temperature_300bar: TemperatureValue


class PietteTemperatureModelArguments(TemperatureModelArguments, kw_only=True):
    model_inputs: TemperatureBounds
    model_parameters: PietteTemperatureModelParameters


class PietteRossTemperatureModelArguments(TemperatureModelArguments, kw_only=True):
    model_inputs: TemperatureBounds
    model_parameters: PietteRossTemperatureModelParameters


def generate_piette_model(
    model_parameters: PietteTemperatureModelParameters,
) -> TemperatureModel:
    log_pressure_nodes: np.ndarray[(10,), np.float64] = np.array(
        [2, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 8.5]
    )  # cgs units (1 bar = 1e6 cgs "barye")
    number_of_pressure_nodes: int = len(log_pressure_nodes)

    temperature_nodes: np.ndarray[(number_of_pressure_nodes,), TemperatureValue] = (
        np.asarray(astuple(model_parameters))
    )

    return partial(
        general_piette_function,
        log_pressure_nodes=log_pressure_nodes,
        temperature_nodes=temperature_nodes,
    )


def generate_piette_model_for_Ross458c(
    model_parameters: PietteRossTemperatureModelParameters,
) -> TemperatureModel:
    log_pressure_nodes: np.ndarray[(8,), np.float64] = np.array(
        [3, 4, 5, 6, 6.5, 7, 7.5, 8]
    )  # cgs units (1 bar = 1e6 cgs "barye")
    number_of_pressure_nodes: int = len(log_pressure_nodes)

    temperature_nodes: np.ndarray[(number_of_pressure_nodes,), TemperatureValue] = (
        np.asarray(astuple(model_parameters))
    )

    return partial(
        general_piette_function,
        log_pressure_nodes=log_pressure_nodes,
        temperature_nodes=temperature_nodes,
    )


piette: TemperatureModel = generate_piette_model  # alias
piette_for_Ross458c: TemperatureModel = generate_piette_model_for_Ross458c  # alias


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


class IsothermalTemperatureModelParameters(TemperatureModelParameters):
    temperature: TemperatureValue


class IsothermalTemperatureModelArguments(TemperatureModelArguments, kw_only=True):
    model_parameters: IsothermalTemperatureModelParameters


def generate_isothermal_model(
    model_parameters: IsothermalTemperatureModelParameters,
) -> TemperatureModel:
    return partial(isothermal_profile, temperature=model_parameters.temperature)


def isothermal_profile(
    log_pressures: np.ndarray[(NumberofModelPressures,), LogPressureValue],
    temperature: TemperatureValue,
) -> np.ndarray[(NumberofModelPressures,), TemperatureValue]:
    return np.full_like(log_pressures, temperature)
