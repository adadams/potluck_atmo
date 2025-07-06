from typing import NamedTuple

import numpy as np

from constants_and_conversions import c, hc, hc_over_k
from xarray_functional_wrappers import Dimensionalize
from xarray_serialization import PressureDimension, WavelengthDimension


def blackbody_intensity_by_wavelength(
    wavelength_in_cm: float | np.ndarray[np.float64],
    temperature_in_K: float | np.ndarray[np.float64],
):
    return (2 * hc * c / wavelength_in_cm**5) / (
        np.exp(hc_over_k / (wavelength_in_cm * temperature_in_K)) - 1
    )


class ThermalIntensityByLayer(NamedTuple):
    thermal_intensity: np.ndarray[np.float64]
    delta_thermal_intensity: np.ndarray[np.float64]


@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
    ),
    result_dimensions=(
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
    ),
)
def calculate_thermal_intensity_by_layer(
    wavelength_grid_in_cm: np.ndarray[np.float64],
    temperature_grid_in_K: np.ndarray[np.float64],
):
    thermal_intensity_bin_edges = blackbody_intensity_by_wavelength(
        wavelength_grid_in_cm, temperature_grid_in_K
    )

    # mean across each layer bin
    thermal_intensity = (
        thermal_intensity_bin_edges[:, :-1] + thermal_intensity_bin_edges[:, 1:]
    ) / 2
    # thermal_intensity = thermal_intensity_bin_edges[:, :-1]

    # change across each layer bin
    delta_thermal_intensity = (
        thermal_intensity_bin_edges[:, 1:] - thermal_intensity_bin_edges[:, :-1]
    )

    return ThermalIntensityByLayer(thermal_intensity, delta_thermal_intensity)
