from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from constants_and_conversions import c, hc, hc_over_k
from xarray_functional_wrappers import Dimensionalize
from xarray_serialization import PressureType, WavelengthType


def blackbody_intensity_by_wavelength(
    wavelength_in_cm: float | NDArray[np.float64],
    temperature_in_K: float | NDArray[np.float64],
):
    return (2 * hc * c / wavelength_in_cm**5) / (
        np.exp(hc_over_k / (wavelength_in_cm * temperature_in_K)) - 1
    )


class ThermalIntensityByLayer(NamedTuple):
    thermal_intensity: NDArray[np.float64]
    delta_thermal_intensity: NDArray[np.float64]


@Dimensionalize(
    argument_dimensions=(
        (WavelengthType, PressureType),
        (WavelengthType, PressureType),
    ),
    result_dimensions=((WavelengthType, PressureType), (WavelengthType, PressureType)),
)
def calculate_thermal_intensity_by_layer(
    wavelength_grid_in_cm: NDArray[np.float64],
    temperature_grid_in_K: NDArray[np.float64],
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
