from typing import Final, NamedTuple

import numpy as np
from numpy.typing import NDArray

from xarray_functional_wrappers import Dimensionalize
from xarray_serialization import PressureType, WavelengthType

c: Final[float] = 2.99792458e10  # in CGS
hc: Final[float] = 1.98644568e-16  # in CGS
hc_over_k: Final[float] = 1.98644568 / 1.38064852


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
    # change across each layer bin
    delta_thermal_intensity = (
        thermal_intensity_bin_edges[:, 1:] - thermal_intensity_bin_edges[:, :-1]
    )

    return ThermalIntensityByLayer(thermal_intensity, delta_thermal_intensity)
