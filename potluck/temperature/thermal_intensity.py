import numpy as np
import xarray as xr

from potluck.basic_types import PressureDimension, WavelengthDimension
from potluck.constants_and_conversions import c, hc, hc_over_k
from potluck.vertical.altitude import (
    calculate_change_across_pressure_layer,
    convert_dataarray_by_pressure_levels_to_pressure_layers,
)
from potluck.xarray_functional_wrappers import Dimensionalize, set_result_name_and_units


@set_result_name_and_units(
    result_names="thermal_intensity_by_level", units="erg s^-1 cm^-3"
)
@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
    ),
    result_dimensions=((WavelengthDimension, PressureDimension),),
)
def blackbody_intensity_by_wavelength(
    wavelength_in_cm: float | np.ndarray[np.float64],
    temperature_in_K: float | np.ndarray[np.float64],
):
    return (2 * hc * c / wavelength_in_cm**5) / (
        np.exp(hc_over_k / (wavelength_in_cm * temperature_in_K)) - 1
    )


def calculate_thermal_intensity_by_layer(
    wavelength_grid_in_cm: xr.DataArray,
    temperature_grid_in_K: xr.DataArray,
):
    thermal_intensity_by_level: xr.DataArray = blackbody_intensity_by_wavelength(
        wavelength_grid_in_cm, temperature_grid_in_K
    )

    # mean across each layer bin
    thermal_intensity: xr.DataArray = (
        convert_dataarray_by_pressure_levels_to_pressure_layers(
            thermal_intensity_by_level
        )
    )

    # change across each layer bin
    delta_thermal_intensity: xr.DataArray = calculate_change_across_pressure_layer(
        thermal_intensity_by_level
    )

    return xr.merge((thermal_intensity, delta_thermal_intensity))
