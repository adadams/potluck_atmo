from typing import Any

import numpy as np
import xarray as xr

from constants_and_conversions import MICRONS_TO_CM
from temperature.thermal_intensity import calculate_thermal_intensity_by_layer
from test_inputs.test_data_structures.input_structs import UserVerticalModelInputs


def compile_thermal_structure_for_forward_model(
    vertical_inputs: UserVerticalModelInputs, model_wavelengths_in_microns: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    model_wavelengths_in_cm = model_wavelengths_in_microns * MICRONS_TO_CM

    temperature_grid, wavelength_grid = np.meshgrid(
        vertical_inputs.temperatures_by_level,
        model_wavelengths_in_cm,
    )

    thermal_intensity, delta_thermal_intensity = calculate_thermal_intensity_by_layer(
        wavelength_grid, temperature_grid
    )

    shared_thermal_intensity_kwargs: dict[str, Any] = {
        "dims": ("wavelength", "pressure"),
        "coords": {
            "pressure": vertical_inputs.pressure,
            "wavelength": model_wavelengths_in_microns,
        },
        "attrs": {"units": "erg s^-1 cm^-3 sr^-1"},
    }

    thermal_intensity = xr.DataArray(
        data=thermal_intensity,
        name="thermal_intensity",
        **shared_thermal_intensity_kwargs,
    )

    delta_thermal_intensity = xr.DataArray(
        data=delta_thermal_intensity,
        name="delta_thermal_intensity",
        **shared_thermal_intensity_kwargs,
    )
