from typing import Any

import numpy as np
import xarray as xr

from constants_and_conversions import MICRONS_TO_CM
from temperature.thermal_intensity import calculate_thermal_intensity_by_layer
from xarray_functional_wrappers import save_xarray_outputs_to_file


@save_xarray_outputs_to_file
def compile_thermal_structure_for_forward_model(
    temperatures_by_level: xr.DataArray,
    pressures_by_layer: xr.DataArray,
    model_wavelengths_in_microns: xr.DataArray,
) -> xr.Dataset:
    model_wavelengths_in_cm = model_wavelengths_in_microns * MICRONS_TO_CM

    temperature_grid, wavelength_grid = np.meshgrid(
        temperatures_by_level,
        model_wavelengths_in_cm,
        # indexing="ij",
    )

    thermal_intensity, delta_thermal_intensity = calculate_thermal_intensity_by_layer(
        wavelength_grid, temperature_grid
    )

    shared_thermal_intensity_kwargs: dict[str, Any] = {
        "dims": ("wavelength", "pressure"),
        "coords": {
            "pressure": pressures_by_layer,
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

    return xr.merge([thermal_intensity, delta_thermal_intensity])


if __name__ == "__main__":
    pass
