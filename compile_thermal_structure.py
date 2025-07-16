import xarray as xr

from temperature.thermal_intensity import calculate_thermal_intensity_by_layer
from xarray_functional_wrappers import call_meshgrid_on_xarray


def compile_thermal_structure_for_forward_model(
    temperatures_by_level: xr.DataArray, model_wavelengths_in_cm: xr.DataArray
) -> xr.Dataset:
    temperature_grid, wavelength_grid = call_meshgrid_on_xarray(
        temperatures_by_level, model_wavelengths_in_cm
    )

    return calculate_thermal_intensity_by_layer(wavelength_grid, temperature_grid)


if __name__ == "__main__":
    pass
