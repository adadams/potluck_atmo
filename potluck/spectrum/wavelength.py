from collections.abc import Callable
from functools import partial, wraps
from typing import Any

import numpy as np
import xarray as xr

from potluck.xarray_serialization import UnitsAttrs, XarrayDimension, XarrayVariable

type CoordinateBuilder = Callable[
    [np.ndarray[np.float64], XarrayDimension, UnitsAttrs, type], XarrayVariable
]


def build_coordinate(
    data: np.ndarray[np.float64],
    dims: XarrayDimension,
    attrs: UnitsAttrs,
    coordinate_class: type = XarrayVariable,
) -> XarrayVariable:
    return coordinate_class(data=data, dims=dims, attrs=attrs)


class WavelengthCoordinate(XarrayVariable): ...


build_wavelength_coordinate: CoordinateBuilder = partial(
    build_coordinate,
    dims=("wavelength",),
    attrs=dict(units="micron"),
    # attrs=UnitsAttrs(units="micron"),
    coordinate_class=xr.Variable,
    # coordinate_class=WavelengthCoordinate,
)


def return_coordinate(dims: XarrayDimension, units: str) -> Callable:
    # attrs: UnitsAttrs = UnitsAttrs(units=units)
    attrs = dict(units=units)

    def build_function_with_result_in_coordinate_form(
        function: Callable[[Any], np.ndarray[np.float64]],
    ):
        @wraps(function)
        def function_with_result_in_coordinate_form(
            *args: Any, **kwargs: Any
        ) -> WavelengthCoordinate:
            function_result: np.ndarray[np.float64] = function(*args, **kwargs)

            return build_wavelength_coordinate(
                data=function_result, dims=dims, attrs=attrs
            )

        return function_with_result_in_coordinate_form

    return build_function_with_result_in_coordinate_form


return_wavelength_coordinate: Callable[
    [XarrayDimension, str], Callable[[Any], WavelengthCoordinate]
] = partial(return_coordinate, dims=("wavelength",))


@return_wavelength_coordinate(units="micron")
def build_wavelength_array(
    minimum_wavelength: float,
    maximum_wavelength: float,
    effective_resolution: float,
) -> np.ndarray[np.float64]:
    number_of_spectral_elements: int = int(
        effective_resolution * np.log(maximum_wavelength / minimum_wavelength) + 1
    )

    return minimum_wavelength * np.exp(
        np.arange(number_of_spectral_elements) / effective_resolution
    )


def get_number_of_wavelengths(
    starting_wavelength: float, ending_wavelength: float, resolution: float
) -> int:
    return int(np.ceil(resolution * np.log(ending_wavelength / starting_wavelength)))


def get_wavelengths_from_number_of_elements_and_resolution(
    starting_wavelength: float, number_of_elements: int, spectral_resolution: float
) -> np.ndarray[np.float64]:
    return starting_wavelength * np.exp(
        np.arange(number_of_elements) / spectral_resolution
    )


def get_wavelengths_from_wavelength_bins(wavelength_bin_starts, wavelength_bin_ends):
    return (wavelength_bin_starts + wavelength_bin_ends) / 2


def calculate_mean_resolution(wavelengths: xr.DataArray):
    inbetween_wavelengths: xr.DataArray = (
        wavelengths.rolling(wavelength=2).mean().dropna("wavelength")
    )

    wavelength_spacings: xr.DataArray = wavelengths.diff("wavelength")

    return (inbetween_wavelengths / wavelength_spacings).mean().item()


def convert_spectral_coordinate_by_level_to_by_layer(
    spectral_coordinate: xr.DataArray,
) -> xr.DataArray:
    mid_spectral_bin_values: np.ndarray = (
        spectral_coordinate.to_numpy()[1:] + spectral_coordinate.to_numpy()[:-1]
    ) / 2

    return xr.DataArray(
        data=mid_spectral_bin_values,
        dims=(spectral_coordinate.name,),
        name=spectral_coordinate.name,
        attrs=spectral_coordinate.attrs,
    )


def calculate_spectral_quantity_at_wavelength_bin_midpoints(
    spectral_quantity: xr.DataArray,
) -> xr.DataArray:
    midbin_wavelengths: xr.DataArray = convert_spectral_coordinate_by_level_to_by_layer(
        spectral_quantity.wavelength
    )

    midbin_fluxes: xr.DataArray = spectral_quantity.interp(
        wavelength=midbin_wavelengths
    )

    return midbin_fluxes


def calculate_spectrally_integrated_flux(
    spectral_dataarray: xr.DataArray, flux_wavelength_units: str = "cm"
) -> xr.DataArray:
    midbin_fluxes: xr.DataArray = (
        calculate_spectral_quantity_at_wavelength_bin_midpoints(spectral_dataarray)
    )

    delta_wavelengths: xr.DataArray = (
        spectral_dataarray.wavelength.diff("wavelength")
        .assign_coords(wavelength=midbin_fluxes.wavelength)
        .rename("delta_wavelength")
        .assign_attrs(units=flux_wavelength_units)
    )

    spectrally_integrated_flux: xr.DataArray = midbin_fluxes * delta_wavelengths

    return spectrally_integrated_flux
