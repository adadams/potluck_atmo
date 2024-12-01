from collections.abc import Callable
from functools import partial, wraps
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from spectres import spectres

from xarray_serialization import UnitsAttrs, XarrayDimension, XarrayVariable

type CoordinateBuilder = Callable[
    [NDArray[np.float64], XarrayDimension, UnitsAttrs, type], XarrayVariable
]


def build_coordinate(
    data: NDArray[np.float64],
    dims: XarrayDimension,
    attrs: UnitsAttrs,
    coordinate_class: type = XarrayVariable,
) -> XarrayVariable:
    return coordinate_class(data=data, dims=dims, attrs=attrs)


class WavelengthCoordinate(XarrayVariable): ...


build_wavelength_coordinate: CoordinateBuilder = partial(
    build_coordinate,
    dims=("wavelength",),
    attrs=UnitsAttrs(units="micron"),
    coordinate_class=WavelengthCoordinate,
)


def return_coordinate(dims: XarrayDimension, units: str) -> Callable:
    attrs: UnitsAttrs = UnitsAttrs(units=units)

    def build_function_with_result_in_coordinate_form(
        function: Callable[[Any], NDArray[np.float64]],
    ):
        @wraps(function)
        def function_with_result_in_coordinate_form(
            *args: Any, **kwargs: Any
        ) -> WavelengthCoordinate:
            function_result: NDArray[np.float64] = function(*args, **kwargs)

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
) -> NDArray[np.float64]:
    number_of_spectral_elements: int = int(
        effective_resolution * np.log(maximum_wavelength / minimum_wavelength) + 1
    )

    return minimum_wavelength * np.exp(
        np.arange(number_of_spectral_elements) / effective_resolution
    )


def get_number_of_wavelengths(
    starting_wavelength: float, ending_wavelength: float, resolution: float
) -> int:
    return int(
        np.ceil(resolution * np.log(ending_wavelength / starting_wavelength)) + 1
    )


def get_wavelengths_from_number_of_elements_and_resolution(
    starting_wavelength: float, number_of_elements: int, spectral_resolution: float
) -> NDArray[np.float64]:
    return starting_wavelength * np.exp(
        np.arange(number_of_elements) / spectral_resolution
    )


def get_wavelengths_from_wavelength_bins(wavelength_bin_starts, wavelength_bin_ends):
    return (wavelength_bin_starts + wavelength_bin_ends) / 2


resample_by_spectres: Callable[
    [xr.DataArray, xr.DataArray, xr.DataArray], xr.DataArray
] = partial(
    xr.apply_ufunc,
    spectres,
    input_core_dims=[["wavelength"], ["wavelength"], ["wavelength"]],
    output_core_dims=[["wavelength"]],
    exclude_dims=set(("wavelength",)),
    keep_attrs=True,
)


def resample_spectral_quantity_to_new_wavelengths(
    new_wavelengths: xr.DataArray,
    model_wavelengths: xr.DataArray,
    model_spectral_quantity: xr.DataArray,
) -> xr.Dataset:
    return resample_by_spectres(
        new_wavelengths, model_wavelengths, model_spectral_quantity
    ).assign_coords(wavelength=new_wavelengths)


if __name__ == "__main__":
    test_wavelength_coordinate: NDArray[np.float64] = build_wavelength_array(
        minimum_wavelength=0.8,
        maximum_wavelength=5.3,
        effective_resolution=200,
    )

    print(f"{test_wavelength_coordinate=}")
