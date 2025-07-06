from collections.abc import Callable
from functools import partial
from typing import Optional

import numpy as np
import xarray as xr
from spectres import spectres

from spectrum.convolve import convolve_with_constant_FWHM_using_FFT

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


def convolve_and_bin_with_constant_FWHM(
    new_wavelengths: np.ndarray[np.float64],
    model_wavelengths: np.ndarray[np.float64],
    model_spectral_quantity: np.ndarray[np.float64],
    fwhm: float,
):
    convolved_spectrum: np.ndarray[np.float64] = convolve_with_constant_FWHM_using_FFT(
        model_wavelengths, model_spectral_quantity, fwhm
    )

    return spectres(new_wavelengths, model_wavelengths, convolved_spectrum)


convolve_and_resample_by_spectres = partial(
    xr.apply_ufunc,
    convolve_and_bin_with_constant_FWHM,
    input_core_dims=[["wavelength"], ["wavelength"], ["wavelength"], []],
    output_core_dims=[["wavelength"]],
    exclude_dims=set(("wavelength",)),
    keep_attrs=True,
)


# @Dimensionalize(
#    argument_dimensions=(
#        (WavelengthDimension,),
#        (WavelengthDimension,),
#        (WavelengthDimension,),
#    ),
#    result_dimensions=((WavelengthDimension,),),
# )
# TODO: write a separate wrapper for when dimensions change size.
def resample_spectral_quantity_to_new_wavelengths(
    new_wavelengths: xr.DataArray,
    model_wavelengths: xr.DataArray,
    model_spectral_quantity: xr.DataArray,
    fwhm: Optional[float] = None,
) -> xr.Dataset:
    if fwhm is None:
        fwhm = 3.0 * (new_wavelengths[-1] - new_wavelengths[-2])

    return convolve_and_resample_by_spectres(
        new_wavelengths, model_wavelengths, model_spectral_quantity, fwhm
    ).assign_coords(wavelength=new_wavelengths)
