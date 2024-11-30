import numpy as np
import xarray as xr
from numpy.typing import NDArray

from material.scattering.types import TwoStreamScatteringCoefficients
from xarray_functional_wrappers import Dimensionalize
from xarray_serialization import PressureType, WavelengthType

REFERENCE_FREQUENCY_IN_HZ: float = 5.0872638e14
C_IN_CGS: float = 2.99792458e10


# Note: don't need to specify dimensionality of any default
# arguments, since xarray will know how to use it.
@Dimensionalize(
    argument_dimensions=(
        (WavelengthType,),
        (
            PressureType,
            WavelengthType,
        ),
    ),
    result_dimensions=(
        (
            PressureType,
            WavelengthType,
        ),
    ),
)
def calculate_rayleigh_scattering_crosssections(
    wavelengths_in_cm: float | NDArray[np.float64],
    crosssections: float | NDArray[np.float64],
    reference_frequency: float = REFERENCE_FREQUENCY_IN_HZ,
) -> float | NDArray[np.float64]:
    frequencies: float | NDArray[np.float64] = C_IN_CGS / wavelengths_in_cm

    return crosssections * (frequencies / reference_frequency) ** 4


@Dimensionalize(
    argument_dimensions=(
        (WavelengthType,),
        (
            PressureType,
            WavelengthType,
        ),
        (PressureType,),
    ),
    result_dimensions=(
        (
            WavelengthType,
            PressureType,
        ),
    ),
)
def calculate_rayleigh_scattering_attenuation_coefficients(
    wavelengths_in_cm: float | NDArray[np.float64],
    crosssections: float | NDArray[np.float64],
    number_density: NDArray[np.float64],
    reference_frequency: float = REFERENCE_FREQUENCY_IN_HZ,
) -> float | NDArray[np.float64]:
    frequencies: float | NDArray[np.float64] = C_IN_CGS / wavelengths_in_cm

    return crosssections * (frequencies / reference_frequency) ** 4 * number_density


def calculate_two_stream_scattering_components(
    wavelengths_in_cm: xr.DataArray,
    crosssections: xr.DataArray,
    number_density: xr.DataArray,
    reference_frequency: float = REFERENCE_FREQUENCY_IN_HZ,
) -> TwoStreamScatteringCoefficients:
    rayleigh_scattering_crosssections: xr.DataArray = (
        calculate_rayleigh_scattering_crosssections(
            wavelengths_in_cm,
            crosssections,
            reference_frequency=reference_frequency,
        )
    )

    rayleigh_scattering_attenuation_coefficients: xr.DataArray = (
        rayleigh_scattering_crosssections * number_density
    )

    forward_scattering_attentuation_coefficients: float | NDArray[np.float64] = (
        0.5 * rayleigh_scattering_attenuation_coefficients
    )

    backward_scattering_attentuation_coefficients: float | NDArray[np.float64] = (
        0.5 * rayleigh_scattering_attenuation_coefficients
    )

    return TwoStreamScatteringCoefficients(
        forward_scattering_attentuation_coefficients,
        backward_scattering_attentuation_coefficients,
    )
