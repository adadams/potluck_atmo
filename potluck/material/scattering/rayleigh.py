import numpy as np
import xarray as xr

from potluck.basic_types import PressureDimension, WavelengthDimension
from potluck.material.scattering.scattering_types import TwoStreamScatteringCoefficients
from potluck.xarray_functional_wrappers import Dimensionalize

REFERENCE_FREQUENCY_IN_HZ: float = 5.0872638e14
C_IN_CGS: float = 2.99792458e10


# Note: don't need to specify dimensionality of any default
# arguments, since xarray will know how to use it.
@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension,),
        (
            PressureDimension,
            WavelengthDimension,
        ),
    ),
    result_dimensions=(
        (
            PressureDimension,
            WavelengthDimension,
        ),
    ),
)
def calculate_rayleigh_scattering_crosssections(
    wavelengths_in_cm: float | np.ndarray[np.float64],
    crosssections: float | np.ndarray[np.float64],
    reference_frequency: float = REFERENCE_FREQUENCY_IN_HZ,
) -> float | np.ndarray[np.float64]:
    frequencies: float | np.ndarray[np.float64] = C_IN_CGS / wavelengths_in_cm

    return crosssections * (frequencies / reference_frequency) ** 4


@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension,),
        (
            # PressureDimension,
            WavelengthDimension,
        ),
        (PressureDimension,),
    ),
    result_dimensions=(
        (
            WavelengthDimension,
            PressureDimension,
        ),
    ),
)
def calculate_rayleigh_scattering_attenuation_coefficients(
    wavelengths_in_cm: float | np.ndarray[np.float64],
    rayleigh_scattering_crosssections: float | np.ndarray[np.float64],
    number_density: np.ndarray[np.float64],
    reference_frequency: float = REFERENCE_FREQUENCY_IN_HZ,
) -> float | np.ndarray[np.float64]:
    frequencies: float | np.ndarray[np.float64] = C_IN_CGS / wavelengths_in_cm

    return (
        rayleigh_scattering_crosssections
        * (frequencies / reference_frequency) ** 4
        * number_density[:, np.newaxis]
    ).T


def calculate_two_stream_scattering_components(
    wavelengths_in_cm: xr.DataArray,
    # scattering_crosssections: xr.DataArray,
    number_density: xr.DataArray,
    # reference_frequency: float = REFERENCE_FREQUENCY_IN_HZ,
) -> TwoStreamScatteringCoefficients:
    rayleigh_scattering_crosssections: xr.DataArray = (
        calculate_H2_rayleigh_scattering_crosssections_in_cm2(
            wavelengths_in_angstroms=wavelengths_in_cm * 1e8
        )
    )

    rayleigh_scattering_attenuation_coefficients: xr.DataArray = (
        calculate_rayleigh_scattering_attenuation_coefficients(
            wavelengths_in_cm,
            rayleigh_scattering_crosssections,
            number_density.sel(species=["h2"]),
        )
    )

    forward_scattering_attentuation_coefficients: float | np.ndarray[np.float64] = (
        0.5 * rayleigh_scattering_attenuation_coefficients
    )

    backward_scattering_attentuation_coefficients: float | np.ndarray[np.float64] = (
        0.5 * rayleigh_scattering_attenuation_coefficients
    )

    return TwoStreamScatteringCoefficients(
        forward_scattering_attentuation_coefficients,
        backward_scattering_attentuation_coefficients,
    )


def calculate_rayleigh_scattering_crosssections_from_refractive_indices(
    wavelengths_in_cm: float | np.ndarray[np.float64],
    number_density: float | np.ndarray[np.float64],
    refractive_indices: float | np.ndarray[np.float64],
    King_correction_factor: float | np.ndarray[np.float64],
) -> float | np.ndarray[np.float64]:
    """
    sigma(nu) = (24 pi^3 nu^4 / N^2) * ((n^2 - 1)/(n^2 + 2))^2 * F_k(nu)
    where sigma(nu) is the Rayleigh scattering cross section,
    n = n(nu) is the refractive index,
    and F_k(nu) is the King correction factor which accounts for depolarization.
    (From Thalman et. al. 2014, Equation 1)
    """

    frequencies: float | np.ndarray[np.float64] = C_IN_CGS / wavelengths_in_cm

    return (
        (24 * np.pi**3 * frequencies**4 / number_density**2)
        * ((refractive_indices**2 - 1) / (refractive_indices**2 + 2)) ** 2
        * King_correction_factor
    )


def calculate_H2_rayleigh_scattering_crosssections_in_cm2(
    wavelengths_in_angstroms: float | np.ndarray[np.float64],
) -> float | np.ndarray[np.float64]:
    """
    From Dalgarno & Williams (1962), Equation 3.
    Neglects terms of order lambda^10.
    """

    return (
        8.14e-13 * wavelengths_in_angstroms ** (-4)
        + 1.28e-6 * wavelengths_in_angstroms ** (-6)
        + 1.61 * wavelengths_in_angstroms ** (-8)
    )
