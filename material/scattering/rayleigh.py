import numpy as np
from numpy.typing import NDArray

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
def calculate_rayleigh_scattering_crosssection(
    wavelength_in_cm: float | NDArray[np.float64],
    reference_crosssection: float | NDArray[np.float64],
    reference_frequency: float = REFERENCE_FREQUENCY_IN_HZ,
) -> float | NDArray[np.float64]:
    frequencies: float | NDArray[np.float64] = C_IN_CGS / wavelength_in_cm

    return reference_crosssection * (frequencies / reference_frequency) ** 4
