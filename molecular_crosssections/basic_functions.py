import numpy as np
from numpy.typing import NDArray


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
