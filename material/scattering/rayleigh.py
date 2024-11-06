from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

REFERENCE_FREQUENCY: float = 5.0872638e14


def calculate_rayleigh_scattering_crosssection(
    wavelengths: NDArray[np.float64],
    reference_crosssections: NDArray[np.float64],
    reference_frequency: float = REFERENCE_FREQUENCY,
) -> float:
    frequencies: NDArray[np.float64] = 1 / (wavelengths * 1e-4)

    return reference_crosssections * (frequencies / reference_frequency) ** 4
