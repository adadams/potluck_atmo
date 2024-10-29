from typing import Final

import numpy as np
from numpy.typing import NDArray

GRAVITATIONAL_CONSTANT_IN_CGS: Final[float] = 6.67408e-8  # [cm^3 g^-1 s^-2]
BOLTZMANN_CONSTANT_IN_CGS: Final[float] = 1.38065e-16  # [cm^2 g s^-2 K^-1]
AMU_IN_GRAMS: Final[float] = 1.66054e-24
BAR_TO_BARYE: Final[float] = 1.0e6
EARTH_RADIUS_IN_CM: Final[float] = 6.371e8
JUPITER_RADIUS_IN_CM: Final[float] = 6.991e8
JUPITER_MASS_IN_G: Final[float] = 1.898e30


def calculate_altitude_profile(
    log_pressures: NDArray[np.float64],
    temperatures: NDArray[np.float64],
    mean_molecular_weight_in_g: float,
    planet_radius_in_cm: float,
    planet_mass_in_g: float,
) -> None:
    log_pressure_differences = log_pressures[1:] - log_pressures[:-1]

    altitudes: NDArray[np.float64] = np.empty_like(log_pressures)
    altitudes[-1] = 0.0

    for i, (log_pressure_difference, temperature) in enumerate(
        zip(log_pressure_differences, reversed(temperatures[:-1])), start=1
    ):
        dPdr: float = (
            GRAVITATIONAL_CONSTANT_IN_CGS
            * planet_mass_in_g
            * mean_molecular_weight_in_g
            / (
                BOLTZMANN_CONSTANT_IN_CGS
                * temperature
                * (planet_radius_in_cm + altitudes[-i]) ** 2
            )
        )

        altitude_difference: float = log_pressure_difference / dPdr
        altitudes[-(i + 1)] = altitudes[-i] + altitude_difference

    return altitudes


def impose_upper_limit_on_altitude(
    altitudes: NDArray[np.float64], upper_altitude_limit: float
):
    return np.clip(altitudes, a_max=upper_altitude_limit)
