from typing import Final

import numpy as np

PARSECS_TO_SOLAR_RADII: Final[float] = 4.435e7


def convert_surface_quantity_to_observed_quantity(
    radius_in_solar_radii: float, distance_in_parsecs: float
):
    distance_in_solar_radii: float = distance_in_parsecs * PARSECS_TO_SOLAR_RADII

    return (radius_in_solar_radii / distance_in_solar_radii) ** 2 / (4 * np.pi)
