import numpy as np


def convert_surface_quantity_to_observed_quantity(
    radius_in_solar_radii: float,
    distance_in_parsecs: float,
):
    distance_in_solar_radii: float = distance_in_parsecs * 4.435e7

    return (radius_in_solar_radii / distance_in_solar_radii) ** 2 / (4 * np.pi)
