import numpy as np

from constants_and_conversions import GRAVITATIONAL_CONSTANT_IN_CGS


def calculate_mass_from_radius_and_surface_gravity(
    radius_in_cm: float | np.ndarray[np.float64],
    surface_gravity_in_cgs: float | np.ndarray[np.float64],
) -> float | np.ndarray[np.float64]:
    return (surface_gravity_in_cgs * radius_in_cm**2) / GRAVITATIONAL_CONSTANT_IN_CGS


def calculate_mean_density_from_radius_and_surface_gravity(
    radius_in_cm: float | np.ndarray[np.float64],
    surface_gravity_in_cgs: float | np.ndarray[np.float64],
) -> float | np.ndarray[np.float64]:
    return calculate_mass_from_radius_and_surface_gravity(
        radius_in_cm, surface_gravity_in_cgs
    ) / ((4 / 3) * np.pi * radius_in_cm**3)
