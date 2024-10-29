from typing import Final

import msgspec
import numpy as np
from numpy.typing import NDArray

# from function_level_type_checking import call_with_type_checks, type_check_arguments

GRAVITATIONAL_CONSTANT_IN_CGS: Final[float] = 6.67408e-8  # [cm^3 g^-1 s^-2]


class RadiusType(msgspec.Struct):
    radius: float | NDArray[np.float64]
    units: str = "Jupiter_radii"


class SurfaceGravityType(msgspec.Struct):
    log10_surface_gravity: float | NDArray[np.float64]
    units: str = "cm/s^2"

    @property
    def surface_gravity(self) -> float | NDArray[np.float64]:
        return np.power(10, self.log10_surface_gravity)


def mass(
    radius_in_cm: float | NDArray[np.float64],
    surface_gravity_in_cgs: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return (surface_gravity_in_cgs * radius_in_cm**2) / GRAVITATIONAL_CONSTANT_IN_CGS


def mean_density(
    radius_in_cm: float | NDArray[np.float64],
    surface_gravity_in_cgs: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    return mass(radius_in_cm, surface_gravity_in_cgs) / (
        (4 / 3) * np.pi * radius_in_cm**3
    )


"""
print(
    type_check_arguments(
        mean_density,
        RadiusType(radius=1.0),
        SurfaceGravityType(log10_surface_gravity=3.5),
    )
)

print(
    call_with_type_checks(
        mean_density,
        RadiusType(radius=1.0),
        SurfaceGravityType(log10_surface_gravity=3.5),
    )
)
"""
