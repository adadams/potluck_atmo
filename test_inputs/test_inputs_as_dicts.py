from typing import Final

import numpy as np
from numpy.typing import NDArray

from density import mass
from temperature.models import intentionally_wrong_piette as TP_model

EARTH_RADIUS_IN_CM: Final[float] = 6.371e8

test_planet_radius: float = 8.184222080729041 * EARTH_RADIUS_IN_CM

test_planet_logg: float = 5.449444667355326
test_planet_gravity: float = 10**test_planet_logg  # cm/s^2

test_planet_mass: float = mass(
    radius_in_cm=test_planet_radius, surface_gravity_in_cgs=test_planet_gravity
)

test_log_pressures: NDArray[np.float64] = np.linspace(-4.0, 2.5, num=71)

test_temperatures: NDArray[np.float64] = TP_model(
    T_m4=524.9407392885498,
    T_m3=682.574488154638,
    T_m2=815.3428307467611,
    T_m1=896.0200907958686,
    T_0=1096.041201994001,
    T_0p5=1286.4883401241386,
    T_1=1383.103452573262,
    T_1p5=1421.8080170773437,
    T_2=1734.0624370129008,
    T_2p5=2173.1408536030112,
    pressures=test_log_pressures,
)

fiducial_test_logabundances: dict[str, float] = {
    "h2o": -4.682117740999784,
    "co": -4.072256898649666,
    "co2": -6.8115469084311355,
    "ch4": -6.280808688108694,
    "Lupu_alk": -3.155131991336514,
    "h2s": -5.360058264487807,
    "nh3": -5.686835906893023,
}

fiducial_test_abundances_without_filler: dict[str, float] = {
    species: 10 ** fiducial_test_logabundances[species]
    for species in fiducial_test_logabundances
}
fiducial_test_abundances = fiducial_test_abundances_without_filler | {
    "h2": 1 - np.sum(list(fiducial_test_abundances_without_filler.values()))
}
