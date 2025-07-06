from pathlib import Path

import numpy as np

from constants_and_conversions import SOLAR_RADIUS_IN_CM
from user.input_structs import UserVerticalModelInputs

model_directory: Path = Path(__file__).parent.parent  # NOTE: bodge


################### VERTICAL STRUCTURE ###################
planet_radius_in_cm: float = 0.116 * SOLAR_RADIUS_IN_CM

planet_logg_in_cgs: float = 3.5
planet_gravity_in_cgs: float = 10**planet_logg_in_cgs  # cm/s^2

pressures_by_level: np.ndarray[np.float64] = 10 ** np.linspace(-4.0, 2.5, 71)  # bars
log_pressures_by_level: np.ndarray[np.float64] = np.log10(pressures_by_level)

temperatures_by_level: np.ndarray[np.float64] = np.full_like(pressures_by_level, 1300.0)

mixing_ratios_by_level: dict[str, np.ndarray] = dict(
    h2=np.ones_like(pressures_by_level)
)

user_vertical_inputs: UserVerticalModelInputs = UserVerticalModelInputs(
    planet_radius_in_cm=planet_radius_in_cm,
    planet_gravity_in_cgs=planet_gravity_in_cgs,
    log_pressures_by_level=log_pressures_by_level,
    pressures_by_level=pressures_by_level,
    temperatures_by_level=temperatures_by_level,
    mixing_ratios_by_level=mixing_ratios_by_level,
)
##########################################################
