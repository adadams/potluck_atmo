from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from constants_and_conversions import SOLAR_RADIUS_IN_CM
from user.input_structs import UserVerticalModelInputs

model_directory: Path = Path(__file__).parent.parent  # NOTE: bodge

reference_model_filepath: Path = model_directory / "ATMO_Phillips_1300K_logg3.00_CEQ.nc"
reference_model: xr.Dataset = xr.open_dataset(reference_model_filepath)

################### VERTICAL STRUCTURE ###################
planet_radius_in_cm: float = 0.116 * SOLAR_RADIUS_IN_CM

planet_logg_in_cgs: float = 3.0
planet_gravity_in_cgs: float = 10**planet_logg_in_cgs  # cm/s^2

pressures_by_level: NDArray[np.float64] = reference_model.pressure
log_pressures_by_level: NDArray[np.float64] = np.log10(pressures_by_level)

temperatures_by_level: NDArray[np.float64] = reference_model.temperature

mixing_ratios_by_level: dict[str, np.ndarray] = dict(
    h2=reference_model.H2.to_numpy(),
    he=reference_model.He.to_numpy(),
    co=reference_model.CO.to_numpy(),
    h2o=reference_model.H2O.to_numpy(),
    ch4=reference_model.CH4.to_numpy(),
    n2=reference_model.N2.to_numpy(),
    Lupu_alk=reference_model.Mg.to_numpy(),
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
