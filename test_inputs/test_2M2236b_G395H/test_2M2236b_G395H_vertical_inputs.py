import numpy as np
from numpy.typing import NDArray

from constants_and_conversions import EARTH_RADIUS_IN_CM
from material.gases.molecular_metrics import (
    MoleculeMetrics,
    calculate_cumulative_molecular_metrics,
    calculate_molecular_metrics,
    calculate_weighted_molecular_weights,
)
from material.mixing_ratios import (
    log_abundances_to_mixing_ratios,
    uniform_log_abundances_to_log_abundances_by_level,
)
from temperature.models import piette as TP_model
from test_inputs.test_data_structures.input_structs import UserVerticalModelInputs

################### VERTICAL STRUCTURE ###################
planet_radius_in_cm: float = 8.184222080729041 * EARTH_RADIUS_IN_CM

planet_logg_in_cgs: float = 5.449444667355326
planet_gravity_in_cgs: float = 10**planet_logg_in_cgs  # cm/s^2

log_pressures_by_level: NDArray[np.float64] = np.linspace(-4.0, 2.5, num=71)
pressures_by_level: NDArray[np.float64] = 10**log_pressures_by_level

temperatures_by_level: NDArray[np.float64] = TP_model(
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
    log_pressures=log_pressures_by_level,
)

# model_setup: dict[str, str] = {"mixing_ratios": "uniform_log_abundances"}

# mixing_ratios: dict = {"uniform_log_abundances": uniform_log_abundances}

uniform_log_abundances: dict[str, float] = {
    "h2o": -4.682117740999784,
    "co": -4.072256898649666,
    "co2": -6.8115469084311355,
    "ch4": -6.280808688108694,
    "Lupu_alk": -3.155131991336514,
    "h2s": -5.360058264487807,
    "nh3": -5.686835906893023,
}

mixing_ratios_by_level: dict[str, np.ndarray] = (
    uniform_log_abundances_to_log_abundances_by_level(
        uniform_log_abundances=log_abundances_to_mixing_ratios(uniform_log_abundances),
        number_of_pressure_levels=len(pressures_by_level),
    )
)

molecular_metrics: MoleculeMetrics = calculate_molecular_metrics(
    mixing_ratios=mixing_ratios_by_level
)

weighted_molecular_weights: NDArray[np.float64] = calculate_weighted_molecular_weights(
    mixing_ratios=mixing_ratios_by_level
)

cumulative_molecular_metrics: MoleculeMetrics = calculate_cumulative_molecular_metrics(
    mixing_ratios_by_level=mixing_ratios_by_level,
    molecular_weights_by_level=weighted_molecular_weights,
    mean_molecular_weight_by_level=molecular_metrics.mean_molecular_weight,
    pressure_in_cgs=pressures_by_level,
    temperatures_in_K=temperatures_by_level,
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
