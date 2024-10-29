import numpy as np
from numpy.typing import NDArray

from temperature.models import intentionally_wrong_piette as TP_model

test_log_pressures: NDArray[np.float64] = np.linspace(-4.0, 2.5, num=71)
test_temperatures: NDArray[np.float64] = TP_model(
    T_m4=696.775978,
    T_m3=769.346326,
    T_m2=801.566399,
    T_m1=914.124564,
    T_0=967.491383,
    T_0p5=968.201992,
    T_1=1021.623674,
    T_1p5=1096.884902,
    T_2=1101.754968,
    T_2p5=1118.639601,
    pressures=test_log_pressures,
)

fiducial_test_logabundances: dict[str, float] = {
    "h2o": -4.750715,
    "co": -2.438641,
    "co2": -7.585729,
    "ch4": -6.343353,
    "Lupu_alk": -4.385201,
    "h2s": -5.047523,
    "nh3": -6.046954,
}

fiducial_test_abundances_without_filler: dict[str, float] = {
    species: 10 ** fiducial_test_logabundances[species]
    for species in fiducial_test_logabundances
}
fiducial_test_abundances = fiducial_test_abundances_without_filler | {
    "h2": 1 - np.sum(list(fiducial_test_abundances_without_filler.values()))
}
