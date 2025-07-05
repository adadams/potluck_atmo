from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from constants_and_conversions import EARTH_RADIUS_IN_CM
from material.mixing_ratios import generate_uniform_mixing_ratios
from temperature.models import piette as TP_model
from user.input_importers import import_model_id
from user.input_structs import UserVerticalModelInputs

current_directory: Path = Path(__file__).parent
model_directory: Path = current_directory.parent

model_directory_label: str = "2M2236b_HK+G395H_logg-normal"

model_id: str = import_model_id(
    model_directory_label=model_directory_label, parent_directory="user"
)

################### VERTICAL STRUCTURE ###################
planet_radius_in_cm: float = 9.655182368 * EARTH_RADIUS_IN_CM

planet_logg_in_cgs: float = 4.143462527
planet_gravity_in_cgs: float = 10**planet_logg_in_cgs  # cm/s^2

log_pressures_by_level: NDArray[np.float64] = np.linspace(-4.0, 2.5, num=71)
pressures_by_level: NDArray[np.float64] = 10**log_pressures_by_level

temperatures_by_level: NDArray[np.float64] = TP_model(
    T_m4=655.8156951,
    T_m3=1005.147353,
    T_m2=1046.759478,
    T_m1=1108.618473,
    T_0=1299.807719,
    T_0p5=1346.346164,
    T_1=1367.168578,
    T_1p5=1374.087072,
    T_2=1574.039082,
    T_2p5=1648.699968,
    log_pressures=log_pressures_by_level,
)


uniform_log_abundances: dict[str, float] = {
    "h2o": -3.429731304,
    "co": -3.032072016,
    "co2": -7.380174847,
    "ch4": -5.666933008,
    "Lupu_alk": -3.155131991336514,
    "h2s": -4.763003846,
    "nh3": -5.40420758,
}

mixing_ratios_by_level: dict[str, np.ndarray] = generate_uniform_mixing_ratios(
    uniform_log_abundances=uniform_log_abundances,
    number_of_pressure_levels=len(pressures_by_level),
    filler_species="h2",
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

planet_radius_in_cm_as_xarray: xr.DataArray = xr.DataArray(
    data=planet_radius_in_cm,
    dims=tuple(),
    attrs={"units": "cm"},
)

planet_gravity_in_cgs_as_xarray: xr.DataArray = xr.DataArray(
    data=planet_gravity_in_cgs,
    dims=tuple(),
    attrs={"units": "cm/s^2"},
)

pressures_by_level_as_xarray: xr.DataArray = xr.DataArray(
    data=pressures_by_level,
    dims=("pressure",),
    attrs={"units": "bar"},
)

temperatures_by_level_as_xarray: xr.DataArray = xr.DataArray(
    data=temperatures_by_level,
    dims=("pressure",),
    attrs={"units": "K"},
)

log_pressures_by_level_as_xarray: xr.DataArray = xr.DataArray(
    data=log_pressures_by_level,
    dims=("pressure",),
    attrs={"units": "log(bar)"},
)

default_mixing_ratios_by_level_as_xarray: xr.Dataset = xr.Dataset(
    data_vars={
        mixing_ratio_name: xr.DataArray(
            data=mixing_ratios_by_level[mixing_ratio_name],
            dims=("pressure",),
            attrs={"units": "mol/mol"},
        )
        for mixing_ratio_name in mixing_ratios_by_level
    },
    coords={
        "pressure": pressures_by_level_as_xarray,
    },
    attrs={
        "model_ID": model_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
)


default_vertical_structure_as_xarray: xr.Dataset = xr.Dataset(
    data_vars={
        "planet_radius_in_cm": planet_radius_in_cm_as_xarray,
        "planet_gravity_in_cgs": planet_gravity_in_cgs_as_xarray,
        "pressures_by_level": pressures_by_level_as_xarray,
        "log_pressures_by_level": log_pressures_by_level_as_xarray,
        "temperatures_by_level": temperatures_by_level_as_xarray,
    },
    coords={
        "pressure": pressures_by_level_as_xarray,
    },
    attrs={
        "model_ID": model_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
)

chemistry_node: xr.DataTree = xr.DataTree(
    name="chemistry",
    dataset=default_mixing_ratios_by_level_as_xarray,
)

default_vertical_structure_datatree: xr.DataTree = xr.DataTree(
    name="vertical_structure",
    dataset=default_vertical_structure_as_xarray,
    children={"chemistry": chemistry_node},
)


default_vertical_structure_datatree.to_netcdf(
    model_directory
    / "intermediate_outputs"
    / f"{model_id}_vertical_structure_as_datatree.nc"
)

default_vertical_structure_as_xarray.to_netcdf(
    model_directory / "intermediate_outputs" / f"{model_id}_vertical_structure.nc"
)

default_mixing_ratios_by_level_as_xarray.to_netcdf(
    model_directory / "intermediate_outputs" / f"{model_id}_mixing_ratios.nc"
)
