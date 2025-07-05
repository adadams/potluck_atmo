from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from constants_and_conversions import EARTH_RADIUS_IN_CM
from material.mixing_ratios import generate_uniform_mixing_ratios
from temperature.models import piette as TP_model
from temperature.models_experimental_uniform import piette as retrieval_TP_model
from user.input_importers import import_model_id
from user.input_structs import UserVerticalModelInputs

current_directory: Path = Path(__file__).parent
model_directory: Path = current_directory.parent

model_directory_label: str = "2M2236b_G395H_logg-free"

model_id: str = import_model_id(
    model_directory_label=model_directory_label, parent_directory="user"
)

################### VERTICAL STRUCTURE ###################
planet_radius_in_cm: float = 8.184222081 * EARTH_RADIUS_IN_CM

planet_logg_in_cgs: float = 5.449444667
planet_gravity_in_cgs: float = 10**planet_logg_in_cgs  # cm/s^2

log_pressures_by_level: NDArray[np.float64] = np.linspace(-4.0, 2.5, num=71)
pressures_by_level: NDArray[np.float64] = 10**log_pressures_by_level

temperatures_by_level: NDArray[np.float64] = TP_model(
    T_m4=524.9407393,
    T_m3=682.5744882,
    T_m2=815.3428307,
    T_m1=896.0200908,
    T_0=1096.041202,
    T_0p5=1286.48834,
    T_1=1383.103453,
    T_1p5=1421.808017,
    T_2=1734.062437,
    T_2p5=1734.062437,
    # T_2p5=2173.140854,
    log_pressures=log_pressures_by_level,
)


uniform_log_abundances: dict[str, float] = {
    "h2o": -4.682117741,
    "co": -4.072256899,
    "co2": -6.811546908,
    "ch4": -6.280808688,
    "Lupu_alk": -3.155131991,
    "h2s": -5.360058264,
    "nh3": -5.686835907,
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

log_pressures_by_level_as_xarray: xr.DataArray = xr.DataArray(
    data=log_pressures_by_level,
    dims=("pressure",),
    attrs={"units": "log(bar)"},
)

"""
temperatures_by_level_as_xarray: xr.DataArray = (
    xr.apply_ufunc(
        retrieval_TP_model,
        0.2889458091719745,
        np.array([0.11159102, 0.02182628, 0.12510834, 0.10768672, 0.01539343]),
        np.array([0.02514635, 0.01982915, 0.06249186, 0.32445998]),
        # np.random.uniform(low=0, high=1),  # initial_temp_sample,
        # np.random.uniform(low=0, high=1, size=5),  # proportions_down,
        # np.random.uniform(low=0, high=1, size=4),  # proportions_up,
        log_pressures_by_level_as_xarray,
        input_core_dims=[[], ["downward"], ["upward"], ["pressure"]],
        output_core_dims=[["pressure"]],
        exclude_dims={"downward", "upward"},
        vectorize=True,
        kwargs={"lower_bound": 75.0, "upper_bound": 3975.0},
    )
    .assign_attrs(
        units="K",
    )
    .rename("temperature")
)

print(f"{temperatures_by_level_as_xarray=}")
"""

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


def build_uniform_model_inputs(
    uniform_log_abundances: dict[str, float],
    initial_temp_sample: float,  # [75, 3975]
    proportions_down: np.ndarray,  # shape = (5,), all [0, 1]
    proportions_up: np.ndarray,  # shape = (4,), all [0, 1]
    planet_radius_in_cm: float = planet_radius_in_cm,
    planet_gravity_in_cgs: float = planet_gravity_in_cgs,
    log_pressures_by_level: NDArray[np.float64] = log_pressures_by_level,
    pressures_by_level: NDArray[np.float64] = pressures_by_level,
    # temperatures_by_level: NDArray[np.float64] = temperatures_by_level,
    filler_species: str = "h2only",
) -> UserVerticalModelInputs:
    mixing_ratios_by_level: dict[str, NDArray[np.float64]] = (
        generate_uniform_mixing_ratios(
            uniform_log_abundances=uniform_log_abundances,
            number_of_pressure_levels=len(pressures_by_level),
            filler_species=filler_species,
        )
    )

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

    temperatures_by_level_as_xarray: xr.DataArray = xr.apply_ufunc(
        retrieval_TP_model,
        initial_temp_sample,
        proportions_down,
        proportions_up,
        log_pressures_by_level,
        input_core_dims=[[], [], [], ["pressure"]],
        output_core_dims=[["pressure"]],
        vectorize=True,
        kwargs={"lower_bound": 75.0, "upper_bound": 4000.0},
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

    return default_vertical_structure_datatree
