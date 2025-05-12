from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from material.mixing_ratios import uniform_log_mixing_ratios
from user.input_importers import import_model_id
from user.input_structs import UserVerticalModelInputs

current_directory: Path = Path(__file__).parent
model_directory: Path = current_directory.parent  # NOTE: bodge

model_directory_label: str = "R1b_retrieval"

model_id: str = import_model_id(
    model_directory_label=model_directory_label, parent_directory="user"
)

malbec_TP_filepath: Path = Path(__file__).parent / "T3B_malbec_TP.txt"

(
    pressures_by_level,
    temperatures_by_level,
    mmw_by_level,
    altitude_by_level,
    H2_by_level,
    He_by_level,
    H2O_by_level,
    CH4_by_level,
) = np.loadtxt(malbec_TP_filepath).T


pressures_by_level = pressures_by_level[::-1]
temperatures_by_level = temperatures_by_level[::-1]

# H2_by_level = H2_by_level[::-1]
# He_by_level = He_by_level[::-1]
# H2O_by_level = H2O_by_level[::-1]
# CH4_by_level = CH4_by_level[::-1]

################### VERTICAL STRUCTURE ###################
planet_radius_in_cm: float = 1.018 * (3.6318e9 / 2)

planet_gravity_in_cgs: float = 756.4  # cm/s^2

log_pressures_by_level: NDArray[np.float64] = np.log10(pressures_by_level)

uniform_log_abundances: dict[str, float] = {"h2o": -3.171, "ch4": -3.318}

mixing_ratios_by_level: dict[str, np.ndarray] = {
    "h2he": H2_by_level + He_by_level,
    "h2o": H2O_by_level,
    "ch4": CH4_by_level,
}

default_vertical_structure: UserVerticalModelInputs = UserVerticalModelInputs(
    planet_radius_in_cm=planet_radius_in_cm,
    planet_gravity_in_cgs=planet_gravity_in_cgs,
    log_pressures_by_level=log_pressures_by_level,
    pressures_by_level=pressures_by_level,
    temperatures_by_level=temperatures_by_level,
    mixing_ratios_by_level=mixing_ratios_by_level,
)
##########################################################


def build_uniform_model_inputs(
    uniform_log_abundances: dict[str, float],
    planet_radius_in_cm: float = planet_radius_in_cm,
    planet_gravity_in_cgs: float = planet_gravity_in_cgs,
    log_pressures_by_level: NDArray[np.float64] = log_pressures_by_level,
    pressures_by_level: NDArray[np.float64] = pressures_by_level,
    temperatures_by_level: NDArray[np.float64] = temperatures_by_level,
    filler_species: str = "h2he",
) -> UserVerticalModelInputs:
    mixing_ratios_by_level: dict[str, NDArray[np.float64]] = uniform_log_mixing_ratios(
        uniform_log_abundances=uniform_log_abundances,
        number_of_pressure_levels=len(pressures_by_level),
        filler_species=filler_species,
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

    # return UserVerticalModelInputs(
    #    planet_radius_in_cm=planet_radius_in_cm,
    #    planet_gravity_in_cgs=planet_gravity_in_cgs,
    #    log_pressures_by_level=log_pressures_by_level,
    #    pressures_by_level=pressures_by_level,
    #    temperatures_by_level=temperatures_by_level,
    #    mixing_ratios_by_level=mixing_ratios_by_level,
    # )
    return default_vertical_structure_datatree


default_uniform_vertical_structure: UserVerticalModelInputs = (
    build_uniform_model_inputs(
        uniform_log_abundances={
            "h2o": -3.171,
            "ch4": -3.318,
        }
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
