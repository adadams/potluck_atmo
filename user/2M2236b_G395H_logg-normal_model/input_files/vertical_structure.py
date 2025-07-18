from datetime import datetime

import xarray as xr

from material.mixing_ratios import MixingRatios, generate_uniform_mixing_ratios
from model_builders.default_builders import (
    DefaultFundamentalParameterInputs,
    PressureProfile,
    build_default_fundamental_parameters,
    build_temperature_profile,
)
from temperature.models import generate_piette_model
from temperature.protocols import TemperatureModelInputs
from user.input_structs import UserVerticalModelInputs


def build_vertical_model_inputs(
    fundamental_parameter_inputs: DefaultFundamentalParameterInputs,
    uniform_log_abundances: dict[str, float],
    temperature_model_inputs: TemperatureModelInputs,
    pressure_profile: PressureProfile,
    model_id: str,
    filler_species: str = "h2",
) -> UserVerticalModelInputs:
    mixing_ratios_by_level: MixingRatios = generate_uniform_mixing_ratios(
        uniform_log_abundances=uniform_log_abundances,
        number_of_pressure_levels=len(pressure_profile.pressures_by_level),
        filler_species=filler_species,
    )

    fundamental_parameters_as_xarray: xr.Dataset = build_default_fundamental_parameters(
        fundamental_parameter_inputs
    )
    planet_radius_in_cm_as_xarray: xr.DataArray = (
        fundamental_parameters_as_xarray.planet_radius_in_cm
    )
    planet_gravity_in_cgs_as_xarray: xr.DataArray = (
        fundamental_parameters_as_xarray.planet_gravity_in_cgs
    )

    temperatures_by_level_as_xarray: xr.DataArray = build_temperature_profile(
        temperature_model_constructor=generate_piette_model,
        temperature_model_inputs=temperature_model_inputs,
        pressure_profile=pressure_profile,
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
        coords=pressure_profile.coords,
        attrs={
            "model_ID": model_id,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    default_vertical_structure_as_xarray: xr.Dataset = xr.Dataset(
        data_vars={
            "planet_radius_in_cm": planet_radius_in_cm_as_xarray,
            "planet_gravity_in_cgs": planet_gravity_in_cgs_as_xarray,
            "pressures_by_level": pressure_profile.pressures_by_level,
            "log_pressures_by_level": pressure_profile.log_pressures_by_level,
            "temperatures_by_level": temperatures_by_level_as_xarray,
        },
        coords=pressure_profile.coords,
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
