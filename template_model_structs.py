from functools import partial
from pathlib import Path
from typing import Any, Final, Optional, TypeAlias

import msgspec
import numpy as np
import xarray as xr
from jax import numpy as jnp
from numpy.typing import ArrayLike
from pint import UnitRegistry

from basic_types import PressureDimension
from material.mixing_ratios import (
    UniformLogMixingRatios,
    generate_uniform_mixing_ratios,
)
from temperature.protocols import (
    TemperatureModel,
    TemperatureModelConstructor,
    TemperatureModelInputs,
)
from xarray_functional_wrappers import Dimensionalize, rename_and_unitize
from xarray_serialization import AttrsType

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory

DEFAULT_UNITS_SYSTEM: Final[str] = "cgs"
ureg: UnitRegistry = UnitRegistry(system=DEFAULT_UNITS_SYSTEM)
ureg.load_definitions(str(project_directory / "additional_units.txt"))

FundamentalParameters: TypeAlias = xr.Dataset
PressureProfile: TypeAlias = xr.Dataset
TemperatureProfile: TypeAlias = xr.Dataset
GasChemistryProfile: TypeAlias = xr.Dataset


class FundamentalParameterInputs(msgspec.Struct):
    planet_radius: float
    planet_gravity: float
    radius_units: str = "cm"
    gravity_units: str = "cm/s^2"
    additional_attributes: Optional[AttrsType]


def build_default_fundamental_parameters(
    fundamental_parameter_inputs: FundamentalParameterInputs,
) -> FundamentalParameters:
    planet_radius_in_cm = fundamental_parameter_inputs.planet_radius * ureg(
        fundamental_parameter_inputs.radius_units
    ).to("cm")
    planet_gravity_in_cgs = fundamental_parameter_inputs.planet_gravity * ureg(
        fundamental_parameter_inputs.gravity_units
    ).to("cm/s^2")
    additional_attributes: dict[str, Any] = (
        fundamental_parameter_inputs.additional_attributes
        if fundamental_parameter_inputs.additional_attributes is not None
        else {}
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

    return xr.Dataset(
        data_vars={
            "planet_radius_in_cm": planet_radius_in_cm_as_xarray,
            "planet_gravity_in_cgs": planet_gravity_in_cgs_as_xarray,
        },
        coords={},
        attrs=additional_attributes,
    )


class PressureProfileInputs(msgspec.Struct):
    log_pressures_by_level: ArrayLike
    units: str = "bar"
    additional_attributes: Optional[dict] = None


def build_pressure_profile_from_log_pressures(
    pressure_profile_inputs: PressureProfileInputs,
) -> PressureProfile:
    pressures_by_level: np.ndarray = np.power(
        10, pressure_profile_inputs.log_pressures_by_level
    ) * ureg(pressure_profile_inputs.units).to("barye")
    log_pressures_by_level: np.ndarray = np.log10(pressures_by_level)

    additional_attributes: dict[str, Any] = (
        pressure_profile_inputs.additional_attributes
        if pressure_profile_inputs.additional_attributes is not None
        else {}
    )

    pressures_by_level_as_xarray: xr.DataArray = xr.DataArray(
        name="pressure",
        data=jnp.array(pressures_by_level),
        dims=("pressure",),
        attrs={"units": "barye"},
    )

    log_pressures_by_level_as_xarray: xr.DataArray = xr.DataArray(
        name="log10_pressure",
        data=jnp.array(log_pressures_by_level),
        dims=("pressure",),
        coords={"pressure": pressures_by_level_as_xarray},
        attrs={"units": "log10(barye)"},
    )

    return xr.Dataset(
        data_vars={
            "pressure": pressures_by_level_as_xarray,
            "log10_pressure": log_pressures_by_level_as_xarray,
        },
        coords={"pressure": pressures_by_level_as_xarray},
        attrs=additional_attributes,
    )


class UniformGasChemistryInputs(msgspec.Struct):
    log_mixing_ratios: UniformLogMixingRatios
    filler_species: Optional[str] = None
    additional_attributes: Optional[AttrsType] = None


def build_uniform_gas_chemistry(
    gas_chemistry_inputs: UniformGasChemistryInputs, pressure_profile: PressureProfile
) -> GasChemistryProfile:
    log_mixing_ratios: UniformLogMixingRatios = gas_chemistry_inputs.log_mixing_ratios
    filler_species: str = gas_chemistry_inputs.filler_species
    additional_attributes: dict[str, Any] = (
        gas_chemistry_inputs.additional_attributes
        if gas_chemistry_inputs.additional_attributes is not None
        else {}
    )

    number_of_pressure_levels: int = len(pressure_profile.pressures_by_level)

    uniform_mixing_ratios_by_level: dict[str, np.ndarray[np.float64]] = (
        generate_uniform_mixing_ratios(
            uniform_log_abundances=log_mixing_ratios,
            number_of_pressure_levels=number_of_pressure_levels,
            filler_species=filler_species,
        )
    )

    mixing_ratios_by_level_as_xarray: xr.Dataset = xr.Dataset(
        data_vars={
            mixing_ratio_name: xr.DataArray(
                data=uniform_mixing_ratios_by_level[mixing_ratio_name],
                dims=("pressure",),
                attrs={"units": "mol/mol"},
            )
            for mixing_ratio_name in uniform_mixing_ratios_by_level
        },
        coords=pressure_profile.coords,
        attrs=additional_attributes,
        # attrs={
        #     "model_ID": model_id,
        #     "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # },
    )

    return mixing_ratios_by_level_as_xarray


def build_temperature_profile(
    temperature_model_constructor: TemperatureModelConstructor,
    temperature_model_inputs: TemperatureModelInputs,
    pressure_profile: PressureProfile,
) -> TemperatureProfile:
    temperature_model: TemperatureModel = partial(
        temperature_model_constructor,
        temperature_model_inputs=temperature_model_inputs,
    )

    temperature_model_for_xarray: TemperatureModel = rename_and_unitize(
        func=Dimensionalize(
            argument_dimensions=((PressureDimension,),),
            result_dimension=(PressureDimension),
        )(temperature_model),
        new_name="temperature",
        units="K",
    )

    temperatures_by_level_as_xarray: xr.DataArray = temperature_model_for_xarray(
        profile_log_pressures=pressure_profile.log_pressures_by_level
    )

    return temperatures_by_level_as_xarray
