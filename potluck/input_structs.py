from typing import NamedTuple, Protocol, TypeAlias

import xarray as xr

XarrayValue: TypeAlias = xr.DataArray  # storing a single value (0-D) as a DataArray


class VerticalModelDataset(Protocol):
    planet_radius_in_cm: XarrayValue
    planet_gravity_in_cgs: XarrayValue
    pressures_by_level: xr.DataArray
    log_pressures_by_level: xr.DataArray
    temperatures_by_level: xr.DataArray
    mixing_ratios_by_level: xr.DataArray


class UserVerticalModelInputs(NamedTuple):
    planet_radius_in_cm: float
    planet_gravity_in_cgs: float
    log_pressures_by_level: xr.DataArray
    pressures_by_level: xr.DataArray
    temperatures_by_level: xr.DataArray
    mixing_ratios_by_level: xr.DataArray


class UserForwardModelInputs(NamedTuple):
    vertical_inputs: UserVerticalModelInputs
    crosssection_catalog: xr.Dataset
    output_wavelengths: xr.DataArray
    path_lengths_by_layer: xr.DataArray
    altitudes_by_layer: xr.DataArray
    distance_to_system_in_cm: float
    stellar_radius_in_cm: float
