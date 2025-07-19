from collections.abc import KeysView

import jax
import numpy as np
import xarray as xr
from jax import numpy as jnp

from potluck.basic_types import PressureDimension
from potluck.constants_and_conversions import (
    BOLTZMANN_CONSTANT_IN_CGS,
    GRAVITATIONAL_CONSTANT_IN_CGS,
)
from potluck.xarray_functional_wrappers import (
    Dimensionalize,
    XarrayStructure,
    set_result_name_and_units,
)


def convert_coordinate_by_level_to_by_layer(coordinate: xr.DataArray) -> xr.DataArray:
    midlayer_values: np.ndarray = np.sqrt(
        coordinate.to_numpy()[1:] * coordinate.to_numpy()[:-1]
    )

    return xr.DataArray(
        data=midlayer_values,
        dims=(coordinate.name,),
        name=coordinate.name,
        attrs=coordinate.attrs,
    )


def convert_data_by_level_to_by_layer(
    xarray_structure: XarrayStructure, coordinate_name: str = "pressure"
) -> XarrayStructure:
    if coordinate_name not in xarray_structure.coords:
        raise ValueError(f"Dataset must have {coordinate_name} as a coordinate.")

    return None


def convert_dataarray_by_pressure_levels_to_pressure_layers(
    dataarray: xr.DataArray, strict: bool = False
) -> xr.DataArray:
    if "pressure" not in dataarray.coords:
        if strict:
            raise ValueError("Dataarray must have pressure as a coordinate.")
        else:
            print("Dataarray does not have pressure as a coordinate.")
            return dataarray

    midlayer_pressures: xr.DataArray = convert_coordinate_by_level_to_by_layer(
        dataarray.pressure
    )

    dataarray_by_layer: xr.DataArray = dataarray.interp(
        pressure=midlayer_pressures
    ).rename(dataarray.name.replace("level", "layer"))

    return dataarray_by_layer


def convert_dataset_by_pressure_levels_to_pressure_layers(
    dataset: xr.Dataset, strict: bool = False
) -> xr.Dataset:
    if "pressure" not in dataset.coords:
        if strict:
            raise ValueError("Dataset must have pressure as a coordinate.")
        else:
            print("Dataset does not have pressure as a coordinate.")
            return dataset

    variable_names: KeysView[str] = dataset.data_vars.keys()

    variable_names_by_layer: dict[str, str] = {
        variable_name: variable_name.replace("level", "layer")
        for variable_name in variable_names
    }

    midlayer_pressures: xr.DataArray = convert_coordinate_by_level_to_by_layer(
        dataset.pressure
    )

    dataset_interpolated_to_layers: xr.Dataset = dataset.interp(
        pressure=midlayer_pressures
    ).rename(variable_names_by_layer)

    return dataset_interpolated_to_layers


def calculate_change_across_pressure_layer(dataarray: xr.DataArray) -> xr.DataArray:
    midlayer_pressures: xr.DataArray = convert_coordinate_by_level_to_by_layer(
        dataarray.pressure
    )

    delta_dataarray: xr.DataArray = (
        dataarray.diff("pressure")
        .assign_coords(pressure=midlayer_pressures)
        .rename(f"delta_{dataarray.name.replace('level', 'layer')}")
    )

    return delta_dataarray


def convert_datatree_by_pressure_levels_to_pressure_layers(
    datatree: xr.DataTree,
) -> xr.DataTree:
    return datatree.map_over_datasets(
        convert_dataset_by_pressure_levels_to_pressure_layers
    )


@set_result_name_and_units(new_name="path_length", units="cm")
def altitudes_by_level_to_path_lengths(
    altitudes_by_level: xr.DataArray,
) -> xr.DataArray:
    return -calculate_change_across_pressure_layer(altitudes_by_level)


def altitudes_by_level_to_by_layer(altitudes_by_level: xr.DataArray) -> xr.DataArray:
    return convert_dataarray_by_pressure_levels_to_pressure_layers(altitudes_by_level)


@set_result_name_and_units(new_name="altitude", units="cm")
@Dimensionalize(
    argument_dimensions=(
        (PressureDimension,),
        (PressureDimension,),
        (PressureDimension,),
        None,
        None,
    ),
    result_dimensions=((PressureDimension,),),
)
@jax.jit
def calculate_altitude_profile(
    log_pressures_in_cgs: np.ndarray[np.float64],
    temperatures_in_K: np.ndarray[np.float64],
    mean_molecular_weights_in_g: np.ndarray[np.float64],
    planet_radius_in_cm: float,
    planet_mass_in_g: float,
) -> np.ndarray[np.float64]:
    def body(carry, x):
        altitude = carry
        log10_pressure_difference = (
            log_pressures_in_cgs[x] - log_pressures_in_cgs[x - 1]
        )
        log_pressure_difference = np.log(10) * log10_pressure_difference
        dlogPdr = (
            GRAVITATIONAL_CONSTANT_IN_CGS
            * planet_mass_in_g
            * mean_molecular_weights_in_g[x]
            / (
                BOLTZMANN_CONSTANT_IN_CGS
                * temperatures_in_K[x]
                * (planet_radius_in_cm + altitude) ** 2
            )
        )
        altitude_difference = log_pressure_difference / dlogPdr
        return (altitude + altitude_difference, altitude + altitude_difference)

    init_altitude = 0.0
    _, altitudes = jax.lax.scan(
        body, init_altitude, np.arange(len(log_pressures_in_cgs) - 1, 0, -1)
    )
    return jnp.append(jnp.array([0.0]), altitudes)[::-1]


@set_result_name_and_units(new_name="altitude", units="cm")
@Dimensionalize(
    argument_dimensions=(
        (PressureDimension,),
        (PressureDimension,),
        (PressureDimension,),
        None,
        None,
    ),
    result_dimensions=((PressureDimension,),),
)
def calculate_altitude_profile_numpy(
    log_pressures_in_cgs: np.ndarray[np.float64],
    temperatures_in_K: np.ndarray[np.float64],
    mean_molecular_weights_in_g: np.ndarray[np.float64],
    planet_radius_in_cm: float,
    planet_mass_in_g: float,
) -> np.ndarray[np.float64]:
    log10_pressure_differences: np.ndarray[np.float64] = (
        log_pressures_in_cgs[1:] - log_pressures_in_cgs[:-1]
    )
    log_pressure_differences: np.ndarray[np.float64] = (
        np.log(10) * log10_pressure_differences
    )

    altitudes: np.ndarray[np.float64] = np.empty_like(log_pressures_in_cgs)
    altitudes[-1] = 0

    for i, (
        log_pressure_difference,
        temperature_in_K,
        mean_molecular_weight_in_g,
    ) in enumerate(
        zip(
            reversed(log_pressure_differences[:]),
            reversed(temperatures_in_K[:]),
            reversed(mean_molecular_weights_in_g[:]),
        ),
        start=1,
    ):
        dlogPdr: float = (
            GRAVITATIONAL_CONSTANT_IN_CGS
            * planet_mass_in_g
            * mean_molecular_weight_in_g
            / (
                BOLTZMANN_CONSTANT_IN_CGS
                * temperature_in_K
                * (planet_radius_in_cm + altitudes[-i]) ** 2
            )
        )

        altitude_difference: float = log_pressure_difference / dlogPdr

        altitudes[-(i + 1)] = altitudes[-i] + altitude_difference

    return altitudes


def impose_upper_limit_on_altitude(
    altitudes: np.ndarray[np.float64], upper_altitude_limit: float
):
    return np.clip(altitudes, a_max=upper_altitude_limit)
