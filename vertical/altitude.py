import numpy as np
import xarray as xr

from constants_and_conversions import (
    BOLTZMANN_CONSTANT_IN_CGS,
    GRAVITATIONAL_CONSTANT_IN_CGS,
)
from xarray_functional_wrappers import Dimensionalize
from xarray_serialization import PressureDimension


def convert_pressure_coordinate_by_level_to_by_layer(
    dataset: xr.DataArray,
) -> xr.DataArray:
    if "pressure" not in dataset.coords:
        raise ValueError("Dataset must have pressure as a coordinate.")

    midlayer_pressures: np.ndarray = np.sqrt(
        dataset.pressure.to_numpy()[1:] * dataset.pressure.to_numpy()[:-1]
    )

    return xr.DataArray(
        data=midlayer_pressures,
        dims=("pressure",),
        name="pressure",
        attrs={"units": dataset.pressure.attrs["units"]},
    )


def convert_dataset_by_pressure_levels_to_pressure_layers(
    dataset: xr.Dataset,
) -> xr.Dataset:
    midlayer_pressures: xr.DataArray = convert_pressure_coordinate_by_level_to_by_layer(
        dataset
    )

    return dataset.interp(pressure=midlayer_pressures)


def altitudes_by_level_to_path_lengths(
    altitudes_by_level: xr.DataArray,
) -> xr.DataArray:
    path_lengths: xr.DataArray = -altitudes_by_level.diff("pressure")

    midlayer_pressures: xr.DataArray = convert_pressure_coordinate_by_level_to_by_layer(
        altitudes_by_level
    )

    return path_lengths.assign_coords(pressure=midlayer_pressures)


def altitudes_by_level_to_by_layer(
    altitudes_by_level: xr.DataArray,
) -> xr.DataArray:
    midlayer_pressures: xr.DataArray = convert_pressure_coordinate_by_level_to_by_layer(
        altitudes_by_level
    )

    return altitudes_by_level.interp(pressure=midlayer_pressures)


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
def calculate_altitude_profile(
    log_pressures_in_cgs: np.ndarray[np.float64],
    temperatures_in_K: np.ndarray[np.float64],
    mean_molecular_weights_in_g: np.ndarray[np.float64],
    planet_radius_in_cm: float,
    planet_mass_in_g: float,
) -> np.ndarray[np.float64]:
    log_pressures_in_cgs: np.ndarray[np.float64] = log_pressures_in_cgs + 6

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
