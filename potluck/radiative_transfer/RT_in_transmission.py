import jax
import numpy as np
from jax import numpy as jnp

from potluck.basic_types import PressureDimension, WavelengthDimension
from potluck.xarray_functional_wrappers import Dimensionalize, set_result_name_and_units


@set_result_name_and_units(result_names="transit_depth", units="dimensionless")
@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension, PressureDimension),
        (PressureDimension,),
        (PressureDimension,),
        None,
        None,
    ),
    result_dimensions=((WavelengthDimension,),),
)
@jax.jit
def calculate_transmission_spectrum(
    cumulative_optical_depth: np.ndarray[np.float64],
    path_lengths: np.ndarray[np.float64],
    altitudes: np.ndarray[np.float64],
    stellar_radius_in_cm: float,
    planet_radius_in_cm: float,
) -> np.ndarray[np.float64]:
    solid_planet_disk_area: float = jnp.pi * planet_radius_in_cm**2
    solid_stellar_disk_area: float = jnp.pi * stellar_radius_in_cm**2

    planet_area_with_atmosphere: np.ndarray[np.float64] = (
        solid_planet_disk_area
        + jnp.sum(
            (1 - jnp.exp(-cumulative_optical_depth))
            * path_lengths
            * 2
            * jnp.pi
            * (planet_radius_in_cm + altitudes),
            axis=-1,
        )
    )

    return planet_area_with_atmosphere / solid_stellar_disk_area
