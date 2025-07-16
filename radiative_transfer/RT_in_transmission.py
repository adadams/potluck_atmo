import numpy as np

from basic_types import PressureDimension, WavelengthDimension
from xarray_functional_wrappers import Dimensionalize, set_result_name_and_units


@set_result_name_and_units(new_name="transit_depth", units="dimensionless")
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
def calculate_transmission_spectrum(
    cumulative_optical_depth: np.ndarray[np.float64],
    path_lengths: np.ndarray[np.float64],
    altitudes: np.ndarray[np.float64],
    stellar_radius_in_cm: float,
    planet_radius_in_cm: float,
) -> np.ndarray[np.float64]:
    solid_planet_disk_area: float = np.pi * planet_radius_in_cm**2
    solid_stellar_disk_area: float = np.pi * stellar_radius_in_cm**2

    planet_area_with_atmosphere: np.ndarray[np.float64] = (
        solid_planet_disk_area
        + np.sum(
            (1 - np.exp(-cumulative_optical_depth))
            * path_lengths
            * 2
            * np.pi
            * (planet_radius_in_cm + altitudes),
            axis=-1,
        )
    )

    return planet_area_with_atmosphere / solid_stellar_disk_area
