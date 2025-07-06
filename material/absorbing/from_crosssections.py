import numpy as np

from xarray_functional_wrappers import Dimensionalize, rename_and_unitize
from xarray_serialization import (
    PressureDimension,
    SpeciesDimension,
    WavelengthDimension,
)


@rename_and_unitize(new_name="attenuation_coefficients", units="cm^-1")
@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension, PressureDimension, SpeciesDimension),
        (PressureDimension, SpeciesDimension),
    ),
    result_dimensions=((WavelengthDimension, PressureDimension, SpeciesDimension),),
)
def crosssections_to_attenuation_coefficients(
    crosssections: np.ndarray[np.float64],
    number_density: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    return crosssections * number_density


@rename_and_unitize(new_name="optical_depth", units="dimensionless")
@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension, PressureDimension),
        (PressureDimension,),
    ),
    result_dimensions=((WavelengthDimension, PressureDimension),),
)
def attenuation_coefficients_to_optical_depths(
    attenuation_coefficients: np.ndarray[np.float64],
    path_length: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    return attenuation_coefficients * path_length


@rename_and_unitize(new_name="optical_depth", units="dimensionless")
@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension, SpeciesDimension, PressureDimension),
        (SpeciesDimension, PressureDimension),
        (PressureDimension,),
    ),
    result_dimensions=((WavelengthDimension, SpeciesDimension, PressureDimension),),
)
def crosssections_to_optical_depths(
    crosssections: np.ndarray[np.float64],
    number_density: np.ndarray[np.float64],
    path_length: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    return crosssections * number_density * path_length
