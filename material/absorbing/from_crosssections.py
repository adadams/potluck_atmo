import numpy as np
from numpy.typing import NDArray

from xarray_functional_wrappers import Dimensionalize
from xarray_serialization import PressureType, SpeciesType, WavelengthType


@Dimensionalize(
    argument_dimensions=(
        (WavelengthType, PressureType, SpeciesType),
        (PressureType, SpeciesType),
    ),
    result_dimensions=((WavelengthType, PressureType, SpeciesType),),
)
def crosssections_to_attenuation_coefficients(
    crosssections: NDArray[np.float64],
    number_density: NDArray[np.float64],
) -> NDArray[np.float64]:
    return crosssections * number_density


@Dimensionalize(
    argument_dimensions=((WavelengthType, PressureType), (PressureType,)),
    result_dimensions=((WavelengthType, PressureType),),
)
def attenuation_coefficients_to_optical_depths(
    attenuation_coefficients: NDArray[np.float64], path_length: NDArray[np.float64]
) -> NDArray[np.float64]:
    return attenuation_coefficients * path_length


@Dimensionalize(
    argument_dimensions=(
        (WavelengthType, SpeciesType, PressureType),
        (SpeciesType, PressureType),
        (PressureType,),
    ),
    result_dimensions=((WavelengthType, SpeciesType, PressureType),),
)
def crosssections_to_optical_depths(
    crosssections: NDArray[np.float64],
    number_density: NDArray[np.float64],
    path_length: NDArray[np.float64],
) -> NDArray[np.float64]:
    return crosssections * number_density * path_length
