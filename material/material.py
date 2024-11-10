from typing import Protocol

import msgspec
import numpy as np
from numpy.typing import NDArray

from xarray_functional_wrappers import Dimensionalize
from xarray_serialization import PressureType, WavelengthType


class TwoStreamScatteringCrossSections(msgspec.Struct):
    forward_scattering_crosssections: NDArray[np.float64]
    backward_scattering_crosssections: NDArray[np.float64]


class TwoStreamScatteringCoefficients(msgspec.Struct):
    forward_scattering_coefficients: NDArray[np.float64]
    backward_scattering_coefficients: NDArray[np.float64]


class TwoStreamMaterial(msgspec.Struct):
    name: str
    scattering_coefficients: TwoStreamScatteringCoefficients
    absorption_coefficient: NDArray[np.float64]


class ScatteringParameters(msgspec.Struct):
    g: NDArray[np.float64]
    w0: NDArray[np.float64]


def calculate_scattering_parameters(
    forward_scattering_coefficient: NDArray[np.float64],
    backward_scattering_coefficient: NDArray[np.float64],
    absorption_coefficient: NDArray[np.float64],
) -> ScatteringParameters:
    total_scattering_coefficient: NDArray[np.float64] = (
        forward_scattering_coefficient + backward_scattering_coefficient
    )

    total_extinction_coefficient: NDArray[np.float64] = (
        absorption_coefficient + total_scattering_coefficient
    )

    g: NDArray[np.float64] = (
        forward_scattering_coefficient - backward_scattering_coefficient
    ) / total_scattering_coefficient
    w0: NDArray[np.float64] = (
        total_scattering_coefficient / total_extinction_coefficient
    )

    return ScatteringParameters(g=g, w0=w0)


def crosssections_to_attenutation_coefficients(
    crosssections: NDArray[np.float64],
    number_density: NDArray[np.float64],
) -> NDArray[np.float64]:
    return crosssections * number_density


def attenuation_coefficients_to_optical_depths(
    attenuation_coefficients: NDArray[np.float64], path_length: NDArray[np.float64]
) -> NDArray[np.float64]:
    return attenuation_coefficients * path_length


@Dimensionalize(
    argument_dimensions=(
        (WavelengthType, PressureType),
        (PressureType,),
        (PressureType,),
    ),
    result_dimensions=((WavelengthType, PressureType),),
)
def crosssections_to_optical_depths(
    crosssections: NDArray[np.float64],
    number_density: NDArray[np.float64],
    path_length: NDArray[np.float64],
) -> NDArray[np.float64]:
    return crosssections * number_density * path_length


class MaterialFunction(Protocol):
    def __call__(self) -> TwoStreamMaterial: ...


class GasFunction(MaterialFunction): ...


class CloudFunction(MaterialFunction): ...
