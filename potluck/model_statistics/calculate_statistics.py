import numpy as np

from basic_types import WavelengthDimension
from xarray_functional_wrappers import Dimensionalize


@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension,),
        (WavelengthDimension,),
        (WavelengthDimension,),
    ),
    result_dimensions=(None,),
)
def calculate_log_likelihood(
    model: np.ndarray, data: np.ndarray, data_error: np.ndarray
) -> float:
    return -0.5 * np.sum(
        ((data - model) / data_error) ** 2 + np.log(2 * np.pi * data_error**2)
    )


@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension,),
        (WavelengthDimension,),
        (WavelengthDimension,),
        None,
    ),
    result_dimensions=(None,),
)
def calculate_reduced_chi_squared_statistic(
    model: np.ndarray,
    data: np.ndarray,
    data_error: np.ndarray,
    number_of_free_parameters: int,
):
    return np.sum((data - model) ** 2 / data_error**2) / (
        len(data) - number_of_free_parameters
    )
