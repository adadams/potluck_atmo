import numpy as np

from potluck.basic_types import WavelengthDimension
from potluck.xarray_functional_wrappers import Dimensionalize


@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension,),
        (WavelengthDimension,),
        None,
        None,
    ),
    result_dimensions=((WavelengthDimension,),),
)
def inflate_errors_by_flux_scaling(
    flux: np.ndarray,
    flux_errors: np.ndarray,
    flux_scaled_error_inflation_factor: float,
    log10_constant_error_inflation_term: float,
) -> np.ndarray:
    inflated_flux_variances: np.ndarray = (
        flux_errors**2
        + (flux_scaled_error_inflation_factor * flux) ** 2
        + 10.0**log10_constant_error_inflation_term
    )

    return inflated_flux_variances**0.5
