# In your Python model code (e.g., atmosphere_model.py)
from dataclasses import dataclass
from functools import (
    wraps,  # For custom decorator if not using decopatch for SimpleDimensionalizer
)
from typing import Any, Callable, Optional, Sequence

import numpy as np
import xarray as xr


# Assuming Dimensionalize or a simplified version is defined (either from your file or adapted)
# For this example, let's imagine a simplified one:
class MyApplyUfuncWrapper:  # Simplified concept, your Dimensionalize aims for more
    def __init__(
        self,
        input_core_dims,
        output_core_dims,
        vectorize=False,
        exclude_dims=None,
        dask="allowed",
        output_dtypes=None,
        keep_attrs=None,
    ):
        self.input_core_dims = input_core_dims
        self.output_core_dims = output_core_dims
        self.vectorize = vectorize
        self.exclude_dims = exclude_dims if exclude_dims is not None else set()
        self.dask = dask
        self.output_dtypes = output_dtypes
        self.keep_attrs = keep_attrs

    def __call__(self, core_func):
        @wraps(core_func)
        def wrapper(*args, **kwargs_for_core_func):  # Arguments for the core_func
            return xr.apply_ufunc(
                core_func,
                *args,  # These are the xarray DataArray inputs
                kwargs=kwargs_for_core_func,  # Pass through any other kwargs to the core_func
                input_core_dims=self.input_core_dims,
                output_core_dims=self.output_core_dims,
                vectorize=self.vectorize,
                exclude_dims=self.exclude_dims,
                dask=self.dask,
                output_dtypes=self.output_dtypes,
                keep_attrs=self.keep_attrs,
            )

        return wrapper


# Your core computation
def _calculate_humidity_ratio_core(
    pressure_pa: np.ndarray, water_vapor_pressure_pa: np.ndarray, epsilon: float = 0.622
) -> np.ndarray:
    # This function expects NumPy arrays and a scalar
    return epsilon * water_vapor_pressure_pa / (pressure_pa - water_vapor_pressure_pa)


# Python function with "in-line" apply_ufunc metadata via a decorator
@MyApplyUfuncWrapper(  # This is the "in-line" specification
    input_core_dims=[[], []],  # pressure_pa, water_vapor_pressure_pa (element-wise)
    output_core_dims=[[]],
    vectorize=True,  # if the core function isn't a true ufunc but handles arrays
    output_dtypes=[np.float64],
)
# You could chain your @rename_and_unitize here too
# @rename_and_unitize(new_name="mixing_ratio", units="kg/kg")
def calculate_mixing_ratio(
    pressure: xr.DataArray,  # Comes from validated input_ds['air_pressure']
    water_vapor_pressure: xr.DataArray,  # Comes from validated_input_ds['water_vapor_pressure']
    epsilon_ratio: float = 0.622,  # This could be a scalar from TOML, passed as a kwarg
) -> xr.DataArray:
    # The decorator handles calling _calculate_humidity_ratio_core via apply_ufunc.
    # The arguments 'pressure' and 'water_vapor_pressure' are passed as *args to the wrapper,
    # and 'epsilon_ratio' as a **kwarg.
    return _calculate_humidity_ratio_core(
        pressure, water_vapor_pressure, epsilon=epsilon_ratio
    )


# --- In your main orchestration script ---
# ... (load config, load & validate input_ds) ...
# mixing_ratio_da = calculate_mixing_ratio(
#     input_ds["air_pressure_variable_name"], # Name from validated dataset
#     input_ds["h2o_vapor_pressure_variable_name"],
#     epsilon_ratio=config.get("epsilon_parameter", 0.622) # Scalar from TOML config
# )
# # Apply rename_and_unitize explicitly if not part of the decorator chain and you want TOML control:
# if config.output_configs and "mixing_ratio" in config.output_configs:
#     output_cfg = config.output_configs["mixing_ratio"]
#     mixing_ratio_da = mixing_ratio_da.rename(output_cfg.name).assign_attrs(
#         units=output_cfg.attrs.units,
#         unit_base_rep=output_cfg.attrs.unit_base_rep,
#         # ... other attrs
#     )

# Ensure Sequence is imported


@dataclass
class SimpleDimensionalizer:
    input_core_dims: Sequence[Sequence[str]]  # e.g., (("lat", "lon"), ("lat", "lon"))
    output_core_dims: Sequence[Sequence[str]]  # e.g., (("lat", "lon"),)
    vectorize: bool = False  # Default for functions not already ufuncs
    exclude_dims: Optional[set[str]] = None  # Use set for exclude_dims
    dask: str = "forbidden"
    # ... other relevant apply_ufunc parameters

    def __call__(self, function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return xr.apply_ufunc(
                function,
                *args,
                kwargs=kwargs,
                input_core_dims=self.input_core_dims,
                output_core_dims=self.output_core_dims,
                vectorize=self.vectorize,
                exclude_dims=self.exclude_dims
                if self.exclude_dims is not None
                else set(),
                dask=self.dask,
                # ... pass other params
            )

        return wrapper


# Conceptual change
# @function_decorator
# def richer_metadata_assigner(output_name: str, attrs_to_assign: dict, function=DECORATED):
#     @wraps(function)
#     def wrapper(*args, **kwargs):
#         result = function(*args, **kwargs)
#         if isinstance(result, xr.DataArray):
#             result = result.rename(output_name).assign_attrs(attrs_to_assign)
#         # ... careful handling for Dataset ...
#         return result
#     return wrapper
