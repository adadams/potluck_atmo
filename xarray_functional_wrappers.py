from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial, reduce, wraps
from inspect import signature
from pathlib import Path
from typing import Any, NamedTuple, Optional, Protocol, TypeAlias

import xarray as xr
from decopatch import DECORATED, function_decorator

from xarray_serialization import (
    ArgumentDimensionType,
    DimensionAnnotation,
    XarrayDimension,
)

# NOTE to self: assess whether we can develop a version of these that uses xr.apply_ufunc;
# maybe partialing apply_ufunc with everything except *args. Can we get away with this without kwargs?
# Could possibly do this with a decorator.


def compose(*functions: Sequence[Callable]) -> Callable:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


class OperatesOnVariableinDataset(Protocol):
    def __call__(self, dataset: xr.Dataset) -> xr.DataArray:
        pass


class OperatesOnDataArray(Protocol):
    def __call__(self, dataset: xr.DataArray) -> xr.DataArray:
        pass


class OperatesOnDataset(Protocol):
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset: ...


"""
Sketching out the structure of a FunctionDimensionType.
a(
  b(
      c("wavelength", d(e("[length]", 1)e,)d )c,
      c("pressure", d(e("[mass]", 1)e, e("[time]", -2)e, e("[length]", -1)e)d )c
  )b,

  b(
      c("wavelength", d(e("[length]", 1)e,)d )c,
  )b
)a
"""


class NamesAndDimensionalities(NamedTuple):
    dimension_names: XarrayDimension
    dimensionalities: DimensionAnnotation


def separate_argument_dimension_into_names_and_dimensions(
    arguments_dimensions_set: ArgumentDimensionType,
) -> NamesAndDimensionalities:
    arguments_dimensions_names: list[Optional[XarrayDimension]] = []
    arguments_dimensions_dimensions: list[Optional[DimensionAnnotation]] = []

    for argument_dimensions_set in arguments_dimensions_set:
        if argument_dimensions_set:
            argument_dimension_names: list[str] = [
                argument_dimension_name
                for argument_dimension_name, _ in argument_dimensions_set
            ]

            argument_dimension_dimensions: list[DimensionAnnotation] = [
                argument_dimension_dimension
                for _, argument_dimension_dimension in argument_dimensions_set
            ]

            arguments_dimensions_names.append(tuple(argument_dimension_names))
            arguments_dimensions_dimensions.append(tuple(argument_dimension_dimensions))

        else:
            arguments_dimensions_names.append(tuple())
            arguments_dimensions_dimensions.append(tuple())

    arguments_dimensions_names: tuple[XarrayDimension] = tuple(
        arguments_dimensions_names
    )
    arguments_dimensions_dimensions: tuple[DimensionAnnotation] = tuple(
        arguments_dimensions_dimensions
    )

    return NamesAndDimensionalities(
        arguments_dimensions_names, arguments_dimensions_dimensions
    )


@dataclass
class Dimensionalize:
    argument_dimensions: ArgumentDimensionType
    result_dimensions: ArgumentDimensionType

    _argument_dimension_names: XarrayDimension = field(init=False)
    _argument_dimensionality: DimensionAnnotation = field(init=False)

    _result_dimension_names: XarrayDimension = field(init=False)
    _result_dimensionality: DimensionAnnotation = field(init=False)

    def __post_init__(self):
        self._argument_dimension_names, self._argument_dimensionality = (
            separate_argument_dimension_into_names_and_dimensions(
                self.argument_dimensions
            )
        )

        self._result_dimension_names, self._result_dimensionality = (
            separate_argument_dimension_into_names_and_dimensions(
                self.result_dimensions
            )
        )

        self.vectorizable: bool = True

    def __call__(self, function: Callable[[Any], Any]):
        @wraps(function)
        def apply_ufunc_wrapper(*args, **kwargs):
            return xr.apply_ufunc(
                function,
                *args,
                kwargs=kwargs,
                input_core_dims=self._argument_dimension_names,
                output_core_dims=self._result_dimension_names,
                vectorize=self.vectorizable,
            )

        return apply_ufunc_wrapper


# TODO: can we use this to define our structural types? xarray datasets would be one
#       possible implementation of the protocol. The types would describe the physical
#       nature of the variables (e.g. do they represent pressure, temperature, wavelength, etc.),
#       and potentially the shapes of the inputs vs. outputs. Not specific shapes, just for example
#       whether the function preserves the shape of the input.

# Below is some borrowing of xarray-specific functional code from a different project.


def get_getter_from_dataset(variable_name: str) -> Callable[[xr.Dataset], xr.DataArray]:
    return lambda dataset: dataset[variable_name]


def apply_to_variable(
    function: Callable[[xr.DataArray], xr.DataArray],
    variable_name: str,
    dataset: xr.Dataset,
    result_variable_name: str = None,
) -> xr.DataArray:
    result = function(get_getter_from_dataset(variable_name))(dataset=dataset)

    return (
        result if result_variable_name is None else result.rename(result_variable_name)
    )


def map_function_of_function_arguments_to_dataset_variables(
    second_order_dataset_function, *dataset_functions
):
    """
    xr.Dataset functions are functions that take dataset variables as arguments.
    Sometimes they also take arguments that are functions of dataset variables themselves.
    That's what I mean by "second order".

    For dataset variables d1, d2, d3, ...
    and dataset functions f1(d1, d2, ...), f2(d1, d2, ...), ...,
    second_order_dataset_function = function(f1(...), f2(...), ..., d1, d2, ...)

    This will produce a function that is equivalent to the original second-order function
    but only needs the dataset as a single argument.
    """

    def apply_all_functions(*dataset_functions, second_order_dataset_function, dataset):
        purely_second_order_dataset_function = (
            partial(second_order_dataset_function, dataset=dataset)
            if "dataset" in signature(second_order_dataset_function).parameters.keys()
            else second_order_dataset_function
        )

        evaluated_dataset_functions = [
            dataset_function(dataset=dataset) for dataset_function in dataset_functions
        ]

        return purely_second_order_dataset_function(*evaluated_dataset_functions)

    return partial(
        apply_all_functions,
        *dataset_functions,
        second_order_dataset_function=second_order_dataset_function,
    )


def map_function_arguments_to_dataset_variables(
    function, variable_mapping
) -> Callable[[xr.Dataset], xr.DataArray]:
    def function_using_dataset(*non_dataset_args, dataset, function, variable_mapping):
        dataset_kwargs = {
            kwarg: dataset.get(label) for kwarg, label in variable_mapping.items()
        }
        return function(*non_dataset_args, **dataset_kwargs)

    return partial(
        function_using_dataset, function=function, variable_mapping=variable_mapping
    )


# def rename_and_unitize(data_array: xr.DataArray, name: str, units: str) -> xr.DataArray:
#    return data_array.rename(name).assign_attrs(units=units)


@function_decorator
def rename_and_unitize(new_name: str, units: str, function=DECORATED) -> Callable:
    @wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        if isinstance(result, xr.DataArray):
            result = result.rename(new_name).assign_attrs(units=units)
        elif isinstance(result, xr.Dataset):
            result = result.rename({var: new_name for var in result.data_vars})
            for var in result.data_vars:
                result[var].attrs["units"] = units
        return result

    return wrapper


XarrayOutputs: TypeAlias = Mapping[str, xr.DataArray | xr.Dataset]


class ProducesXarrayOutputs(Protocol):
    def __call__(self, *args, **kwargs) -> XarrayOutputs: ...


def save_xarray_outputs_to_file(
    function: ProducesXarrayOutputs,
) -> ProducesXarrayOutputs:
    @wraps(function)
    def wrapper(
        *args,
        output_directory: Path = Path.cwd(),
        filename_prefix: str = "output",
        **kwargs,
    ):
        def make_output_filepath(case_name: str) -> str:
            return str(output_directory / f"{filename_prefix}_{case_name}.nc")

        result: XarrayOutputs = function(*args, **kwargs)

        if isinstance(result, xr.Dataset):
            result_as_dict = {"output": result}
        elif isinstance(result, dict):
            result_as_dict = result
        elif isinstance(result, tuple):
            result_as_dict = result._asdict()

        for dataset_name, dataset in result_as_dict.items():
            output_dataset: xr.Dataset = (
                dataset
                if isinstance(dataset, xr.Dataset)
                else dataset.to_dataset(name=dataset_name)
            )
            output_dataset.to_netcdf(make_output_filepath(case_name=dataset_name))

        return result

    return wrapper
