from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial, reduce, wraps
from inspect import signature
from pathlib import Path
from typing import Any, Final, NamedTuple, Optional, Protocol, TypeAlias

import numpy as np
import pint_xarray
import xarray as xr
from decopatch import DECORATED, function_decorator
from pint import UnitRegistry

from potluck.basic_types import DimensionAnnotation
from potluck.xarray_serialization import XarrayDimension

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory

DEFAULT_UNITS_SYSTEM: Final[str] = "cgs"
ureg: UnitRegistry = UnitRegistry(system=DEFAULT_UNITS_SYSTEM)
ureg.load_definitions(str(project_directory / "additional_units.txt"))
ureg = pint_xarray.setup_registry(ureg)

XarrayStructure: TypeAlias = xr.Dataset | xr.DataArray

ArgumentDimensionType: TypeAlias = tuple[DimensionAnnotation]
FunctionDimensionType: TypeAlias = tuple[ArgumentDimensionType]


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
                keep_attrs=True,
            )

        return apply_ufunc_wrapper


def call_meshgrid_on_xarray(
    xarray_structure_x: xr.DataArray,
    xarray_structure_y: xr.DataArray,
    **meshgrid_kwargs,
):
    # assumes xarray_structure_x and xarray_structure_y each have one dimension
    # that are distinct
    if not isinstance(xarray_structure_x, xr.DataArray) or not isinstance(
        xarray_structure_y, xr.DataArray
    ):
        raise TypeError("Both arguments must be of type xr.DataArray.")

    return xr.apply_ufunc(
        np.meshgrid,
        xarray_structure_x,
        xarray_structure_y,
        input_core_dims=(xarray_structure_x.dims, xarray_structure_y.dims),
        output_core_dims=(
            [*xarray_structure_y.dims, *xarray_structure_x.dims],
            [*xarray_structure_y.dims, *xarray_structure_x.dims],
        ),
        keep_attrs=True,
        kwargs=meshgrid_kwargs,
    )


# TODO: can we use this to define our structural types? xarray datasets would be one
#       possible implementation of the protocol. The types would describe the physical
#       nature of the variables (e.g. do they represent pressure, temperature, wavelength, etc.),
#       and potentially the shapes of the inputs vs. outputs. Not specific shapes, just for example
#       whether the function preserves the shape of the input.


@function_decorator
def set_result_name_and_units(
    result_names: str | Iterable[str] | Mapping[str, str],
    units: str | Iterable[str] | Mapping[str, str],
    function=DECORATED,
) -> Callable:
    @wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)

        if isinstance(result, xr.DataArray):
            result_name = result_names

            result = result.rename(result_name).assign_attrs(units=units)

        elif isinstance(result, tuple):
            if any(not isinstance(dataarray, xr.DataArray) for dataarray in result):
                raise TypeError(
                    "All elements must be of type xarray DataArray. ",
                    "This may be because the wrapped function is not returning a tuple of dataarrays.",
                )
            # as in the result of a function that returns a tuple of dataarrays

            # check that the length of new_name and units are the same as the length of the tuple
            if len(result_names) != len(units) != len(result):
                raise ValueError(
                    "The number of variable names and units for the result of the function must be the same."
                )

            result = tuple(
                dataarray.rename(result_name).assign_attrs(units=result_unit)
                for dataarray, result_name, result_unit in zip(
                    result, result_names, units
                )
            )

        return result

    return wrapper


def convert_units(
    xarray_structure: XarrayStructure | xr.DataTree,
    new_variable_units: Mapping[str, str],
) -> XarrayStructure:
    def map_over_dataarray(
        xarray_structure: xr.DataArray, new_variable_units: Mapping[str, str]
    ) -> xr.DataArray:
        for variable_name, units in new_variable_units.items():
            if variable_name == xarray_structure.name:
                xarray_structure: xr.DataArray = (
                    xarray_structure.pint.quantify(unit_registry=ureg)
                    .pint.to(units)
                    .pint.dequantify()
                )

            elif variable_name in xarray_structure.coords:
                xarray_structure: xr.DataArray = xarray_structure.assign_coords(
                    {
                        variable_name: xarray_structure[variable_name]
                        .pint.quantify(unit_registry=ureg)
                        .pint.to(units)
                        .pint.dequantify()
                        for variable_name, units in new_variable_units.items()
                    }
                )

            else:
                raise ValueError(
                    "Variable name(s) either need to be the name of the dataarray, "
                    "or need to be present in the coordinates of the dataarray."
                )

        return xarray_structure

    if isinstance(xarray_structure, xr.DataArray):
        return map_over_dataarray(xarray_structure, new_variable_units)

    def map_over_dataset(
        xarray_structure: xr.Dataset, new_variable_units: Mapping[str, str]
    ):
        all_variables_present: bool = all(
            variable_name in xarray_structure.data_vars
            or variable_name in xarray_structure.coords
            for variable_name in new_variable_units
        )

        if not all_variables_present:
            raise ValueError(
                "All variables in new_variable_units must be present in xarray_structure. "
                f"Missing variables: {set(new_variable_units.keys()) - set(xarray_structure.data_vars.keys())}."
            )

        else:
            xarray_structure: xr.Dataset = xarray_structure.assign(
                {
                    variable_name: xarray_structure[variable_name]
                    .pint.quantify()
                    .pint.to(units)
                    .pint.dequantify()
                    for variable_name, units in new_variable_units.items()
                }
            )

        return xarray_structure

    if isinstance(xarray_structure, xr.Dataset):
        return map_over_dataset(xarray_structure, new_variable_units)

    elif isinstance(xarray_structure, xr.DataTree):
        return xarray_structure.map_over_datasets(
            partial(map_over_dataset, new_variable_units=new_variable_units)
        )


# Generator of an empty tuple, purely for clarity of purpose.
def set_dimensionless_quantity() -> tuple:
    return tuple()


XarrayOutputs: TypeAlias = Mapping[str, XarrayStructure]


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


# Below is some borrowing of xarray-specific functional code from a different project:


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
