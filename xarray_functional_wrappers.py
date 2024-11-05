from collections.abc import Callable, Sequence
from functools import partial, reduce
from inspect import signature
from typing import Protocol

import xarray as xr

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


# TODO: given a dataset that contains named variables, that can be used as arguments to
#       a function, map the variable names to the argument names (if different), then
#       return a function that takes the dataset --------as a single argument.
def map_to_xarray(
    function: Callable[[xr.DataArray], xr.DataArray],
) -> OperatesOnDataset: ...


# TODO: can we use this to define our structural types? xarray datasets would be one
#       possible implementation of the protocol. The types would describe the physical
#       nature of the variables (e.g. do they represent pressure, temperature, wavelength, etc.),
#       and potentially the shapes of the inputs vs. outputs. Not specific shapes, just for example
#       whether the function preserves the shape of the input.
