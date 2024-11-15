from collections.abc import Callable, Sequence
from functools import reduce

import numpy as np
from numpy.typing import NDArray


def compose(*functions: Sequence[Callable]) -> Callable:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def interleave(
    first_terms: NDArray[np.float64],
    second_terms: NDArray[np.float64],
    interleaved_axis: int = -1,
):
    interleaved_dimension_size = (
        np.shape(first_terms)[interleaved_axis]
        + np.shape(second_terms)[interleaved_axis]
    )
    interleaved_array_shape = np.asarray(np.shape(first_terms))
    interleaved_array_shape[interleaved_axis] = interleaved_dimension_size
    interleaved_array = np.empty(interleaved_array_shape, dtype=first_terms.dtype)

    base_slice_list = [slice(None)] * first_terms.ndim

    first_slice = slice(0, None, 2)
    first_slices = base_slice_list
    first_slices[interleaved_axis] = first_slice

    second_slice = slice(1, None, 2)
    second_slices = base_slice_list
    second_slices[interleaved_axis] = second_slice

    interleaved_array[*first_slices] = first_terms
    interleaved_array[*second_slices] = second_terms

    return interleaved_array
