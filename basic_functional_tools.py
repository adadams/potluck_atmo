from collections.abc import Callable, Sequence
from functools import reduce

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray


def compose(*functions: Sequence[Callable]) -> Callable:
    return reduce(lambda f, g: lambda *x: g(f(*x)), functions)


def interleave(
    first_terms: NDArray[np.float64],
    second_terms: NDArray[np.float64],
    interleaved_axis: int = -1,
) -> np.ndarray:
    interleaved_dimension_size = (
        np.shape(first_terms)[interleaved_axis]
        + np.shape(second_terms)[interleaved_axis]
    )
    interleaved_array_shape = np.asarray(np.shape(first_terms))
    interleaved_array_shape[interleaved_axis] = interleaved_dimension_size
    interleaved_array = np.empty(interleaved_array_shape, dtype=first_terms.dtype)

    base_slice_list = [slice(None)] * first_terms.ndim

    first_slice = slice(0, None, 2)
    first_slices = base_slice_list.copy()
    first_slices[interleaved_axis] = first_slice

    second_slice = slice(1, None, 2)
    second_slices = base_slice_list.copy()
    second_slices[interleaved_axis] = second_slice

    interleaved_array[*first_slices] = first_terms
    interleaved_array[*second_slices] = second_terms

    return interleaved_array


def interleave_with_jax(
    first_terms: Array, second_terms: Array, interleaved_axis: int = -1
) -> Array:
    """
    Interleaves two JAX arrays along a specified axis.
    This is the JAX-compatible version of the original interleave.
    """
    interleaved_dimension_size = (
        jnp.shape(first_terms)[interleaved_axis]
        + jnp.shape(second_terms)[interleaved_axis]
    )

    original_shape_tuple: tuple[int] = jnp.shape(first_terms)

    interleaved_array_shape_list: list[int] = list(original_shape_tuple)
    interleaved_array_shape_list[interleaved_axis] = int(interleaved_dimension_size)

    interleaved_array_shape: tuple[int] = tuple(interleaved_array_shape_list)
    interleaved_array = jnp.empty(interleaved_array_shape, dtype=first_terms.dtype)

    base_slice_list = [slice(None)] * first_terms.ndim

    first_slice_indices = base_slice_list.copy()
    first_slice_indices[interleaved_axis] = slice(0, None, 2)  # even positions

    second_slice_indices = base_slice_list.copy()
    second_slice_indices[interleaved_axis] = slice(1, None, 2)  # odd positions

    interleaved_array = interleaved_array.at[tuple(first_slice_indices)].set(
        first_terms
    )
    interleaved_array = interleaved_array.at[tuple(second_slice_indices)].set(
        second_terms
    )

    return interleaved_array
