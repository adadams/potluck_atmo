from typing import Final

import numpy as np
from numpy.typing import NDArray

from xarray_functional_wrappers import Dimensionalize
from xarray_serialization import PressureType, WavelengthType

STREAM_COSINE_ANGLES: Final[NDArray[np.float64]] = np.array(
    [
        0.0446339553,
        0.1443662570,
        0.2868247571,
        0.4548133152,
        0.6280678354,
        0.7856915206,
        0.9086763921,
        0.9822200849,
    ]
)

STREAM_SINE_ANGLES: Final[NDArray[np.float64]] = np.sqrt(1 - STREAM_COSINE_ANGLES**2)

STREAM_WEIGHTS: Final[NDArray[np.float64]] = np.array(
    [
        0.0032951914,
        0.0178429027,
        0.0454393195,
        0.0791995995,
        0.1060473494,
        0.1125057995,
        0.0911190236,
        0.0445508044,
    ]
)


@Dimensionalize(
    argument_dimensions=(
        (WavelengthType, PressureType),
        (WavelengthType, PressureType),
    ),
    result_dimensions=((WavelengthType,),),
)
def calculate_spectral_intensity_at_surface(
    thermal_intensity: NDArray[np.float64],
    cumulative_optical_depth_by_layer: NDArray[np.float64],
    stream_cosine_angles: NDArray[np.float64] = STREAM_COSINE_ANGLES,
    stream_sine_angles: NDArray[np.float64] = STREAM_SINE_ANGLES,
    stream_weights: NDArray[np.float64] = STREAM_WEIGHTS,
) -> NDArray[np.float64]:
    stream_cosine_angles = np.expand_dims(
        stream_cosine_angles,
        axis=tuple(range(1, cumulative_optical_depth_by_layer.ndim + 1)),
    )

    stream_sine_angles = np.expand_dims(
        stream_sine_angles,
        axis=tuple(range(1, cumulative_optical_depth_by_layer.ndim + 1)),
    )

    attenutation_factors_by_layer: NDArray[np.float64] = np.exp(
        -cumulative_optical_depth_by_layer / stream_cosine_angles
    )
    previous_attenuation_factors_by_layer: NDArray[np.float64] = np.concatenate(
        (
            np.zeros([*attenutation_factors_by_layer.shape[:-1], 1]),
            attenutation_factors_by_layer[..., :-1],
        ),
        axis=-1,
    )

    spectral_intensity_by_layer: NDArray[np.float64] = thermal_intensity * (
        previous_attenuation_factors_by_layer - attenutation_factors_by_layer
    )

    spectral_intensity_at_surface_by_angle: NDArray[np.float64] = np.sum(
        spectral_intensity_by_layer * stream_sine_angles, axis=-1
    )

    stream_weights = np.expand_dims(
        stream_weights,
        axis=tuple(range(1, spectral_intensity_at_surface_by_angle.ndim)),
    )

    spectral_intensity_at_surface = np.pi**2 * np.sum(
        spectral_intensity_at_surface_by_angle * stream_weights, axis=0
    )
    return spectral_intensity_at_surface
