from dataclasses import dataclass, field
from typing import Final

import numpy as np
import xarray as xr

from potluck.basic_types import (
    CosineAngleDimension,
    PressureDimension,
    WavelengthDimension,
)
from potluck.xarray_functional_wrappers import Dimensionalize, set_result_name_and_units

STREAM_COSINE_ANGLES: Final[np.ndarray[np.float64]] = np.array(
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

STREAM_SINE_ANGLES: Final[np.ndarray[np.float64]] = np.sqrt(1 - STREAM_COSINE_ANGLES**2)

STREAM_WEIGHTS: Final[np.ndarray[np.float64]] = np.array(
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

stream_cosine_angles_as_dataarray: Final[xr.DataArray] = xr.DataArray(
    data=STREAM_COSINE_ANGLES,
    dims=("cosine_angle",),
    coords={"cosine_angle": STREAM_COSINE_ANGLES},
    name="stream_cosine_angles",
)

stream_sine_angles_as_dataarray: Final[xr.DataArray] = xr.DataArray(
    data=STREAM_SINE_ANGLES,
    dims=("cosine_angle",),
    coords={"cosine_angle": STREAM_COSINE_ANGLES},
    name="stream_sine_angles",
)

stream_weights_as_dataarray: Final[xr.DataArray] = xr.DataArray(
    data=STREAM_WEIGHTS,
    dims=("cosine_angle",),
    coords={"cosine_angle": STREAM_COSINE_ANGLES},
    name="stream_weights",
)


@dataclass
class OneStreamRTInputs:
    thermal_intensity: xr.DataArray  # (wavelength, pressure)
    cumulative_optical_depth_by_layer: xr.DataArray  # (wavelength, pressure)
    stream_cosine_angles: xr.DataArray = field(
        default_factory=lambda: stream_cosine_angles_as_dataarray.copy()
    )
    stream_sine_angles: xr.DataArray = field(
        default_factory=lambda: stream_sine_angles_as_dataarray.copy()
    )
    stream_weights: xr.DataArray = field(
        default_factory=lambda: stream_weights_as_dataarray.copy()
    )


@set_result_name_and_units(new_name="emitted_onestream_flux", units="erg s^-1 cm^-3")
@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
        (CosineAngleDimension,),
        (CosineAngleDimension,),
        (CosineAngleDimension,),
    ),
    result_dimensions=((WavelengthDimension,),),
)
def calculate_spectral_intensity_at_surface(
    thermal_intensity: np.ndarray[np.float64],
    cumulative_optical_depth_by_layer: np.ndarray[np.float64],
    stream_cosine_angles: np.ndarray[np.float64],
    stream_sine_angles: np.ndarray[np.float64],
    stream_weights: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    stream_cosine_angles: np.ndarray[np.float64] = np.expand_dims(
        stream_cosine_angles,
        axis=tuple(range(1, cumulative_optical_depth_by_layer.ndim + 1)),
    )

    stream_sine_angles: np.ndarray[np.float64] = np.expand_dims(
        stream_sine_angles,
        axis=tuple(range(1, cumulative_optical_depth_by_layer.ndim + 1)),
    )

    attenutation_factors_by_layer: np.ndarray[np.float64] = np.exp(
        -cumulative_optical_depth_by_layer / stream_cosine_angles
    )

    previous_attenuation_factors_by_layer: np.ndarray[np.float64] = np.concatenate(
        (
            np.ones([*attenutation_factors_by_layer.shape[:-1], 1]),  # exp(0) = 1
            attenutation_factors_by_layer[..., :-1],
        ),
        axis=-1,
    )

    spectral_intensity_by_layer: np.ndarray[np.float64] = thermal_intensity * (
        previous_attenuation_factors_by_layer - attenutation_factors_by_layer
    )

    spectral_intensity_at_surface_by_angle: np.ndarray[np.float64] = np.sum(
        spectral_intensity_by_layer * stream_sine_angles, axis=-1
    )

    stream_weights: np.ndarray[np.float64] = np.expand_dims(
        stream_weights,
        axis=tuple(range(1, spectral_intensity_at_surface_by_angle.ndim)),
    )

    spectral_intensity_at_surface: np.ndarray[np.float64] = np.pi**2 * np.sum(
        spectral_intensity_at_surface_by_angle * stream_weights, axis=0
    )

    return spectral_intensity_at_surface
