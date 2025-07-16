from typing import Protocol, TypeAlias

import msgspec
import numpy as np

from basic_types import Shape, TemperatureValue


class TemperatureModel(Protocol):
    def __call__(
        self, profile_log_pressures: np.ndarray[Shape, np.float64]
    ) -> np.ndarray[Shape, TemperatureValue]: ...


TemperatureModelInputs: TypeAlias = msgspec.Struct
TemperatureModelParameters: TypeAlias = msgspec.Struct


class TemperatureModelConstructor(Protocol):
    def __call__(
        self,
        temperature_model_inputs: TemperatureModelInputs,
        temperature_model_parameters: TemperatureModelParameters,
    ) -> TemperatureModel: ...
