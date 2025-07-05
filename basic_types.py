from typing import Annotated, Literal, Tuple, TypeAlias, TypeVar

import msgspec
import numpy as np

Shape = TypeVar("Shape", bound=Tuple[int, ...])
DType = TypeVar("DType", bound=np.generic)

NormalizedValue = Annotated[float, msgspec.Meta(ge=0, le=1)]

AltitudeValue: TypeAlias = Annotated[float, msgspec.Meta(ge=0)]
TemperatureValue: TypeAlias = Annotated[float, msgspec.Meta(gt=0)]
PressureValue: TypeAlias = Annotated[float, msgspec.Meta(ge=0)]
LogPressureValue: TypeAlias = float
MixingRatioValue: TypeAlias = Annotated[float, msgspec.Meta(ge=0, le=1)]
LogMixingRatioValue: TypeAlias = Annotated[float, msgspec.Meta(le=0)]

BaseUnits: TypeAlias = Literal["[mass]", "[time]", "[length]", "[temperature]"]
BaseUnitType: TypeAlias = Annotated[str, BaseUnits]
UnitType: TypeAlias = tuple[tuple[BaseUnitType, int]]

# Organize units here:
PressureUnits: UnitType = (("[mass]", 1), ("[time]", -2), ("[length]", -1))
WavelengthUnits: UnitType = (("[length]", 1),)
DimensionlessUnits: UnitType = (("", 1),)

DimensionAnnotation: TypeAlias = Tuple[str, UnitType]

# Organize dimensions here:
WavelengthDimension: DimensionAnnotation = ("wavelength", WavelengthUnits)
PressureDimension: DimensionAnnotation = ("pressure", PressureUnits)
SpeciesDimension: DimensionAnnotation = ("species", DimensionlessUnits)
CosineAngleDimension: DimensionAnnotation = ("cosine_angle", DimensionlessUnits)
