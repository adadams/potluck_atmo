from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    Required,
    Type,
    TypeAlias,
    TypedDict,
)

import msgspec
from nptyping import DataFrame, Float, NDArray, Shape, Structure

"""
Ideas to jot down here for easy reference:
xarray dataset print-outs (reprs) already show
shape and structure information under "dimensions"
that look a lot like they could be annotations in
something like nptyping.
"""


BaseUnits: Type = Literal["[mass]", "[time]", "[length]", "[temperature]"]
BaseUnitType: TypeAlias = Annotated[str, BaseUnits]
UnitType: TypeAlias = tuple[tuple[BaseUnitType, int]]

PressureUnits: UnitType = (("[mass]", 1), ("[time]", -2), ("[length]", -1))

PressureData: TypeAlias = NDArray[Shape["number_of_pressures"], Float]  # noqa: F821


class QuantityAttrs(TypedDict):
    units: Required[str]
    unit_base_rep: UnitType


class PressureCoordinate(msgspec.Struct):
    data: PressureData
    name: str
    dims: tuple[str, ...] = ("pressure",)
    attrs: dict[str, Any]  # = {"units": PressureUnits}


class OperatesonVerticalStructure(Protocol):
    def __call__() -> None: ...
