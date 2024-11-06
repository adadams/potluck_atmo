from typing import (
    Annotated,
    Literal,
    Optional,
    Required,
    Type,
    TypeAlias,
    TypedDict,
)

import msgspec
import xarray as xr

# from nptyping import DataFrame, Float, NDArray, Shape, Structure

type XarrayDimension = tuple[str, ...]
type XarrayData = list[float]

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
# PressureData: TypeAlias = NDArray[Shape["number_of_pressures"], Float]  # noqa: F821

WavelengthUnits: UnitType = (("[length]", 1),)
# WavelengthData: TypeAlias = NDArray[Shape["number_of_wavelengths"], Float]  # noqa: F821


class QuantityAttrs(TypedDict):
    units: Required[str]
    unit_base_rep: UnitType


class UnitsAttrs(msgspec.Struct):
    units: str


class XarrayVariable(msgspec.Struct):
    dims: XarrayDimension
    attrs: UnitsAttrs
    data: XarrayData


class XarrayDataArray(msgspec.Struct):
    dims: XarrayDimension
    attrs: UnitsAttrs
    data: XarrayData
    coords: dict[str, XarrayVariable]
    name: str


class XarrayDataset(msgspec.Struct):
    dims: XarrayDimension
    data_vars: dict[str, XarrayDataArray]
    coords: dict[str, XarrayVariable]
    attrs: Optional[dict[str, str | float]]


'''
def enc_hook(obj: Any) -> Any:
    """Given an object that msgspec doesn't know how to serialize by
    default, convert it into an object that it does know how to
    serialize"""
    pass


def dec_hook(type: Type, obj: Any) -> Any:
    """Given a type in a schema, convert ``obj`` (composed of natively
    supported objects) into an object of type ``type``.

    Any `TypeError` or `ValueError` exceptions raised by this method will
    be considered "user facing" and converted into a `ValidationError` with
    additional context. All other exceptions will be raised directly.
    """
    pass
'''

"""
class PressureCoordinate(XarrayVariable):
    # data: PressureData
    data: NDArray[np.float64]
    dims: tuple[str, ...] = ("pressure",)
    attrs: dict[str, Any]  # = {"units": PressureUnits}
"""


def convert_xarray_to_msgspec(dataset: xr.Dataset) -> dict:
    return dataset.to_dict()


def convert_msgspec_to_xarray(serialized_dataset: dict) -> xr.Dataset:
    return xr.Dataset.from_dict(serialized_dataset)
