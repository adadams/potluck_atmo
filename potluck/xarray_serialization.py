from typing import Optional, TypeAlias

import msgspec
import xarray as xr

from potluck.basic_types import UnitType

XarrayDimension: TypeAlias = tuple[str, ...]
XarrayData: TypeAlias = list[float]

SerializablePrimitiveType = str | int | float | bool | None
SerializableType = (
    SerializablePrimitiveType
    | dict[str, SerializablePrimitiveType]
    | list[SerializablePrimitiveType]
)

AttrsType: TypeAlias = dict[str, SerializableType]


class QuantityAttrs(msgspec.Struct):
    units: str
    unit_base_rep: UnitType


class UnitsAttrs(msgspec.Struct):
    units: str


class XarrayVariable(msgspec.Struct):
    dims: XarrayDimension
    attrs: UnitsAttrs
    data: XarrayData


XarrayCoordinate: TypeAlias = dict[str, XarrayVariable]


class XarrayDataArray(msgspec.Struct):
    dims: XarrayDimension
    attrs: UnitsAttrs
    data: XarrayData
    coords: XarrayCoordinate
    name: str


class XarrayDataset(msgspec.Struct):
    dims: XarrayDimension
    data_vars: dict[str, XarrayDataArray]
    coords: XarrayCoordinate
    attrs: Optional[AttrsType]


def convert_xarray_to_msgspec(dataset: xr.Dataset) -> dict:
    return dataset.to_dict()


def convert_msgspec_to_xarray(serialized_dataset: dict) -> xr.Dataset:
    return xr.Dataset.from_dict(serialized_dataset)
