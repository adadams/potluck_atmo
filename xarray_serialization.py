from typing import Dict, List, Optional, Tuple, TypeAlias, Union

import msgspec
import xarray as xr

from basic_types import UnitType

XarrayDimension: TypeAlias = Tuple[str, ...]
XarrayData: TypeAlias = List[float]

SerializablePrimitiveType = Union[str, int, float, bool, None]
SerializableType = Union[
    SerializablePrimitiveType,
    Dict[str, SerializablePrimitiveType],
    List[SerializablePrimitiveType],
]

# SerializableContainerType = Union[
#    SerializablePrimitiveType,
#    Dict[str, SerializablePrimitiveType],
#    List[SerializablePrimitiveType],
# ]

# SerializableType = Union[
#    SerializableContainerType,
#    Dict[str, SerializableContainerType],
#    List[SerializableContainerType],
# ]

AttrsType: TypeAlias = Dict[str, SerializableType]


class QuantityAttrs(msgspec.Struct):
    units: str
    unit_base_rep: UnitType


class UnitsAttrs(msgspec.Struct):
    units: str


class XarrayVariable(msgspec.Struct):
    dims: XarrayDimension
    attrs: UnitsAttrs
    data: XarrayData


XarrayCoordinate: TypeAlias = Dict[str, XarrayVariable]


class XarrayDataArray(msgspec.Struct):
    dims: XarrayDimension
    attrs: UnitsAttrs
    data: XarrayData
    coords: XarrayCoordinate
    name: str


class XarrayDataset(msgspec.Struct):
    dims: XarrayDimension
    data_vars: Dict[str, XarrayDataArray]
    coords: XarrayCoordinate
    attrs: Optional[AttrsType]


def convert_xarray_to_msgspec(dataset: xr.Dataset) -> dict:
    return dataset.to_dict()


def convert_msgspec_to_xarray(serialized_dataset: dict) -> xr.Dataset:
    return xr.Dataset.from_dict(serialized_dataset)
