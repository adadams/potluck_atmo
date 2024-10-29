from typing import Any, Optional, Type

import msgspec
import xarray as xr

type XarrayDimension = tuple[str, ...]
type XarrayData = list[float]


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


def convert_xarray_to_msgspec(dataset: xr.Dataset) -> dict:
    return dataset.to_dict()


def convert_msgspec_to_xarray(serialized_dataset: dict) -> xr.Dataset:
    return xr.Dataset.from_dict(serialized_dataset)
