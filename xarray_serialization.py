from typing import Annotated, Literal, Optional, Type, TypeAlias

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

DimensionlessUnits: UnitType = (("", 1),)

DimensionAnnotation: TypeAlias = tuple[str, UnitType]

WavelengthType: DimensionAnnotation = ("wavelength", WavelengthUnits)
PressureType: DimensionAnnotation = ("pressure", PressureUnits)
SpeciesType: DimensionAnnotation = ("species", DimensionlessUnits)

ArgumentDimensionType: TypeAlias = tuple[DimensionAnnotation]
FunctionDimensionType: TypeAlias = tuple[ArgumentDimensionType]


class QuantityAttrs(msgspec.Struct):
    units: str
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
