from typing import (  # Added Mapping
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypeAlias,
)

import msgspec
import numpy as np
import xarray as xr

# --- Type Aliases for Units (consistent with previous versions) ---
BaseUnits: TypeAlias = Literal[
    "[mass]",
    "[time]",
    "[length]",
    "[temperature]",
    "[current]",
    "[luminosity]",
    "[substance]",
    "[]",
]
BaseUnitType: TypeAlias = Annotated[str, BaseUnits]
UnitType: TypeAlias = Tuple[Tuple[BaseUnitType, int], ...]

# --- msgspec.Struct Definitions for the Schema ("Spec Sheet") ---


class AttrsSpec(
    msgspec.Struct, gc=False, array_like=True
):  # array_like for TOML array of tables if needed
    """Defines expected attributes for a variable or coordinate."""

    units: str
    unit_base_rep: UnitType
    long_name: Optional[str] = None  # If present in TOML, this exact value is expected
    standard_name: Optional[str] = (
        None  # If present in TOML, this exact value is expected
    )
    # Add other specific attributes you want to enforce
    # Example: check_existence_keys: Optional[List[str]] = None


class ShapeSpec(msgspec.Struct, gc=False, array_like=True):
    """Defines expected shape characteristics."""

    ndim: int  # Expected number of dimensions
    # Potentially: dim_order: Optional[List[str]] = None
    # Potentially: fixed_sizes: Optional[Dict[str, int]] = None # e.g., {"lat": 73, "lon": 144}


class VariableSpec(msgspec.Struct, gc=False, array_like=True):
    """Defines the specification for a single data variable or coordinate."""

    dims: List[str]  # Expected dimension names
    dtype: str  # Expected data type as a string (e.g., "float32", "datetime64[ns]")
    attrs_spec: AttrsSpec
    shape_spec: Optional[ShapeSpec] = None


class DatasetSchema(msgspec.Struct, gc=False):
    """Defines the overall schema for an xarray.Dataset."""

    schema_description: str
    dataset_attrs_spec: Optional[Dict[str, Any]] = (
        None  # Key-value pairs of expected dataset attributes
    )
    coords_spec: Optional[Dict[str, VariableSpec]] = None
    data_vars_spec: Dict[str, VariableSpec]


# --- Validation Logic ---


def validate_variable_against_spec(
    var_name: str,
    var_data: xr.DataArray,  # Can also be xr.Variable for coordinates not promoted
    spec: VariableSpec,
    is_coordinate: bool = False,
) -> List[str]:
    """Validates a single xarray.DataArray or xr.Variable against its spec."""
    errors: List[str] = []
    prefix = f"{'Coordinate' if is_coordinate else 'Data variable'} '{var_name}'"

    # 1. Check dimensions
    if set(var_data.dims) != set(spec.dims):
        errors.append(
            f"{prefix} dimension names mismatch. Expected: {sorted(spec.dims)}, Got: {sorted(list(var_data.dims))}."
        )
    # Optional: Check dimension order if spec.shape_spec.dim_order is defined
    elif (
        spec.shape_spec
        and spec.shape_spec.dim_order
        and list(var_data.dims) != spec.shape_spec.dim_order
    ):
        errors.append(
            f"{prefix} dimension order mismatch. Expected: {spec.shape_spec.dim_order}, Got: {list(var_data.dims)}."
        )

    # 2. Check dtype
    actual_dtype = str(var_data.dtype)
    if actual_dtype != spec.dtype:
        errors.append(
            f"{prefix} dtype mismatch. Expected: '{spec.dtype}', Got: '{actual_dtype}'."
        )

    # 3. Check shape (ndim primarily, more can be added)
    if spec.shape_spec:
        if var_data.ndim != spec.shape_spec.ndim:
            errors.append(
                f"{prefix} number of dimensions (ndim) mismatch. Expected: {spec.shape_spec.ndim}, Got: {var_data.ndim}."
            )
        # Potentially check spec.shape_spec.fixed_sizes against var_data.shape here

    # 4. Check attributes
    actual_attrs: Mapping[Any, Any] = var_data.attrs

    # Units
    if spec.attrs_spec.units != actual_attrs.get("units"):
        errors.append(
            f"{prefix} attribute 'units' mismatch. Expected: '{spec.attrs_spec.units}', Got: '{actual_attrs.get('units')}'."
        )

    # unit_base_rep (ensure tuple comparison for order and content)
    # Convert actual to tuple of tuples if it's list of lists from xarray attrs
    actual_ubr_raw = actual_attrs.get("unit_base_rep")
    actual_ubr: Optional[UnitType] = None
    if isinstance(actual_ubr_raw, (list, tuple)):
        try:
            actual_ubr = tuple(tuple(item) for item in actual_ubr_raw)  # type: ignore
        except TypeError:
            pass  # actual_ubr remains None or wrong type, error will be caught below

    if spec.attrs_spec.unit_base_rep != actual_ubr:
        errors.append(
            f"{prefix} attribute 'unit_base_rep' mismatch or malformed. Expected: {spec.attrs_spec.unit_base_rep}, Got: {actual_ubr_raw}."
        )

    # Long Name (if specified in spec)
    if (
        spec.attrs_spec.long_name is not None
        and spec.attrs_spec.long_name != actual_attrs.get("long_name")
    ):
        errors.append(
            f"{prefix} attribute 'long_name' mismatch. Expected: '{spec.attrs_spec.long_name}', Got: '{actual_attrs.get('long_name')}'."
        )
    elif spec.attrs_spec.long_name is not None and "long_name" not in actual_attrs:
        errors.append(f"{prefix} missing expected attribute 'long_name'.")

    # Standard Name (if specified in spec)
    if (
        spec.attrs_spec.standard_name is not None
        and spec.attrs_spec.standard_name != actual_attrs.get("standard_name")
    ):
        errors.append(
            f"{prefix} attribute 'standard_name' mismatch. Expected: '{spec.attrs_spec.standard_name}', Got: '{actual_attrs.get('standard_name')}'."
        )
    elif (
        spec.attrs_spec.standard_name is not None
        and "standard_name" not in actual_attrs
    ):
        errors.append(f"{prefix} missing expected attribute 'standard_name'.")

    return errors


def validate_dataset_against_schema(
    xr_dataset: xr.Dataset, schema: DatasetSchema
) -> List[str]:
    """
    Validates an xarray.Dataset against a DatasetSchema.
    Returns a list of error messages. An empty list means validation passed.
    """
    all_errors: List[str] = []

    # 1. Validate dataset-level attributes
    if schema.dataset_attrs_spec:
        for key, expected_value in schema.dataset_attrs_spec.items():
            actual_value = xr_dataset.attrs.get(key)
            if expected_value == "__EXIST__":  # Convention to check for existence only
                if key not in xr_dataset.attrs:
                    all_errors.append(
                        f"Dataset attribute '{key}': Expected to exist, but not found."
                    )
            elif actual_value != expected_value:
                all_errors.append(
                    f"Dataset attribute '{key}': Expected value '{expected_value}', Got '{actual_value}'."
                )

    # 2. Validate coordinates
    if schema.coords_spec:
        for coord_name, coord_spec in schema.coords_spec.items():
            if coord_name not in xr_dataset.coords:
                all_errors.append(f"Missing expected coordinate: '{coord_name}'.")
                continue
            all_errors.extend(
                validate_variable_against_spec(
                    coord_name,
                    xr_dataset.coords[coord_name],
                    coord_spec,
                    is_coordinate=True,
                )
            )

    # Check for unspecified coordinates (optional, could be a strictness setting)
    # for actual_coord_name in xr_dataset.coords:
    #     if schema.coords_spec is None or actual_coord_name not in schema.coords_spec:
    #         all_errors.append(f"Found unspecified coordinate: '{actual_coord_name}'.")

    # 3. Validate data variables
    if schema.data_vars_spec:  # Should always be present based on schema def
        for var_name, var_spec in schema.data_vars_spec.items():
            if var_name not in xr_dataset.data_vars:
                all_errors.append(f"Missing expected data variable: '{var_name}'.")
                continue
            all_errors.extend(
                validate_variable_against_spec(
                    var_name,
                    xr_dataset.data_vars[var_name],
                    var_spec,
                    is_coordinate=False,
                )
            )

    # Check for unspecified data variables (optional)
    # for actual_var_name in xr_dataset.data_vars:
    #     if actual_var_name not in schema.data_vars_spec:
    #         all_errors.append(f"Found unspecified data variable: '{actual_var_name}'.")

    return all_errors


# --- Example TOML Schema String ---
EXAMPLE_TOML_SCHEMA = """
schema_description = "Schema for Standard Atmospheric Profile Data V3.0"

# Expected dataset-level attributes
[dataset_attrs_spec]
Conventions = "CF-1.8"
title = "__EXIST__" # Special value to indicate key must exist, value can be anything
institution = "University of Example"

# Specification for dataset-level coordinates
[coords_spec.time]
dims = ["time"]
dtype = "datetime64[ns]"
  [coords_spec.time.attrs_spec]
  units = "seconds since 1970-01-01T00:00:00Z"
  long_name = "Time"
  standard_name = "time" # Added standard_name for time
  unit_base_rep = [["[time]", 1]]
  [coords_spec.time.shape_spec]
  ndim = 1

[coords_spec.level]
dims = ["level"]
dtype = "float32"
  [coords_spec.level.attrs_spec]
  units = "hPa"
  long_name = "Pressure Level"
  standard_name = "air_pressure"
  unit_base_rep = [["[mass]", 1], ["[length]", -1], ["[time]", -2]]
  [coords_spec.level.shape_spec]
  ndim = 1

# Specification for data variables
[data_vars_spec.temperature]
dims = ["time", "level"] # Simplified for example
dtype = "float32"
  [data_vars_spec.temperature.attrs_spec]
  units = "K"
  long_name = "Air Temperature"
  standard_name = "air_temperature"
  unit_base_rep = [["[temperature]", 1]]
  [data_vars_spec.temperature.shape_spec]
  ndim = 2

[data_vars_spec.specific_humidity]
dims = ["time", "level"] # Simplified for example
dtype = "float64" # Changed dtype for testing
  [data_vars_spec.specific_humidity.attrs_spec]
  units = "kg kg-1"
  long_name = "Specific Humidity"
  # standard_name deliberately omitted from spec to test optionality
  unit_base_rep = [["[]", 0]]
  [data_vars_spec.specific_humidity.shape_spec]
  ndim = 2
"""

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Load the schema from the TOML string
    schema: DatasetSchema = msgspec.toml.decode(
        EXAMPLE_TOML_SCHEMA.encode("utf-8"), type=DatasetSchema
    )
    print(f"Loaded schema: {schema.schema_description}")

    # 2. Create a sample compliant xarray.Dataset for testing
    time_data = np.array(
        ["2023-01-01T00:00:00", "2023-01-01T06:00:00"], dtype="datetime64[ns]"
    )
    level_data = np.array([1000.0, 925.0, 850.0], dtype=np.float32)
    temp_data = np.random.rand(2, 3).astype(np.float32) * 30 + 273.15
    q_data = np.random.rand(2, 3).astype(np.float64) * 0.01

    compliant_ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "level"),
                temp_data,
                {
                    "units": "K",
                    "long_name": "Air Temperature",
                    "standard_name": "air_temperature",
                    "unit_base_rep": (("[temperature]", 1),),
                },
            ),
            "specific_humidity": (
                ("time", "level"),
                q_data,
                {
                    "units": "kg kg-1",
                    "long_name": "Specific Humidity",
                    # standard_name is not in the data, matching spec's optionality
                    "unit_base_rep": (("[]", 0),),
                },
            ),
        },
        coords={
            "time": (
                "time",
                time_data,
                {
                    "units": "seconds since 1970-01-01T00:00:00Z",  # Matched schema
                    "long_name": "Time",
                    "standard_name": "time",
                    "unit_base_rep": (("[time]", 1),),
                },
            ),
            "level": (
                "level",
                level_data,
                {
                    "units": "hPa",
                    "long_name": "Pressure Level",
                    "standard_name": "air_pressure",
                    "unit_base_rep": (("[mass]", 1), ("[length]", -1), ("[time]", -2)),
                },
            ),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "Sample Compliant Dataset",
            "institution": "University of Example",
            "extra_attr_not_in_spec": "This is fine",
        },
    )
    print("\n--- Validating Compliant Dataset ---")
    errors_compliant = validate_dataset_against_schema(compliant_ds, schema)
    if not errors_compliant:
        print("Compliant dataset PASSED validation.")
    else:
        print("Compliant dataset FAILED validation with errors:")
        for err in errors_compliant:
            print(f"  - {err}")

    # 3. Create a sample non-compliant xarray.Dataset
    non_compliant_ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "level"),
                temp_data.astype(np.float64),
                {  # Wrong dtype
                    "units": "Celsius",  # Wrong units
                    "long_name": "Air Temperature",
                    "standard_name": "air_temperature",
                    "unit_base_rep": (("[temperature]", 1),),
                },
            ),
            # specific_humidity is missing
        },
        coords={
            "time": (
                "time",
                time_data,
                {
                    "units": "days since 2000-01-01",  # Wrong units
                    "long_name": "Time",
                    "unit_base_rep": (("[time]", 1),),
                },
            ),
            # level coordinate is missing
        },
        attrs={
            "Conventions": "CF-1.7",  # Wrong value
            # title is missing
            "institution": "Some Other Place",
        },
    )
    print("\n--- Validating Non-Compliant Dataset ---")
    errors_non_compliant = validate_dataset_against_schema(non_compliant_ds, schema)
    if not errors_non_compliant:
        print("Non-compliant dataset PASSED validation (UNEXPECTED).")
    else:
        print("Non-compliant dataset FAILED validation with errors (as expected):")
        for err in errors_non_compliant:
            print(f"  - {err}")

    # Example for a variable not specified in schema (if strict checking for unknowns is added)
    # compliant_ds_with_extra_var = compliant_ds.copy()
    # compliant_ds_with_extra_var["extra_temp"] = (("time", "level"), temp_data, {"units":"X"})
    # print("\n--- Validating Compliant Dataset with Extra Variable ---")
    # errors_extra = validate_dataset_against_schema(compliant_ds_with_extra_var, schema)
    # ...
