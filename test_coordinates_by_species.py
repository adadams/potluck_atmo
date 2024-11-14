from pathlib import Path

import xarray as xr

current_directory: Path = Path(__file__).parent

test_gas_absorption_coefficient_directory: Path = (
    current_directory / "test_inputs" / "test_data_structures"
)

test_gas_absorption_coefficient_filepath: Path = (
    test_gas_absorption_coefficient_directory
    / "test_gas_absorption_coefficients_nir.nc"
)

test_gas_absorption_coefficient_dataset: xr.Dataset = xr.open_dataset(
    test_gas_absorption_coefficient_filepath
)

print(f"{test_gas_absorption_coefficient_dataset=}")

test_gas_absorption_coefficient_dataarray: xr.DataArray = (
    test_gas_absorption_coefficient_dataset.to_array(
        dim="species", name="gas_absorption_coefficient"
    ).assign_attrs(units="cm^(-1)")
)

print(f"{test_gas_absorption_coefficient_dataarray=}")

test_gas_absorption_coefficient_dataarray.to_netcdf(
    test_gas_absorption_coefficient_directory
    / "test_gas_absorption_coefficients_dataarray_nir.nc"
)
