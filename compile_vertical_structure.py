from pathlib import Path
from typing import Final, Optional

import msgspec
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from altitude import calculate_altitude_profile
from density import mass
from molecular_crosssections.molecular_metrics import (
    MoleculeMetrics,
    calculate_molecular_metrics,
)
from test_inputs.test_inputs_as_dicts import (
    fiducial_test_abundances,
    test_log_pressures,
    test_temperatures,
)

GRAVITATIONAL_CONSTANT_IN_CGS: Final[float] = 6.67408e-8  # [cm^3 g^-1 s^-2]
BOLTZMANN_CONSTANT_IN_CGS: Final[float] = 1.38065e-16  # [cm^2 g s^-2 K^-1]
AMU_IN_GRAMS: Final[float] = 1.66054e-24
BAR_TO_BARYE: Final[float] = 1.0e6
EARTH_RADIUS_IN_CM: Final[float] = 6.371e8
JUPITER_RADIUS_IN_CM: Final[float] = 6.991e8
JUPITER_MASS_IN_G: Final[float] = 1.898e30


if __name__ == "__main__":
    current_directory: Path = Path(__file__).parent
    test_data_structure_directory: Path = (
        current_directory / "test_inputs" / "test_data_structures"
    )

    fiducial_molecular_metrics: MoleculeMetrics = calculate_molecular_metrics(
        gas_abundances=fiducial_test_abundances
    )

    test_planet_radius: float = 13.526406 * EARTH_RADIUS_IN_CM
    test_planet_gravity: float = 10**4.766214  # cm/s^2
    test_planet_mass: float = mass(
        radius_in_cm=test_planet_radius, surface_gravity_in_cgs=test_planet_gravity
    )

    test_molecular_weight: float = (
        fiducial_molecular_metrics.mean_molecular_weight * AMU_IN_GRAMS
    )

    test_altitudes_in_cm: NDArray[np.float64] = calculate_altitude_profile(
        log_pressures=test_log_pressures,
        temperatures=test_temperatures,
        mean_molecular_weight_in_g=test_molecular_weight,
        planet_radius_in_cm=test_planet_radius,
        planet_mass_in_g=test_planet_mass,
    )
    test_altitudes_in_planet_radii: NDArray[np.float64] = (
        test_altitudes_in_cm / test_planet_radius
    )

    shared_coordinates: dict[str, xr.Variable] = {
        "pressure": xr.Variable(
            "pressure", 10**test_log_pressures, attrs={"units": "bar"}
        )
    }

    temperature_dataarray: xr.DataArray = xr.DataArray(
        test_temperatures,
        coords=shared_coordinates,
        dims=["pressure"],
        name="temperature",
        attrs={"units": "kelvin"},
    )

    altitude_dataarray: xr.DataArray = xr.DataArray(
        test_altitudes_in_cm,
        coords=shared_coordinates,
        dims=["pressure"],
        name="altitude",
        attrs={"units": "cm"},
    )

    scaled_altitude_dataarray: xr.DataArray = xr.DataArray(
        test_altitudes_in_planet_radii,
        coords=shared_coordinates,
        dims=["pressure"],
        name="altitude_in_planet_radii",
        attrs={"scale": "planet radius", "units": "dimensionless"},
    )

    test_molecular_mixing_ratios: dict[str, xr.DataArray] = {
        species: xr.DataArray(
            np.ones_like(test_log_pressures) * abundance,
            coords=shared_coordinates,
            dims=["pressure"],
            name=f"{species} mixing ratio",
            attrs={"units": "dimensionless"},
        )
        for species, abundance in fiducial_test_abundances.items()
    }

    test_vertical_dataset: xr.Dataset = xr.Dataset(
        data_vars={
            "temperature": temperature_dataarray,
            "altitude": altitude_dataarray,
            "altitude_in_planet_radii": scaled_altitude_dataarray,
        },
        coords=shared_coordinates,
        attrs={
            "planet_radius_in_cm": test_planet_radius,
            "planet_gravity_in_cgs": test_planet_gravity,
            "planet_mass_in_g": test_planet_mass,
        },
    )

    test_molecular_inputs_dataset: xr.Dataset = xr.Dataset(
        data_vars=test_molecular_mixing_ratios,
        coords=shared_coordinates,
        attrs=msgspec.structs.asdict(fiducial_molecular_metrics),
    )

    vertical_datatree_structure: dict[str, xr.Dataset] = {
        "./vertical_structure": test_vertical_dataset,
        "./molecular_inputs": test_molecular_inputs_dataset,
    }

    vertical_datatree: xr.DataTree = xr.DataTree.from_dict(vertical_datatree_structure)
    vertical_datatree.to_netcdf(
        test_data_structure_directory / "test_vertical_structure.nc"
    )

    test_vertical_as_dict: dict = test_vertical_dataset.to_dict()
    test_altitude_as_dict: dict = test_vertical_dataset.altitude.to_dict()

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

    msg_altitude: XarrayDataArray = XarrayDataArray(**test_altitude_as_dict)
    msg_vertical: XarrayDataset = XarrayDataset(**test_vertical_as_dict)

    with open(test_data_structure_directory / "test_altitude.toml", "wb") as file:
        file.write(msgspec.toml.encode(msg_altitude))

    with open(test_data_structure_directory / "test_vertical.toml", "wb") as file:
        file.write(msgspec.toml.encode(msg_vertical))
