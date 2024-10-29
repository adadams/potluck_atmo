from pathlib import Path
from typing import Final

import msgspec
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from altitude import calculate_altitude_profile
from density import mass
from molecular_crosssections.molecular_metrics import (
    MOLECULAR_WEIGHTS,
    MoleculeMetrics,
    calculate_molecular_metrics,
)
from test_inputs.test_inputs_as_dicts import (
    fiducial_test_abundances,
    test_log_pressures,
    test_temperatures,
)
from xarray_serialization import XarrayDataArray, XarrayDataset

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
            attrs={
                "molecular_weight": MOLECULAR_WEIGHTS[species],
                "units": "dimensionless",
            },
        )
        for species, abundance in fiducial_test_abundances.items()
    }

    test_molecular_mixing_ratios_dataset: xr.Dataset = xr.Dataset(
        data_vars=test_molecular_mixing_ratios,
        coords=shared_coordinates,
        attrs=msgspec.structs.asdict(fiducial_molecular_metrics),
    )

    test_molecular_partial_pressure_fractions: dict[str, xr.DataArray] = {
        species: xr.DataArray(
            abundance.attrs["molecular_weight"]
            * abundance.values
            / test_molecular_mixing_ratios_dataset.attrs["mean_molecular_weight"],
            coords=shared_coordinates,
            dims=["pressure"],
            name=f"{species} partial pressure",
            attrs={"molecular_weight": MOLECULAR_WEIGHTS[species], "units": "bar"},
        )
        for species, abundance in test_molecular_mixing_ratios_dataset.data_vars.items()
    }

    test_molecular_partial_pressure_fractions_dataset: xr.Dataset = xr.Dataset(
        data_vars=test_molecular_partial_pressure_fractions,
        coords=shared_coordinates,
        attrs=msgspec.structs.asdict(fiducial_molecular_metrics),
    )

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

    test_molecular_number_densities: dict[str, xr.DataArray] = {
        species: xr.DataArray(
            partial_pressure_fraction
            * (partial_pressure_fraction.pressure.values * BAR_TO_BARYE)
            / (BOLTZMANN_CONSTANT_IN_CGS * test_vertical_dataset.temperature.values),
            coords=shared_coordinates,
            dims=["pressure"],
            name=f"{species} mass density",
            attrs={"molecular_weight": MOLECULAR_WEIGHTS[species], "units": "cm^-3"},
        )
        for species, partial_pressure_fraction in test_molecular_partial_pressure_fractions_dataset.data_vars.items()
    }

    test_molecular_number_densities_dataset: xr.Dataset = xr.Dataset(
        data_vars=test_molecular_number_densities,
        coords=shared_coordinates,
        attrs=msgspec.structs.asdict(fiducial_molecular_metrics),
    )

    test_molecular_mass_densities: dict[str, xr.DataArray] = {
        species: xr.DataArray(
            number_density * number_density.attrs["molecular_weight"] * AMU_IN_GRAMS,
            coords=shared_coordinates,
            dims=["pressure"],
            name=f"{species} mass density",
            attrs={"molecular_weight": MOLECULAR_WEIGHTS[species], "units": "g cm^-3"},
        )
        for species, number_density in test_molecular_number_densities_dataset.data_vars.items()
    }

    test_molecular_mass_densities_dataset: xr.Dataset = xr.Dataset(
        data_vars=test_molecular_mass_densities,
        coords=shared_coordinates,
        attrs=msgspec.structs.asdict(fiducial_molecular_metrics),
    )

    vertical_datatree_structure: dict[str, xr.Dataset] = {
        "./vertical_structure": test_vertical_dataset,
        "./molecular_inputs/mixing_ratios": test_molecular_mixing_ratios_dataset,
        "./molecular_inputs/partial_pressure_fractions": test_molecular_partial_pressure_fractions_dataset,
        "./molecular_inputs/number_densities": test_molecular_number_densities_dataset,
        "./molecular_inputs/mass_densities": test_molecular_mass_densities_dataset,
    }

    vertical_datatree: xr.DataTree = xr.DataTree.from_dict(vertical_datatree_structure)
    vertical_datatree.to_netcdf(
        test_data_structure_directory / "test_vertical_structure.nc"
    )

    test_vertical_as_dict: dict = test_vertical_dataset.to_dict()
    test_altitude_as_dict: dict = test_vertical_dataset.altitude.to_dict()
    test_number_density_as_dict: dict = (
        test_molecular_number_densities_dataset.to_dict()
    )

    msg_altitude: XarrayDataArray = XarrayDataArray(**test_altitude_as_dict)
    msg_vertical: XarrayDataset = XarrayDataset(**test_vertical_as_dict)

    with open(test_data_structure_directory / "test_altitude.toml", "wb") as file:
        file.write(msgspec.toml.encode(msg_altitude))

    with open(test_data_structure_directory / "test_vertical.toml", "wb") as file:
        file.write(msgspec.toml.encode(msg_vertical))

    with open(test_data_structure_directory / "test_number_density.toml", "wb") as file:
        file.write(msgspec.toml.encode(test_number_density_as_dict))
