from pathlib import Path

import msgspec
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from density import mass
from helpful_unit_conversions import (
    AMU_IN_GRAMS,
    BAR_TO_BARYE,
    BOLTZMANN_CONSTANT_IN_CGS,
)
from molecular_crosssections.molecular_metrics import (
    MOLECULAR_WEIGHTS,
    MoleculeMetrics,
    calculate_molecular_metrics,
)
from test_inputs.test_2M2236b_G395H.test_2M2236b_G395H_vertical_inputs import (
    user_vertical_inputs,
)
from vertical.altitude import calculate_altitude_profile
from xarray_serialization import XarrayDataArray, XarrayDataset

if __name__ == "__main__":
    current_directory: Path = Path(__file__).parent
    data_structure_directory: Path = (
        current_directory / "test_inputs" / "test_data_structures"
    )

    planet_mass_in_g: float = mass(
        radius_in_cm=user_vertical_inputs.planet_radius_in_cm,
        surface_gravity_in_cgs=user_vertical_inputs.planet_gravity_in_cgs,
    )

    fiducial_molecular_metrics: MoleculeMetrics = calculate_molecular_metrics(
        gas_abundances=user_vertical_inputs.mixing_ratios_by_level
    )

    molecular_weight: float = (
        fiducial_molecular_metrics.mean_molecular_weight * AMU_IN_GRAMS
    )

    altitudes_in_cm: NDArray[np.float64] = calculate_altitude_profile(
        log_pressures=user_vertical_inputs.pressures_by_level,
        temperatures=user_vertical_inputs.temperatures_by_level,
        mean_molecular_weight_in_g=molecular_weight,
        planet_radius_in_cm=user_vertical_inputs.planet_radius_in_cm,
        planet_mass_in_g=planet_mass_in_g,
    )
    altitudes_in_planet_radii: NDArray[np.float64] = (
        altitudes_in_cm / user_vertical_inputs.planet_radius_in_cm
    )

    shared_coordinates: dict[str, xr.Variable] = {
        "pressure": xr.Variable(
            dims="pressure",
            data=10**user_vertical_inputs.pressures_by_level,
            attrs={"units": "bar"},
        )
    }

    temperature_dataarray: xr.DataArray = xr.DataArray(
        user_vertical_inputs.temperatures_by_level,
        coords=shared_coordinates,
        dims=["pressure"],
        name="temperature",
        attrs={"units": "kelvin"},
    )

    altitude_dataarray: xr.DataArray = xr.DataArray(
        altitudes_in_cm,
        coords=shared_coordinates,
        dims=["pressure"],
        name="altitude",
        attrs={"units": "cm"},
    )

    scaled_altitude_dataarray: xr.DataArray = xr.DataArray(
        altitudes_in_planet_radii,
        coords=shared_coordinates,
        dims=["pressure"],
        name="altitude_in_planet_radii",
        attrs={"scale": "planet radius", "units": "dimensionless"},
    )

    molecular_mixing_ratios: dict[str, xr.DataArray] = {
        species: xr.DataArray(
            np.ones_like(user_vertical_inputs.pressures_by_level) * abundance,
            coords=shared_coordinates,
            dims=["pressure"],
            name=f"{species} mixing ratio",
            attrs={
                "molecular_weight": MOLECULAR_WEIGHTS[species],
                "units": "dimensionless",
            },
        )
        for species, abundance in user_vertical_inputs.mixing_ratios_by_level.items()
    }

    molecular_mixing_ratios_dataset: xr.Dataset = xr.Dataset(
        data_vars=molecular_mixing_ratios,
        coords=shared_coordinates,
        attrs=msgspec.structs.asdict(fiducial_molecular_metrics),
    )

    molecular_partial_pressure_fractions: dict[str, xr.DataArray] = {
        species: xr.DataArray(
            abundance.attrs["molecular_weight"]
            * abundance.values
            / molecular_mixing_ratios_dataset.attrs["mean_molecular_weight"],
            coords=shared_coordinates,
            dims=["pressure"],
            name=f"{species} partial pressure",
            attrs={"molecular_weight": MOLECULAR_WEIGHTS[species], "units": "bar"},
        )
        for species, abundance in molecular_mixing_ratios_dataset.data_vars.items()
    }

    molecular_partial_pressure_fractions_dataset: xr.Dataset = xr.Dataset(
        data_vars=molecular_partial_pressure_fractions,
        coords=shared_coordinates,
        attrs=msgspec.structs.asdict(fiducial_molecular_metrics),
    )

    vertical_dataset: xr.Dataset = xr.Dataset(
        data_vars={
            "temperature": temperature_dataarray,
            "altitude": altitude_dataarray,
            "altitude_in_planet_radii": scaled_altitude_dataarray,
        },
        coords=shared_coordinates,
        attrs={
            "planet_radius_in_cm": user_vertical_inputs.planet_radius_in_cm,
            "planet_gravity_in_cgs": user_vertical_inputs.planet_gravity_in_cgs,
            "planet_mass_in_g": planet_mass_in_g,
        },
    )

    molecular_number_densities: dict[str, xr.DataArray] = {
        species: xr.DataArray(
            partial_pressure_fraction
            * (partial_pressure_fraction.pressure.values * BAR_TO_BARYE)
            / (BOLTZMANN_CONSTANT_IN_CGS * vertical_dataset.temperature.values),
            coords=shared_coordinates,
            dims=["pressure"],
            name=f"{species} mass density",
            attrs={"molecular_weight": MOLECULAR_WEIGHTS[species], "units": "cm^-3"},
        )
        for species, partial_pressure_fraction in molecular_partial_pressure_fractions_dataset.data_vars.items()
    }

    molecular_number_densities_dataset: xr.Dataset = xr.Dataset(
        data_vars=molecular_number_densities,
        coords=shared_coordinates,
        attrs=msgspec.structs.asdict(fiducial_molecular_metrics),
    )

    molecular_mass_densities: dict[str, xr.DataArray] = {
        species: xr.DataArray(
            number_density * number_density.attrs["molecular_weight"] * AMU_IN_GRAMS,
            coords=shared_coordinates,
            dims=["pressure"],
            name=f"{species} mass density",
            attrs={"molecular_weight": MOLECULAR_WEIGHTS[species], "units": "g cm^-3"},
        )
        for species, number_density in molecular_number_densities_dataset.data_vars.items()
    }

    molecular_mass_densities_dataset: xr.Dataset = xr.Dataset(
        data_vars=molecular_mass_densities,
        coords=shared_coordinates,
        attrs=msgspec.structs.asdict(fiducial_molecular_metrics),
    )

    vertical_datatree_structure: dict[str, xr.Dataset] = {
        "./vertical_structure": vertical_dataset,
        "./molecular_inputs/mixing_ratios": molecular_mixing_ratios_dataset,
        "./molecular_inputs/partial_pressure_fractions": molecular_partial_pressure_fractions_dataset,
        "./molecular_inputs/number_densities": molecular_number_densities_dataset,
        "./molecular_inputs/mass_densities": molecular_mass_densities_dataset,
    }

    vertical_datatree: xr.DataTree = xr.DataTree.from_dict(vertical_datatree_structure)
    vertical_datatree.to_netcdf(data_structure_directory / "test_vertical_structure.nc")

    vertical_as_dict: dict = vertical_dataset.to_dict()
    altitude_as_dict: dict = vertical_dataset.altitude.to_dict()
    number_density_as_dict: dict = molecular_number_densities_dataset.to_dict()

    msg_altitude: XarrayDataArray = XarrayDataArray(**altitude_as_dict)
    msg_vertical: XarrayDataset = XarrayDataset(**vertical_as_dict)

    with open(data_structure_directory / "test_altitude.toml", "wb") as file:
        file.write(msgspec.toml.encode(msg_altitude))

    with open(data_structure_directory / "test_vertical.toml", "wb") as file:
        file.write(msgspec.toml.encode(msg_vertical))

    with open(data_structure_directory / "test_number_density.toml", "wb") as file:
        file.write(msgspec.toml.encode(number_density_as_dict))
