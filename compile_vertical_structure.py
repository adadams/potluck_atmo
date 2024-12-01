from pathlib import Path
from typing import NamedTuple

import msgspec
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from constants_and_conversions import (
    AMU_IN_GRAMS,
    BAR_TO_BARYE,
    BOLTZMANN_CONSTANT_IN_CGS,
)
from density import calculate_mass_from_radius_and_surface_gravity
from material.gases.molecular_metrics import (
    MOLECULAR_WEIGHTS,
    MoleculeMetrics,
    calculate_mean_molecular_weight,
    calculate_molecular_metrics,
    calculate_weighted_molecular_weights,
    mixing_ratios_to_number_densities,
)
from test_inputs.test_2M2236b_G395H.test_2M2236b_G395H_vertical_inputs import (
    user_vertical_inputs,
)
from test_inputs.test_data_structures.input_structs import UserVerticalModelInputs
from vertical.altitude import (
    calculate_altitude_profile,
    convert_dataset_by_pressure_levels_to_pressure_layers,
)
from xarray_serialization import XarrayDataArray, XarrayDataset


class ForwardModelXarrayInputs(NamedTuple):
    by_level: xr.Dataset
    by_layer: xr.Dataset


def compile_vertical_structure_for_forward_model(
    user_vertical_inputs: UserVerticalModelInputs,
) -> xr.Dataset:
    pressure_coordinates: dict[str, xr.Variable] = {
        "pressure": xr.Variable(
            dims=("pressure",),
            data=user_vertical_inputs.pressures_by_level,
            attrs={"units": "bar"},
        )
    }

    temperature_dataarray: xr.DataArray = xr.DataArray(
        user_vertical_inputs.temperatures_by_level,
        coords=pressure_coordinates,
        dims=("pressure",),
        name="temperature",
        attrs={"units": "kelvin"},
    )

    molecular_weights_by_level: dict[str, NDArray[np.float64]] = (
        calculate_weighted_molecular_weights(
            mixing_ratios=user_vertical_inputs.mixing_ratios_by_level
        )
    )

    mean_molecular_weight_by_level: NDArray[np.float64] = (
        calculate_mean_molecular_weight(
            mixing_ratios=user_vertical_inputs.mixing_ratios_by_level
        )
    )

    number_densities: dict[str, NDArray[np.float64]] = {
        species: xr.DataArray(data=number_density_array, coords=pressure_coordinates)
        for species, number_density_array in mixing_ratios_to_number_densities(
            mixing_ratios_by_level=user_vertical_inputs.mixing_ratios_by_level,
            molecular_weights_by_level=molecular_weights_by_level,
            mean_molecular_weight_by_level=mean_molecular_weight_by_level,
            pressure_in_cgs=user_vertical_inputs.pressures_by_level,
            temperatures_in_K=user_vertical_inputs.temperatures_by_level,
        ).items()
    }

    number_densities_dataarray: xr.Dataset = xr.Dataset(
        data_vars=number_densities,
        coords=pressure_coordinates,
        attrs={"units": "cm^-3"},
    )

    number_densities_dataarray = number_densities_dataarray.to_array(
        dim="species", name="number_density"
    )

    inputs_by_level: xr.Dataset = xr.Dataset(
        {
            "temperature": temperature_dataarray,
            "number_density": number_densities_dataarray,
        }
    )

    inputs_by_layer: xr.Dataset = convert_dataset_by_pressure_levels_to_pressure_layers(
        inputs_by_level
    )

    return ForwardModelXarrayInputs(
        by_level=inputs_by_level,
        by_layer=inputs_by_layer,
    )


if __name__ == "__main__":
    current_directory: Path = Path(__file__).parent
    data_structure_directory: Path = (
        current_directory / "test_inputs" / "test_data_structures"
    )

    planet_mass_in_g: float = calculate_mass_from_radius_and_surface_gravity(
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
