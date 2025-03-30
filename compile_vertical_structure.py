from pathlib import Path
from typing import Any, NamedTuple

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
    calculate_mean_molecular_weight,
    calculate_weighted_molecular_weights,
    mixing_ratios_to_number_densities,
)
from user.input_importers import import_model_id, import_user_vertical_inputs
from user.input_structs import UserVerticalModelInputs
from vertical.altitude import (
    calculate_altitude_profile,
    convert_dataset_by_pressure_levels_to_pressure_layers,
)
from xarray_functional_wrappers import save_xarray_outputs_to_file
from xarray_serialization import XarrayDataArray, XarrayDataset

current_directory: Path = Path(__file__).parent


class ForwardModelXarrayInputs(NamedTuple):
    by_level: xr.Dataset
    by_layer: xr.Dataset


@save_xarray_outputs_to_file
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
            pressure_in_cgs=user_vertical_inputs.pressures_by_level * BAR_TO_BARYE,
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


def compile_comprehensive_vertical_structure(
    user_vertical_inputs: UserVerticalModelInputs,
) -> xr.DataTree:
    planet_mass_in_g: float = calculate_mass_from_radius_and_surface_gravity(
        radius_in_cm=user_vertical_inputs.planet_radius_in_cm,
        surface_gravity_in_cgs=user_vertical_inputs.planet_gravity_in_cgs,
    )

    mean_molecular_weights_in_g: float = (
        calculate_mean_molecular_weight(
            mixing_ratios=user_vertical_inputs.mixing_ratios_by_level
        )
        * AMU_IN_GRAMS
    )

    altitudes_in_cm: NDArray[np.float64] = calculate_altitude_profile(
        log_pressures_in_cgs=user_vertical_inputs.log_pressures_by_level,
        temperatures_in_K=user_vertical_inputs.temperatures_by_level,
        mean_molecular_weights_in_g=mean_molecular_weights_in_g,
        planet_radius_in_cm=user_vertical_inputs.planet_radius_in_cm,
        planet_mass_in_g=planet_mass_in_g,
    )
    altitudes_in_planet_radii: NDArray[np.float64] = (
        altitudes_in_cm / user_vertical_inputs.planet_radius_in_cm
    )

    shared_coordinates: dict[str, xr.Variable] = {
        "pressure": xr.Variable(
            dims="pressure",
            data=user_vertical_inputs.pressures_by_level,
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

    shared_molecular_attrs: dict[str, Any] = {
        "mean_molecular_weight": mean_molecular_weights_in_g / AMU_IN_GRAMS
    }  # msgspec.structs.asdict(fiducial_molecular_metrics),

    molecular_mixing_ratios_dataset: xr.Dataset = xr.Dataset(
        data_vars=molecular_mixing_ratios,
        coords=shared_coordinates,
        attrs=shared_molecular_attrs,
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
        attrs=shared_molecular_attrs,
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
        attrs=shared_molecular_attrs,
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
        attrs=shared_molecular_attrs,
    )

    vertical_datatree_structure: dict[str, xr.Dataset] = {
        "./vertical_structure": vertical_dataset,
        "./molecular_inputs/mixing_ratios": molecular_mixing_ratios_dataset,
        "./molecular_inputs/partial_pressure_fractions": molecular_partial_pressure_fractions_dataset,
        "./molecular_inputs/number_densities": molecular_number_densities_dataset,
        "./molecular_inputs/mass_densities": molecular_mass_densities_dataset,
    }

    return xr.DataTree.from_dict(vertical_datatree_structure)


def test_serialized_vertical_outputs(
    vertical_datatree: xr.DataTree,
    model_directory_label: str,
    serial_output_directory: Path,
) -> None:
    model_case_name: str = import_model_id(
        model_directory_label=model_directory_label, parent_directory="user"
    )
    # Pull apart pieces and serialize, as a test.
    vertical_dataset: xr.Dataset = vertical_datatree["vertical_structure"].to_dataset()
    molecular_number_densities_dataset: xr.Dataset = vertical_datatree[
        "molecular_inputs"
    ]["number_densities"].to_dataset()

    vertical_as_dict: dict = vertical_dataset.to_dict()
    altitude_as_dict: dict = vertical_dataset.altitude.to_dict()
    number_density_as_dict: dict = molecular_number_densities_dataset.to_dict()

    msg_altitude: XarrayDataArray = XarrayDataArray(**altitude_as_dict)
    msg_vertical: XarrayDataset = XarrayDataset(**vertical_as_dict)

    with open(
        serial_output_directory / f"{model_case_name}_altitude.toml", "wb"
    ) as file:
        file.write(msgspec.toml.encode(msg_altitude))

    with open(
        serial_output_directory / f"{model_case_name}_vertical.toml", "wb"
    ) as file:
        file.write(msgspec.toml.encode(msg_vertical))

    with open(
        serial_output_directory / f"{model_case_name}_number_density.toml", "wb"
    ) as file:
        file.write(msgspec.toml.encode(number_density_as_dict))


if __name__ == "__main__":
    model_directory_label: str = "test_isothermal"

    current_directory: Path = Path(__file__).parent
    user_directory: Path = current_directory / "user"
    model_directory: Path = user_directory / f"{model_directory_label}_model"
    intermediate_output_directory: Path = model_directory / "intermediate_outputs"
    serial_output_directory: Path = model_directory / "serial_outputs"

    model_case_name: str = import_model_id(
        model_directory_label=model_directory_label, parent_directory="user"
    )
    user_vertical_inputs: UserVerticalModelInputs = import_user_vertical_inputs(
        model_directory_label=model_directory_label, parent_directory="user"
    )

    vertical_datatree: xr.DataTree = compile_comprehensive_vertical_structure(
        user_vertical_inputs
    )
    vertical_datatree.to_netcdf(
        intermediate_output_directory / f"{model_case_name}_vertical_structure.nc"
    )

    test_serialized_vertical_outputs(
        vertical_datatree=vertical_datatree,
        model_directory_label=model_directory_label,
        serial_output_directory=serial_output_directory,
    )
