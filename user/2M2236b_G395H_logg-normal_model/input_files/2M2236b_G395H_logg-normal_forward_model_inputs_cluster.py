from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import ModuleType

import numpy as np
import xarray as xr

from constants_and_conversions import AMU_IN_GRAMS, EARTH_RADIUS_IN_CM, PARSEC_TO_CM
from density import calculate_mass_from_radius_and_surface_gravity
from material.gases.molecular_metrics import calculate_mean_molecular_weight
from user.input_importers import import_model_id
from user.input_structs import UserForwardModelInputs, UserVerticalModelInputs
from vertical.altitude import (
    altitudes_by_level_to_by_layer,
    altitudes_by_level_to_path_lengths,
    calculate_altitude_profile,
)

current_directory: Path = Path(__file__).parent
model_directory: Path = current_directory.parent
input_files_directory: Path = model_directory / "input_files"
intermediate_output_directory: Path = model_directory / "intermediate_outputs"
user_directory: Path = model_directory.parent
potluck_directory: Path = user_directory.parent

################### FORWARD MODEL DATA ###################
model_directory_label: str = "2M2236b_G395H_logg-normal"

opacity_catalog: str = "jwst50k"  # "jwst50k", "wide-jwst", "wide"

opacities_directory: Path = Path("/Volumes/Orange") / "Opacities_0v10"
# opacities_directory: Path = Path("/home/gba8kj/storage") / "Opacities_0v10"
opacity_data_directory: Path = opacities_directory / "gases"

catalog_filepath: Path = opacity_data_directory / f"{opacity_catalog}.nc"
crosssection_catalog: xr.Dataset = xr.open_dataset(catalog_filepath)

reference_model_filepath: Path = (
    input_files_directory
    / "reference_inputs"
    / "G395H"
    / "2M2236b_NIRSpec_G395H_R500_APOLLO.nc"
)
# reference_model_filepath: Path = model_directory / "reference_data_wavelengths.nc"
reference_model: xr.Dataset = xr.open_dataset(reference_model_filepath)
reference_model_flux_lambda: xr.DataArray = reference_model.flux_lambda
reference_model_flux_lambda_errors: xr.DataArray = reference_model.flux_lambda_error
reference_model_wavelengths: xr.DataArray = reference_model.wavelength

parent_directory: str = "user"

model_case_name: str = import_model_id(
    model_directory_label=model_directory_label, parent_directory=parent_directory
)

vertical_structure_module: ModuleType = import_module(
    f"{parent_directory}.{model_directory_label}_model.input_files.{model_directory_label}_vertical_inputs"
)

build_uniform_vertical_structure: Callable = (
    vertical_structure_module.build_uniform_model_inputs
)

default_vertical_structure_datatree: xr.DataTree = (
    vertical_structure_module.default_vertical_structure_datatree
)

default_mixing_ratios_as_xarray: xr.Dataset = (
    vertical_structure_module.default_mixing_ratios_by_level_as_xarray
)

distance_to_system_in_cm: float = 63.0 * PARSEC_TO_CM  # 64.5

stellar_radius_in_cm: float = 1.53054e10
stellar_radius_in_cm_as_xarray: xr.DataArray = xr.DataArray(
    data=stellar_radius_in_cm,
    dims=tuple(),
    attrs={"units": "cm"},
)


def build_forward_model(
    vertical_structure: xr.DataTree,
    crosssection_catalog: xr.Dataset = crosssection_catalog,
    output_wavelengths: xr.DataArray = reference_model_wavelengths,
    distance_to_system_in_cm: float = distance_to_system_in_cm,
    stellar_radius_in_cm: float = stellar_radius_in_cm,
) -> UserForwardModelInputs:
    mean_molecular_weight_in_g: float = (
        calculate_mean_molecular_weight(vertical_structure.chemistry) * AMU_IN_GRAMS
    )

    planet_mass_in_g: float = calculate_mass_from_radius_and_surface_gravity(
        vertical_structure.planet_radius_in_cm,
        vertical_structure.planet_gravity_in_cgs,
    )

    altitudes_in_cm: xr.DataArray = xr.DataArray(
        data=calculate_altitude_profile(
            vertical_structure.log_pressures_by_level,
            vertical_structure.temperatures_by_level,
            mean_molecular_weight_in_g,
            vertical_structure.planet_radius_in_cm,
            planet_mass_in_g,
        ),
        coords={
            "pressure": xr.Variable(
                dims="pressure",
                data=vertical_structure.pressures_by_level,
                attrs={"units": "bar"},
            )
        },
        dims=("pressure",),
        name="altitude",
        attrs={"units": "cm"},
    )

    path_lengths_by_layer: xr.DataArray = altitudes_by_level_to_path_lengths(
        altitudes_in_cm
    )

    altitudes_by_layer: xr.DataArray = altitudes_by_level_to_by_layer(altitudes_in_cm)

    return UserForwardModelInputs(
        vertical_inputs=vertical_structure,
        crosssection_catalog=crosssection_catalog,
        output_wavelengths=output_wavelengths,
        path_lengths_by_layer=path_lengths_by_layer,
        altitudes_by_layer=altitudes_by_layer,
        distance_to_system_in_cm=distance_to_system_in_cm,
        stellar_radius_in_cm=stellar_radius_in_cm,
    )


default_forward_model_inputs: UserForwardModelInputs = build_forward_model(
    default_vertical_structure_datatree
)

##########################################################
"""
def build_uniform_model_inputs(
    uniform_log_abundances: dict[str, float],
    initial_temp_sample: float,  # [75, 3975]
    proportions_down: np.ndarray,  # shape = (5,), all [0, 1]
    proportions_up: np.ndarray,  # shape = (4,), all [0, 1]
    planet_radius_in_cm: float = planet_radius_in_cm,
    planet_gravity_in_cgs: float = planet_gravity_in_cgs,
    log_pressures_by_level: np.ndarray[np.float64] = log_pressures_by_level,
    pressures_by_level: np.ndarray[np.float64] = pressures_by_level,
    # temperatures_by_level: np.ndarray[np.float64] = temperatures_by_level,
    filler_species: str = "h2only",
)
"""


def build_uniform_mixing_ratio_forward_model(
    planet_radius_relative_to_earth: float,
    planet_log_gravity_in_cgs: float,
    uniform_log_abundances: dict[str, float],
    initial_temp_sample: float,
    proportions_down: np.ndarray,
    proportions_up: np.ndarray,
) -> UserForwardModelInputs:
    planet_radius_in_cm: float = planet_radius_relative_to_earth * EARTH_RADIUS_IN_CM
    planet_gravity_in_cgs: float = 10**planet_log_gravity_in_cgs

    vertical_inputs: UserVerticalModelInputs = build_uniform_vertical_structure(
        uniform_log_abundances=uniform_log_abundances,
        initial_temp_sample=initial_temp_sample,
        proportions_down=proportions_down,
        proportions_up=proportions_up,
        planet_radius_in_cm=planet_radius_in_cm,
        planet_gravity_in_cgs=planet_gravity_in_cgs,
    )

    return build_forward_model(vertical_structure=vertical_inputs)
