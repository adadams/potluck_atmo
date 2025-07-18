from pathlib import Path

import xarray as xr

from constants_and_conversions import AMU_IN_GRAMS, PARSEC_TO_CM
from density import calculate_mass_from_radius_and_surface_gravity
from material.gases.molecular_metrics import calculate_mean_molecular_weight
from user.input_importers import import_model_id, import_user_vertical_inputs
from user.input_structs import UserForwardModelInputs, UserVerticalModelInputs
from vertical.altitude import (
    altitudes_by_level_to_by_layer,
    altitudes_by_level_to_path_lengths,
    calculate_altitude_profile,
)

model_directory: Path = Path(__file__).parent.parent  # NOTE: bodge
intermediate_output_directory: Path = model_directory / "intermediate_outputs"
user_directory: Path = model_directory.parent
potluck_directory: Path = user_directory.parent

################### FORWARD MODEL DATA ###################
model_directory_label: str = "example_isothermal"

opacity_catalog: str = "wide-jwst"

opacity_data_directory: Path = Path("/Volumes/Orange") / "Opacities_0v10" / "gases"

catalog_filepath: Path = opacity_data_directory / f"{opacity_catalog}.nc"
crosssection_catalog: xr.Dataset = xr.open_dataset(catalog_filepath)

reference_model_filepath: Path = model_directory / "reference_data_wavelengths.nc"
reference_model: xr.Dataset = xr.open_dataset(reference_model_filepath)
reference_model_wavelengths: xr.DataArray = reference_model.wavelength


model_case_name: str = import_model_id(
    model_directory_label=model_directory_label, parent_directory="user"
)
user_vertical_inputs: UserVerticalModelInputs = import_user_vertical_inputs(
    model_directory_label=model_directory_label, parent_directory="user"
)

vertical_structure_datatree_path: Path = (
    intermediate_output_directory / f"{model_case_name}_vertical_structure.nc"
)

distance_to_system_in_cm: float = 64.5 * PARSEC_TO_CM

mean_molecular_weight_in_g: float = (
    calculate_mean_molecular_weight(user_vertical_inputs.mixing_ratios_by_level)
    * AMU_IN_GRAMS
)

planet_mass_in_g: float = calculate_mass_from_radius_and_surface_gravity(
    user_vertical_inputs.planet_radius_in_cm, user_vertical_inputs.planet_gravity_in_cgs
)

altitudes_in_cm: xr.DataArray = xr.DataArray(
    calculate_altitude_profile(
        user_vertical_inputs.log_pressures_by_level,
        user_vertical_inputs.temperatures_by_level,
        mean_molecular_weight_in_g,
        user_vertical_inputs.planet_radius_in_cm,
        planet_mass_in_g,
    ),
    coords={
        "pressure": xr.Variable(
            dims="pressure",
            data=user_vertical_inputs.pressures_by_level,
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

user_forward_model_inputs: UserForwardModelInputs = UserForwardModelInputs(
    vertical_inputs=user_vertical_inputs,
    crosssection_catalog=crosssection_catalog,
    output_wavelengths=reference_model_wavelengths,
    path_lengths_by_layer=path_lengths_by_layer,
    altitudes_by_layer=altitudes_by_layer,
    distance_to_system_in_cm=distance_to_system_in_cm,
)
##########################################################
