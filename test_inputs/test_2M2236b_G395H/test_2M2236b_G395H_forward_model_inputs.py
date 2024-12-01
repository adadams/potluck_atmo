from pathlib import Path

import xarray as xr

from constants_and_conversions import AMU_IN_GRAMS, PARSEC_TO_CM
from density import calculate_mass_from_radius_and_surface_gravity
from material.gases.molecular_metrics import calculate_mean_molecular_weight
from test_inputs.test_2M2236b_G395H.test_2M2236b_G395H_vertical_inputs import (
    user_vertical_inputs,
)
from test_inputs.test_data_structures.input_structs import UserForwardModelInputs
from vertical.altitude import (
    altitudes_by_level_to_path_lengths,
    calculate_altitude_profile,
)

potluck_directory: Path = Path(__file__).parent.parent.parent

################### FORWARD MODEL DATA ###################
opacity_catalog: str = "jwst50k"

opacity_data_directory: Path = (
    potluck_directory / "material" / "gases" / "reference_data"
)

catalog_filepath: Path = opacity_data_directory / f"{opacity_catalog}.nc"
crosssection_catalog_dataset: xr.Dataset = xr.open_dataset(catalog_filepath)

data_structure_directory: Path = (
    potluck_directory / "test_inputs" / "test_data_structures"
)

data_filepath: Path = data_structure_directory / "2M2236b_NIRSpec_G395H_R500_APOLLO.nc"
data: xr.Dataset = xr.open_dataset(data_filepath)
data_wavelengths: xr.DataArray = data.wavelength

vertical_structure_datatree_path: Path = (
    data_structure_directory / "test_vertical_structure.nc"
)

distance_to_system_in_cm: float = 63.0 * PARSEC_TO_CM

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

path_lengths_by_level: xr.DataArray = altitudes_by_level_to_path_lengths(
    altitudes_in_cm
)

user_forward_model_inputs: UserForwardModelInputs = UserForwardModelInputs(
    vertical_inputs=user_vertical_inputs,
    crosssection_catalog=crosssection_catalog_dataset,
    output_wavelengths=data_wavelengths,
    path_lengths_by_level=path_lengths_by_level,
    distance_to_system_in_cm=distance_to_system_in_cm,
)
##########################################################
