import inspect
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, Final, Optional, TypeAlias

import msgspec
import numpy as np
import xarray as xr
from jax import numpy as jnp
from pint import UnitRegistry

from basic_types import PositiveValue, PressureDimension
from constants_and_conversions import AMU_IN_GRAMS, PARSEC_TO_CM
from density import calculate_mass_from_radius_and_surface_gravity
from material.gases.molecular_metrics import (
    calculate_mean_molecular_weight,
    mixing_ratios_to_number_densities,
)
from material.mixing_ratios import (
    UniformLogMixingRatios,
    generate_uniform_mixing_ratios,
)
from temperature.protocols import (
    TemperatureModel,
    TemperatureModelConstructor,
    TemperatureModelInputs,
)
from vertical.altitude import (
    altitudes_by_level_to_by_layer,
    altitudes_by_level_to_path_lengths,
    calculate_altitude_profile,
    convert_datatree_by_pressure_levels_to_pressure_layers,
)
from xarray_functional_wrappers import Dimensionalize, rename_and_unitize
from xarray_serialization import AttrsType

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory

DEFAULT_UNITS_SYSTEM: Final[str] = "cgs"
ureg: UnitRegistry = UnitRegistry(system=DEFAULT_UNITS_SYSTEM)
ureg.load_definitions(str(project_directory / "additional_units.txt"))

FundamentalParameters: TypeAlias = xr.Dataset
PressureProfile: TypeAlias = xr.Dataset
TemperatureProfile: TypeAlias = xr.Dataset
GasChemistryProfile: TypeAlias = xr.Dataset


class FundamentalParameterInputs(msgspec.Struct):
    planet_radius: float
    planet_gravity: float
    radius_units: str = "cm"
    gravity_units: str = "cm/s^2"
    additional_attributes: Optional[AttrsType] = None


def build_default_fundamental_parameters(
    fundamental_parameter_inputs: FundamentalParameterInputs,
) -> FundamentalParameters:
    planet_radius_in_cm = (
        fundamental_parameter_inputs.planet_radius
        * ureg(fundamental_parameter_inputs.radius_units).to("cm").magnitude
    )
    planet_gravity_in_cgs = (
        fundamental_parameter_inputs.planet_gravity
        * ureg(fundamental_parameter_inputs.gravity_units).to("cm/s^2").magnitude
    )

    additional_attributes: dict[str, Any] = (
        fundamental_parameter_inputs.additional_attributes
        if fundamental_parameter_inputs.additional_attributes is not None
        else {}
    )

    planet_radius_in_cm_as_xarray: xr.DataArray = xr.DataArray(
        data=planet_radius_in_cm,
        dims=tuple(),
        attrs={"units": "cm"},
    )
    planet_gravity_in_cgs_as_xarray: xr.DataArray = xr.DataArray(
        data=planet_gravity_in_cgs,
        dims=tuple(),
        attrs={"units": "cm/s^2"},
    )

    return xr.Dataset(
        data_vars={
            "planet_radius": planet_radius_in_cm_as_xarray,
            "planet_gravity": planet_gravity_in_cgs_as_xarray,
        },
        coords={},
        attrs=additional_attributes,
    )


class PressureProfileInputs(msgspec.Struct):
    shallowest_log10_pressure: float
    deepest_log10_pressure: float
    number_of_levels: int
    units: str = "bar"
    additional_attributes: Optional[AttrsType] = None


def build_pressure_profile_from_log_pressures(
    pressure_profile_inputs: PressureProfileInputs,
) -> PressureProfile:
    log_pressures_by_level: np.ndarray = np.linspace(
        pressure_profile_inputs.shallowest_log10_pressure,
        pressure_profile_inputs.deepest_log10_pressure,
        pressure_profile_inputs.number_of_levels,
    )

    pressures_by_level: np.ndarray = 10**log_pressures_by_level

    pressures_by_level_in_cgs: np.ndarray = (
        pressures_by_level * ureg(pressure_profile_inputs.units).to("bar").magnitude
    )

    log_pressures_by_level_in_cgs: np.ndarray = np.log10(pressures_by_level_in_cgs)

    additional_attributes: dict[str, Any] = (
        pressure_profile_inputs.additional_attributes
        if pressure_profile_inputs.additional_attributes is not None
        else {}
    )

    pressures_by_level_as_xarray: xr.DataArray = xr.DataArray(
        name="pressure",
        data=jnp.array(pressures_by_level_in_cgs),
        dims=("pressure",),
        attrs={"units": "barye"},
    )

    log_pressures_by_level_as_xarray: xr.DataArray = xr.DataArray(
        name="log10_pressure",
        data=jnp.array(log_pressures_by_level_in_cgs),
        dims=("pressure",),
        coords={"pressure": pressures_by_level_as_xarray},
        attrs={"units": "log10(barye)"},
    )

    return xr.Dataset(
        data_vars={"log_pressures_by_level": log_pressures_by_level_as_xarray},
        coords={"pressure": pressures_by_level_as_xarray},
        attrs=additional_attributes,
    )


class UniformGasChemistryInputs(msgspec.Struct):
    log_mixing_ratios: UniformLogMixingRatios
    filler_species: Optional[str] = None
    additional_attributes: Optional[AttrsType] = None


def build_uniform_gas_chemistry(
    gas_chemistry_inputs: UniformGasChemistryInputs, pressure_profile: PressureProfile
) -> GasChemistryProfile:
    log_mixing_ratios: UniformLogMixingRatios = gas_chemistry_inputs.log_mixing_ratios
    filler_species: str = gas_chemistry_inputs.filler_species
    additional_attributes: dict[str, Any] = (
        gas_chemistry_inputs.additional_attributes
        if gas_chemistry_inputs.additional_attributes is not None
        else {}
    )

    number_of_pressure_levels: int = len(pressure_profile.pressure)

    uniform_mixing_ratios_by_level: dict[str, np.ndarray[np.float64]] = (
        generate_uniform_mixing_ratios(
            uniform_log_abundances=log_mixing_ratios,
            number_of_pressure_levels=number_of_pressure_levels,
            filler_species=filler_species,
        )
    )

    mixing_ratios_by_level_as_xarray: xr.Dataset = xr.Dataset(
        data_vars={
            mixing_ratio_name: xr.DataArray(
                data=uniform_mixing_ratios_by_level[mixing_ratio_name],
                dims=("pressure",),
                attrs={"units": "mol/mol"},
            )
            for mixing_ratio_name in uniform_mixing_ratios_by_level
        },
        coords=pressure_profile.coords,
        attrs=additional_attributes,
        # attrs={
        #     "model_ID": model_id,
        #     "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # },
    )

    return mixing_ratios_by_level_as_xarray


def build_temperature_profile(
    temperature_model_constructor: TemperatureModelConstructor,
    temperature_model_inputs: TemperatureModelInputs,
    pressure_profile: PressureProfile,
) -> TemperatureProfile:
    temperature_model: TemperatureModel = temperature_model_constructor(
        **temperature_model_inputs
    )

    temperature_model_for_xarray: TemperatureModel = rename_and_unitize(
        new_name="temperature", units="K"
    )(
        Dimensionalize(
            argument_dimensions=((PressureDimension,),),
            result_dimensions=((PressureDimension,),),
        )(temperature_model),
    )

    temperatures_by_level_as_xarray: xr.DataArray = temperature_model_for_xarray(
        pressure_profile.log_pressures_by_level
    )

    return temperatures_by_level_as_xarray


def compile_vertical_structure(
    fundamental_parameters: xr.Dataset,
    pressure_profile: xr.Dataset,
    temperature_profile: xr.DataArray,
    gas_chemistry: xr.Dataset,
) -> xr.DataTree:
    default_vertical_structure_as_xarray: xr.Dataset = xr.Dataset(
        data_vars={
            "planet_radius_in_cm": fundamental_parameters.planet_radius,
            "planet_gravity_in_cgs": fundamental_parameters.planet_gravity,
            "pressures_by_level": pressure_profile.pressure,
            "log_pressures_by_level": pressure_profile.log_pressures_by_level,
            "temperatures_by_level": temperature_profile,
        },
        coords={
            "pressure": pressure_profile.pressure,
        },
        attrs={
            # "model_ID": model_id,
            # "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    chemistry_node: xr.DataTree = xr.DataTree(name="chemistry", dataset=gas_chemistry)

    default_vertical_structure_datatree: xr.DataTree = xr.DataTree(
        name="vertical_structure",
        dataset=default_vertical_structure_as_xarray,
        children={"chemistry": chemistry_node},
    )

    return default_vertical_structure_datatree


def build_forward_model(
    vertical_structure: xr.DataTree,
    crosssection_catalog_dataset: xr.Dataset,
    output_wavelengths: xr.DataArray,
    distance_to_system_in_cm: PositiveValue,
    stellar_radius_in_cm: PositiveValue,
) -> xr.DataTree:
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
    ).to_dataset()

    altitudes_by_layer: xr.DataArray = altitudes_by_level_to_by_layer(
        altitudes_in_cm
    ).to_dataset()

    path_length_node: xr.DataTree = xr.DataTree(
        name="path_lengths_by_layer", dataset=path_lengths_by_layer
    )

    altitude_node: xr.DataTree = xr.DataTree(
        name="altitudes_by_layer", dataset=altitudes_by_layer
    )

    distance_to_system_in_cm_as_xarray: xr.Dataset = xr.DataArray(
        data=distance_to_system_in_cm,
        dims=tuple(),
        attrs={"units": "cm"},
        name="distance_to_system_in_cm",
    ).to_dataset()
    distance_node: xr.DataTree = xr.DataTree(
        name="distance_to_system_in_cm", dataset=distance_to_system_in_cm_as_xarray
    )

    stellar_radius_in_cm_as_xarray: xr.Dataset = xr.DataArray(
        data=stellar_radius_in_cm,
        dims=tuple(),
        attrs={"units": "cm"},
        name="stellar_radius_in_cm",
    ).to_dataset()
    stellar_radius_node: xr.DataTree = xr.DataTree(
        name="stellar_radius_in_cm", dataset=stellar_radius_in_cm_as_xarray
    )

    number_densities_by_level: xr.Dataset = xr.Dataset(
        data_vars={
            species: xr.DataArray(
                data=number_density_array, coords=vertical_structure.coords
            )
            for species, number_density_array in mixing_ratios_to_number_densities(
                mixing_ratios_by_level=vertical_structure.chemistry,
                pressure_in_cgs=vertical_structure.pressures_by_level,  #  * BAR_TO_BARYE,
                temperatures_in_K=vertical_structure.temperatures_by_level,
            ).items()
        },
        coords=vertical_structure.coords,
        attrs={"units": "cm^-3"},
    )
    number_density_node: xr.DataTree = xr.DataTree(
        name="number_densities_by_level", dataset=number_densities_by_level
    )

    vertical_structure_by_level: xr.DataTree = vertical_structure.assign(
        items={
            "number_densities_by_level": number_density_node,
        }
    )
    print(f"{vertical_structure_by_level=}")

    vertical_structure_by_layer: xr.DataTree = (
        convert_datatree_by_pressure_levels_to_pressure_layers(
            vertical_structure_by_level
        )
    )

    # vertical_structure_by_layer.children["chemistry"].children[
    #    "number_densities_by_level"
    # ] = number_densities_by_level

    # vertical_structure_by_level.children["chemistry"].children[
    #    "number_densities_by_level"
    # ] = number_densities_by_level

    return (
        vertical_structure_by_layer.assign(
            items={
                "path_lengths_by_layer": path_length_node,
                "altitudes_by_layer": altitude_node,
                "distance_to_system_in_cm": distance_node,
                "stellar_radius_in_cm": stellar_radius_node,
            }
        ),
        crosssection_catalog_dataset,
        output_wavelengths,
    )


if __name__ == "__main__":
    test_model_inputs_filepath: Path = current_directory / "test_model_inputs.toml"
    test_model_inputs_as_py_filepath: Path = current_directory / "test_model_inputs.py"

    with open(test_model_inputs_filepath, "rb") as file:
        test_model_inputs: FundamentalParameterInputs = msgspec.toml.decode(
            file.read()  # , type=FundamentalParameterInputs
        )

    for (
        model_component_function_name,
        model_component_parameter_set,
    ) in test_model_inputs.items():
        model_component_function: Callable = getattr(
            __import__(__name__), model_component_function_name
        )

        # get the type signature of the only argument of this function
        model_component_function_signature = inspect.signature(
            model_component_function
        ).parameters

        model_component_function_arguments_class: type = list(
            model_component_function_signature.values()
        )[0].annotation

        model_component_function_arguments = model_component_function_arguments_class(
            **model_component_parameter_set
        )
        print(f"{model_component_function_arguments=}")

        test_output = model_component_function(model_component_function_arguments)

        print(f"{test_output=}")

    test_model_inputs = import_module("test_model_inputs")

    test_vertical_structure: xr.DataTree = compile_vertical_structure(
        fundamental_parameters=test_model_inputs.fundamental_parameters,
        pressure_profile=test_model_inputs.pressure_profile,
        temperature_profile=test_model_inputs.temperature_profile,
        gas_chemistry=test_model_inputs.gas_chemistry,
    )

    opacity_catalog: str = "jwst50k"  # "jwst50k", "wide-jwst", "wide"

    # opacities_directory: Path = Path("/Volumes/Orange") / "Opacities_0v10"
    opacities_directory: Path = Path("/media/gba8kj/Orange") / "Opacities_0v10"
    opacity_data_directory: Path = opacities_directory / "gases"

    catalog_filepath: Path = opacity_data_directory / f"{opacity_catalog}.nc"
    crosssection_catalog_dataset: xr.Dataset = xr.open_dataset(catalog_filepath)

    reference_model_filepath: Path = (
        current_directory / "2M2236b_NIRSpec_G395H_R500_APOLLO.nc"
    )
    reference_model: xr.Dataset = xr.open_dataset(reference_model_filepath)
    output_wavelengths: xr.DataArray = reference_model.wavelength.to_dataset(
        name="output_wavelengths"
    )

    distance_to_system_in_cm: float = 63.0 * PARSEC_TO_CM  # 64.5
    stellar_radius_in_cm: float = 1.53054e10

    test_forward_model_structure: xr.DataTree = build_forward_model(
        vertical_structure=test_vertical_structure,
        crosssection_catalog_dataset=crosssection_catalog_dataset,
        output_wavelengths=output_wavelengths,
        distance_to_system_in_cm=distance_to_system_in_cm,
        stellar_radius_in_cm=stellar_radius_in_cm,
    )

    print(f"{test_forward_model_structure=}")
