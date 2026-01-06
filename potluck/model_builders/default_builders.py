from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Optional, TypeAlias

import msgspec
import numpy as np
import xarray as xr
from jax import numpy as jnp
from msgspec.structs import asdict
from pint import UnitRegistry

from potluck.basic_types import (
    LogMixingRatioValue,
    LogPressureValue,
    NonnegativeValue,
    NormalizedValue,
    PositiveValue,
    PressureDimension,
)
from potluck.calculate_RT import (
    calculate_emitted_fluxes_without_clouds_via_two_stream,
    calculate_observed_fluxes_with_clouds_via_two_stream,
    calculate_observed_fluxes_with_power_law_clouds_via_two_stream,
    calculate_observed_fluxes_without_clouds_via_two_stream,
    calculate_observed_transmission_spectrum,
)
from potluck.compile_crosssection_data import (
    curate_cloud_crosssection_catalog,
    curate_gas_crosssection_catalog,
)
from potluck.constants_and_conversions import AMU_IN_GRAMS
from potluck.density import calculate_mass_from_radius_and_surface_gravity
from potluck.material.absorbing.Hminus_functions import (
    calculate_h_minus_mixing_ratios,
    compile_anionic_opacity_crosssections,
    compute_h_minus_opacity_factors,
)
from potluck.material.clouds.cloud_metrics import (
    calculate_cloud_mixing_ratios_by_layer,
    convert_cloud_remaining_fraction_to_thickness,
)
from potluck.material.gases.molecular_metrics import (
    calculate_mean_molecular_weight,
    calculate_total_number_density,
    mixing_ratios_to_number_densities_by_species,
)
from potluck.material.mixing_ratios import (
    TwoLevelGasChemistryInputs,
    UniformLogMixingRatios,
    calculate_single_filler_species_mixing_ratios_by_level,
    convert_two_level_mixing_ratios_by_species_from_levels_to_layers,
    generate_two_level_mixing_ratios_by_level,
    generate_uniform_mixing_ratios,
)
from potluck.spectrum.bin import resample_spectral_quantity_to_new_wavelengths
from potluck.temperature.protocols import (
    TemperatureModel,
    TemperatureModelConstructor,
    TemperatureModelParameters,
)
from potluck.vertical.altitude import (
    altitudes_by_level_to_path_lengths,
    calculate_altitude_profile,
    convert_coordinate_by_level_to_by_layer,
    convert_datatree_by_pressure_levels_to_pressure_layers,
)
from potluck.xarray_functional_wrappers import (
    Dimensionalize,
    XarrayStructure,
    convert_units,
    set_dimensionless_quantity,
    set_result_name_and_units,
)
from potluck.xarray_serialization import AttrsType

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory.parent

DEFAULT_UNITS_SYSTEM: Final[str] = "cgs"
ureg: UnitRegistry = UnitRegistry(system=DEFAULT_UNITS_SYSTEM)
ureg.load_definitions(str(project_directory / "additional_units.txt"))


FundamentalParameters: TypeAlias = xr.Dataset
PressureProfile: TypeAlias = xr.Dataset
TemperatureProfile: TypeAlias = xr.Dataset
GasChemistryProfile: TypeAlias = xr.Dataset
CloudChemistryProfile: TypeAlias = xr.Dataset
CloudPrescribedProfile: TypeAlias = xr.Dataset


class DefaultFundamentalParameterInputs(msgspec.Struct):
    planet_radius: PositiveValue
    radius_units: str
    log10_planet_gravity: float
    gravity_units: str
    additional_attributes: Optional[AttrsType] = msgspec.field(default_factory=dict)


def build_default_fundamental_parameters(
    fundamental_parameter_inputs: DefaultFundamentalParameterInputs,
) -> FundamentalParameters:
    planet_radius_in_cm = (
        fundamental_parameter_inputs.planet_radius
        * ureg(fundamental_parameter_inputs.radius_units).to("cm").magnitude
    )
    planet_gravity_in_cgs = (
        10**fundamental_parameter_inputs.log10_planet_gravity
        * ureg(fundamental_parameter_inputs.gravity_units).to("cm/s^2").magnitude
    )

    additional_attributes: dict[str, Any] = (
        fundamental_parameter_inputs.additional_attributes
        if fundamental_parameter_inputs.additional_attributes is not None
        else {}
    )

    planet_radius_in_cm_as_xarray: xr.DataArray = xr.DataArray(
        data=planet_radius_in_cm,
        dims=set_dimensionless_quantity(),
        attrs={"units": "cm"},
    )
    planet_gravity_in_cgs_as_xarray: xr.DataArray = xr.DataArray(
        data=planet_gravity_in_cgs,
        dims=set_dimensionless_quantity(),
        attrs={"units": "cm/s^2"},
    )

    fundamental_parameter_dataset: xr.Dataset = xr.Dataset(
        data_vars={
            "planet_radius": planet_radius_in_cm_as_xarray,
            "planet_gravity": planet_gravity_in_cgs_as_xarray,
        },
        coords={},
        attrs=additional_attributes,
    )

    return fundamental_parameter_dataset


class EvenlyLogSpacedPressureProfileInputs(msgspec.Struct):
    shallowest_log10_pressure: float
    deepest_log10_pressure: float
    units: str
    number_of_levels: int
    additional_attributes: Optional[AttrsType] = msgspec.field(default_factory=dict)


@dataclass
class ArbitraryPressureProfileInputs:
    log10_pressures: np.ndarray[np.float64]
    units: str
    additional_attributes: Optional[AttrsType] = field(default_factory=dict)


def build_pressure_profile_from_arbitrary_log_pressures(
    arbitrary_pressure_profile_inputs: ArbitraryPressureProfileInputs,
) -> PressureProfile:
    log_pressures_by_level: np.ndarray = (
        arbitrary_pressure_profile_inputs.log10_pressures
    )

    pressures_by_level: np.ndarray = 10**log_pressures_by_level

    pressures_by_level_in_cgs: np.ndarray = (
        pressures_by_level
        * ureg(arbitrary_pressure_profile_inputs.units).to("barye").magnitude
    )

    log_pressures_by_level_in_cgs: np.ndarray = np.log10(pressures_by_level_in_cgs)

    additional_attributes: dict[str, Any] = (
        arbitrary_pressure_profile_inputs.additional_attributes
        if arbitrary_pressure_profile_inputs.additional_attributes is not None
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

    pressure_profile_dataset: xr.Dataset = xr.Dataset(
        data_vars={"log_pressures_by_level": log_pressures_by_level_as_xarray},
        coords={"pressure": pressures_by_level_as_xarray},
        attrs=additional_attributes,
    )

    return pressure_profile_dataset


def build_pressure_profile_from_log_pressures(
    pressure_profile_inputs: EvenlyLogSpacedPressureProfileInputs,
) -> PressureProfile:
    log_pressures_by_level: np.ndarray = np.linspace(
        pressure_profile_inputs.shallowest_log10_pressure,
        pressure_profile_inputs.deepest_log10_pressure,
        pressure_profile_inputs.number_of_levels,
    )

    pressures_by_level: np.ndarray = 10**log_pressures_by_level

    pressures_by_level_in_cgs: np.ndarray = (
        pressures_by_level * ureg(pressure_profile_inputs.units).to("barye").magnitude
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

    pressure_profile_dataset: xr.Dataset = xr.Dataset(
        data_vars={"log_pressures_by_level": log_pressures_by_level_as_xarray},
        coords={"pressure": pressures_by_level_as_xarray},
        attrs=additional_attributes,
    )

    return pressure_profile_dataset


class LDwarfFreeGasChemistryInputs(msgspec.Struct):
    uniform_log_mixing_ratios: UniformLogMixingRatios
    two_level_log_mixing_ratios: TwoLevelGasChemistryInputs
    filler_species: Optional[str] = None
    additional_attributes: Optional[AttrsType] = msgspec.field(default_factory=dict)


def build_LDwarf_gas_chemistry(
    gas_chemistry_inputs: LDwarfFreeGasChemistryInputs,
    pressure_profile: PressureProfile,
) -> GasChemistryProfile:
    filler_species: str = gas_chemistry_inputs.filler_species

    uniform_log_mixing_ratios: UniformLogMixingRatios = (
        gas_chemistry_inputs.uniform_log_mixing_ratios
    )

    number_of_pressure_levels: int = len(pressure_profile.pressure)

    uniform_mixing_ratios_by_level: dict[str, np.ndarray[np.float64]] = (
        generate_uniform_mixing_ratios(
            uniform_log_abundances=uniform_log_mixing_ratios,
            number_of_pressure_levels=number_of_pressure_levels,
            filler_species=None,  # wait to add filler until both kinds of gas prescriptions are included
        )
    )

    uniform_mixing_ratios_by_level_as_xarray: xr.Dataset = xr.Dataset(
        data_vars={
            mixing_ratio_name: xr.DataArray(
                data=uniform_mixing_ratio_by_layer,
                dims=("pressure",),
                attrs={"units": "mol/mol"},
            )
            for mixing_ratio_name, uniform_mixing_ratio_by_layer in uniform_mixing_ratios_by_level.items()
        },
        coords={"pressure": pressure_profile.pressure},
    )

    two_level_log_mixing_ratio_inputs: TwoLevelGasChemistryInputs = (
        gas_chemistry_inputs.two_level_log_mixing_ratios
    )

    two_level_mixing_ratios_by_level: dict[str, np.ndarray[np.float64]] = (
        generate_two_level_mixing_ratios_by_level(
            two_level_gas_inputs=two_level_log_mixing_ratio_inputs,
            log10_pressures_by_level=pressure_profile.log_pressures_by_level,
        )
    )

    two_level_mixing_ratios_by_level_as_xarray: xr.Dataset = xr.Dataset(
        data_vars={
            species_name: xr.DataArray(
                data=mixing_ratio_by_level,
                dims=("pressure",),
                attrs={
                    "units": "mol/mol",
                    **asdict(two_level_log_mixing_ratio_inputs[species_name]),
                },
            )
            for species_name, mixing_ratio_by_level in two_level_mixing_ratios_by_level.items()
        },
        coords={"pressure": pressure_profile.pressure},
    )

    additional_attributes: AttrsType = (
        gas_chemistry_inputs.additional_attributes
        if gas_chemistry_inputs.additional_attributes is not None
        else {}
    )

    mixing_ratios_by_level_as_xarray: xr.Dataset = xr.merge(
        [
            uniform_mixing_ratios_by_level_as_xarray,
            two_level_mixing_ratios_by_level_as_xarray,
        ],
    ).assign_attrs(additional_attributes)

    filler_mixing_ratios_by_level: np.ndarray = (
        calculate_single_filler_species_mixing_ratios_by_level(
            existing_mixing_ratios=uniform_mixing_ratios_by_level
            | two_level_mixing_ratios_by_level
        )
    )

    mixing_ratios_by_level_as_xarray[filler_species] = xr.DataArray(
        data=filler_mixing_ratios_by_level,
        dims=("pressure",),
        coords={"pressure": pressure_profile.pressure},
        attrs={"units": "mol/mol"},
    )

    return mixing_ratios_by_level_as_xarray


class UniformGasChemistryInputs(msgspec.Struct):
    uniform_log_mixing_ratios: UniformLogMixingRatios
    filler_species: Optional[str] = None
    additional_attributes: Optional[AttrsType] = msgspec.field(default_factory=dict)


def build_uniform_gas_chemistry(
    gas_chemistry_inputs: UniformGasChemistryInputs, pressure_profile: PressureProfile
) -> GasChemistryProfile:
    log_mixing_ratios: UniformLogMixingRatios = (
        gas_chemistry_inputs.uniform_log_mixing_ratios
    )
    filler_species: str = gas_chemistry_inputs.filler_species

    number_of_pressure_levels: int = len(pressure_profile.pressure)

    uniform_mixing_ratios_by_level: dict[str, np.ndarray[np.float64]] = (
        generate_uniform_mixing_ratios(
            uniform_log_abundances=log_mixing_ratios,
            number_of_pressure_levels=number_of_pressure_levels,
            filler_species=filler_species,
        )
    )

    additional_attributes: AttrsType = (
        gas_chemistry_inputs.additional_attributes
        if gas_chemistry_inputs.additional_attributes is not None
        else {}
    )

    mixing_ratios_by_level_as_xarray: xr.Dataset = xr.Dataset(
        data_vars={
            mixing_ratio_name: xr.DataArray(
                data=uniform_mixing_ratio_by_level,
                dims=("pressure",),
                attrs={"units": "mol/mol"},
            )
            for mixing_ratio_name, uniform_mixing_ratio_by_level in uniform_mixing_ratios_by_level.items()
        },
        coords=pressure_profile.coords,
        attrs=additional_attributes,
    )

    return mixing_ratios_by_level_as_xarray


class PowerLawSlabCloudInputs(msgspec.Struct):
    cloud_top_log10_pressure: LogPressureValue
    cloud_log10_thickness: NonnegativeValue
    pressure_units: str
    power_law_exponent: float
    single_scattering_albedo: NormalizedValue
    maximum_optical_depth_at_reference_wavelength: NonnegativeValue
    reference_wavelength_in_microns: NonnegativeValue = 1.00


class SingleLayerPowerLawSlabCloudInputs(msgspec.Struct):
    layer_1: PowerLawSlabCloudInputs


MultiplePowerLawSlabCloudInputs: TypeAlias = dict[str, PowerLawSlabCloudInputs]


class PowerLawSlabCloudSamples(msgspec.Struct):
    cloud_top_log10_pressure: LogPressureValue
    cloud_thickness_relative_to_remaining_pressures: NormalizedValue
    pressure_units: str
    power_law_exponent: float
    single_scattering_albedo: NormalizedValue
    maximum_optical_depth_at_reference_wavelength: NonnegativeValue
    reference_wavelength_in_microns: NonnegativeValue = 1.00


def create_power_law_cloud_inputs_from_samples(
    cloud_top_log10_pressure: LogPressureValue,
    cloud_thickness_relative_to_remaining_pressures: NormalizedValue,
    pressure_units: str,
    power_law_exponent: float,
    single_scattering_albedo: NormalizedValue,
    maximum_optical_depth_at_reference_wavelength: NonnegativeValue,
    reference_wavelength_in_microns: NonnegativeValue,
    maximum_log10_pressure: float,
) -> PowerLawSlabCloudInputs:
    cloud_log10_thickness: PositiveValue = (
        convert_cloud_remaining_fraction_to_thickness(
            cloud_top_log10_pressure=cloud_top_log10_pressure,
            cloud_remaining_fraction=cloud_thickness_relative_to_remaining_pressures,
            maximum_log10_pressure=maximum_log10_pressure,
        )
    )

    return PowerLawSlabCloudInputs(
        cloud_top_log10_pressure=cloud_top_log10_pressure,
        cloud_log10_thickness=cloud_log10_thickness,
        pressure_units=pressure_units,
        power_law_exponent=power_law_exponent,
        single_scattering_albedo=single_scattering_albedo,
        maximum_optical_depth_at_reference_wavelength=maximum_optical_depth_at_reference_wavelength,
        reference_wavelength_in_microns=reference_wavelength_in_microns,
    )


def calculate_power_law_cloud_opacity_at_reference_wavelength(
    log10_pressures_by_level: xr.DataArray,
    power_law_slab_cloud_inputs: PowerLawSlabCloudInputs,
) -> xr.DataArray:
    cloud_top_pressure: NonnegativeValue = (
        10**power_law_slab_cloud_inputs.cloud_top_log10_pressure
    )

    cloud_base_log10_pressure: LogPressureValue = (
        power_law_slab_cloud_inputs.cloud_top_log10_pressure
        + power_law_slab_cloud_inputs.cloud_log10_thickness
    )
    cloud_base_pressure: NonnegativeValue = 10**cloud_base_log10_pressure

    maximum_optical_depth_at_reference_wavelength: NonnegativeValue = (
        power_law_slab_cloud_inputs.maximum_optical_depth_at_reference_wavelength
    )

    cloudy_pressures: xr.DataArray = 10 ** (
        log10_pressures_by_level.where(
            (
                log10_pressures_by_level
                >= power_law_slab_cloud_inputs.cloud_top_log10_pressure
            )
            & (log10_pressures_by_level <= cloud_base_log10_pressure)
        )
    )

    cloud_optical_depth_at_reference_wavelength: xr.DataArray = (
        (
            maximum_optical_depth_at_reference_wavelength
            * (
                (cloudy_pressures**2 - cloud_top_pressure**2)
                / (cloud_base_pressure**2 - cloud_top_pressure**2)
            )
        )
        .fillna(0.0)
        .rename("cloud_optical_depth_at_reference_wavelength")
    ).assign_attrs(
        units="dimensionless",
        reference_wavelength_in_microns=power_law_slab_cloud_inputs.reference_wavelength_in_microns,
        power_law_exponent=power_law_slab_cloud_inputs.power_law_exponent,
        single_scattering_albedo=power_law_slab_cloud_inputs.single_scattering_albedo,
    )

    return cloud_optical_depth_at_reference_wavelength


def build_power_law_cloud_profiles(
    multiple_power_law_slab_cloud_inputs: MultiplePowerLawSlabCloudInputs,
    pressure_profile: PressureProfile,
) -> CloudPrescribedProfile:
    # in short, compile a list of profiles at the reference wavelength
    # then, at some point we will back out the attenutation and scattering
    # coefficients for the two-stream routine

    multiple_power_law_slab_cloud_inputs_as_dict: dict[str, SlabCloudInputs] = (
        (msgspec.structs.asdict(multiple_power_law_slab_cloud_inputs))
        if isinstance(multiple_power_law_slab_cloud_inputs, msgspec.Struct)
        else multiple_power_law_slab_cloud_inputs
    )

    cloud_mixing_ratios_by_level: xr.Dataset = xr.Dataset(
        data_vars={
            cloud_layer_name: calculate_power_law_cloud_opacity_at_reference_wavelength(
                log10_pressures_by_level=pressure_profile.log_pressures_by_level,
                power_law_slab_cloud_inputs=slab_cloud_inputs,
            )
            for cloud_layer_name, slab_cloud_inputs in multiple_power_law_slab_cloud_inputs_as_dict.items()
        },
        coords=pressure_profile.coords,
    )

    return cloud_mixing_ratios_by_level


class SlabCloudInputs(msgspec.Struct):
    uniform_log_mixing_ratio: LogMixingRatioValue
    cloud_top_log10_pressure: float
    cloud_log10_thickness: float
    pressure_units: str
    mean_particle_radius: float
    radius_units: str
    log10_particle_distribution_standard_deviation: float = 0.5
    additional_attributes: Optional[AttrsType] = msgspec.field(default_factory=dict)


MultipleSlabCloudInputs: TypeAlias = dict[str, SlabCloudInputs]


class Na2SSlabCloudInputs(msgspec.Struct):
    crystalline_Na2S_Mie: SlabCloudInputs


class Mg2SiO4SlabCloudInputs(msgspec.Struct):
    amorphous_forsterite_Mie: SlabCloudInputs


class SlabCloudSamples(msgspec.Struct):
    uniform_log_mixing_ratio: LogMixingRatioValue
    cloud_top_log10_pressure: float
    cloud_thickness_relative_to_remaining_pressures: NormalizedValue
    log10_mean_particle_radius: float
    log10_particle_distribution_standard_deviation: float = 0.5


def create_cloud_inputs_from_samples(
    uniform_log_mixing_ratio: LogMixingRatioValue,
    cloud_top_log10_pressure: float,
    cloud_thickness_relative_to_remaining_pressures: NormalizedValue,
    log10_mean_particle_radius: float,
    log10_particle_distribution_standard_deviation: float,
    maximum_log10_pressure: float,
) -> SlabCloudInputs:
    cloud_log10_thickness: PositiveValue = (
        convert_cloud_remaining_fraction_to_thickness(
            cloud_top_log10_pressure=cloud_top_log10_pressure,
            cloud_remaining_fraction=cloud_thickness_relative_to_remaining_pressures,
            maximum_log10_pressure=maximum_log10_pressure,
        )
    )

    return SlabCloudInputs(
        uniform_log_mixing_ratio=uniform_log_mixing_ratio,
        cloud_top_log10_pressure=cloud_top_log10_pressure,
        cloud_log10_thickness=cloud_log10_thickness,
        pressure_units="bar",
        mean_particle_radius=10**log10_mean_particle_radius,
        radius_units="cm",
        log10_particle_distribution_standard_deviation=log10_particle_distribution_standard_deviation,
    )


def build_uniform_slab_cloud_chemistry(
    multiple_slab_cloud_inputs: MultipleSlabCloudInputs,
    pressure_profile: PressureProfile,
) -> CloudChemistryProfile:
    multiple_slab_cloud_inputs_as_dict: dict[str, SlabCloudInputs] = (
        (msgspec.structs.asdict(multiple_slab_cloud_inputs))
        if isinstance(multiple_slab_cloud_inputs, msgspec.Struct)
        else multiple_slab_cloud_inputs
    )

    cloud_mixing_ratios_by_layer: xr.Dataset = xr.Dataset(
        data_vars={
            cloud_species_name: xr.DataArray(
                data=calculate_cloud_mixing_ratios_by_layer(
                    log10_uniform_cloud_mixing_ratio=slab_cloud_inputs.uniform_log_mixing_ratio,
                    cloud_top_log10_pressure=slab_cloud_inputs.cloud_top_log10_pressure,
                    cloud_log10_thickness=slab_cloud_inputs.cloud_log10_thickness,
                    log10_pressures_by_level=pressure_profile.log_pressures_by_level,
                ),
                dims=("pressure",),
                attrs={
                    "units": "mol/mol",
                    "mean_particle_radius": slab_cloud_inputs.mean_particle_radius,
                    "radius_units": slab_cloud_inputs.radius_units,
                    "log10_particle_distribution_standard_deviation": slab_cloud_inputs.log10_particle_distribution_standard_deviation,
                },
            )
            for cloud_species_name, slab_cloud_inputs in multiple_slab_cloud_inputs_as_dict.items()
        },
        coords={
            "pressure": convert_coordinate_by_level_to_by_layer(
                pressure_profile.pressure
            )
        },
    )

    return cloud_mixing_ratios_by_layer


def renormalize_chemistry_profile_with_clouds(
    gas_chemistry_profile: GasChemistryProfile,
    cloud_chemistry_profile: CloudChemistryProfile,
    total_number_densities_by_layer: xr.DataArray,
) -> GasChemistryProfile:
    renormalization_factor: float = 1 - (
        cloud_chemistry_profile.to_dataarray(dim="species").sum(dim="species")
        / total_number_densities_by_layer
    )

    renormalized_gas_chemistry_profile: GasChemistryProfile = (
        gas_chemistry_profile / renormalization_factor
    )

    return renormalized_gas_chemistry_profile


def build_temperature_profile(
    temperature_model_constructor: TemperatureModelConstructor,
    temperature_model_parameters: TemperatureModelParameters,
    pressure_profile: PressureProfile,
) -> TemperatureProfile:
    temperature_model: TemperatureModel = temperature_model_constructor(
        model_parameters=temperature_model_parameters
    )

    temperature_model_for_xarray: TemperatureModel = set_result_name_and_units(
        result_names="temperature", units="K"
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
    additional_attributes: Optional[AttrsType] = None,
) -> xr.DataTree:
    planet_mass_in_g: float = calculate_mass_from_radius_and_surface_gravity(
        fundamental_parameters.planet_radius,  # assumed to be pre-processed into cgs
        fundamental_parameters.planet_gravity,  # assumed to be pre-processed into cgs
    )

    total_number_density: np.ndarray[np.float64] = calculate_total_number_density(
        pressure_profile.pressure, temperature_profile
    )

    if "h2" in gas_chemistry.data_vars:
        h2_filler_species: str = "h2"
        h2_filler_fraction: float = 1.00

    elif "h2he" in gas_chemistry.data_vars:
        h2_filler_species: str = "h2he"
        h2_filler_fraction: float = 0.83

    else:
        raise NotImplementedError(
            "h2 or h2he are currently the only filler species implemented for H minus opacities."
        )

    hminus_mixing_ratios_by_level: xr.Dataset = calculate_h_minus_mixing_ratios(
        gas_chemistry,
        temperature_profile,
        total_number_density,
        h2_filler_species,
        h2_filler_fraction,
    )

    gas_chemistry_with_non_reference_table_species: xr.Dataset = xr.merge(
        [gas_chemistry, hminus_mixing_ratios_by_level]
    )

    mean_molecular_weight_in_g: float = (
        calculate_mean_molecular_weight(gas_chemistry_with_non_reference_table_species)
        * AMU_IN_GRAMS
    )

    altitudes_by_level_in_cm: xr.DataArray = calculate_altitude_profile(
        pressure_profile.log_pressures_by_level,
        temperature_profile,
        mean_molecular_weight_in_g,
        fundamental_parameters.planet_radius,
        planet_mass_in_g,
    ).assign_coords(pressure=pressure_profile.pressure)

    additional_attributes: AttrsType = (
        additional_attributes if additional_attributes is not None else {}
    )

    vertical_structure_by_level: xr.Dataset = xr.Dataset(
        data_vars={
            "planet_radius": fundamental_parameters.planet_radius,
            "planet_gravity": fundamental_parameters.planet_gravity,
            "pressures_by_level": pressure_profile.pressure,
            "log_pressures_by_level": pressure_profile.log_pressures_by_level,
            "temperatures_by_level": temperature_profile,
            "altitudes_by_level": altitudes_by_level_in_cm,
        },
        coords={
            "pressure": pressure_profile.pressure,
        },
        attrs=additional_attributes,
    )
    vertical_structure_by_level_node: xr.DataTree = xr.DataTree(
        name="vertical_structure", dataset=vertical_structure_by_level
    )

    mixing_ratios_by_level: xr.Dataset = gas_chemistry_with_non_reference_table_species

    mixing_ratios_by_level_node: xr.DataTree = xr.DataTree(
        name="mixing_ratios", dataset=mixing_ratios_by_level
    )

    atmospheric_structure_by_level: xr.DataTree = xr.DataTree(
        name="atmospheric_structure",
        children={
            "vertical_structure": vertical_structure_by_level_node,
            "gas_mixing_ratios": mixing_ratios_by_level_node,
        },
    )

    atmospheric_structure_by_layer: xr.DataTree = (
        convert_datatree_by_pressure_levels_to_pressure_layers(
            atmospheric_structure_by_level
        )
    )

    species_with_discrete_boundaries: list[str] = [
        species_name
        for species_name in gas_chemistry.data_vars
        if "boundary_log_pressure" in gas_chemistry[species_name].attrs
    ]

    if species_with_discrete_boundaries:
        for species in species_with_discrete_boundaries:
            species_mixing_ratios: xr.DataArray = atmospheric_structure_by_layer[
                "gas_mixing_ratios"
            ][species]

            species_boundary_log_pressure: float = species_mixing_ratios.attrs[
                "boundary_log_pressure"
            ]
            species_shallow_log_mixing_ratio: float = species_mixing_ratios.attrs[
                "shallow_log_mixing_ratio"
            ]
            species_deep_log_mixing_ratio: float = species_mixing_ratios.attrs[
                "deep_log_mixing_ratio"
            ]

            species_mixing_ratio_values: np.ndarray = species_mixing_ratios.values

            # using the values property means the operation is done in place on the underlying array,
            # so no need to reassign
            species_mixing_ratio_values: np.ndarray = (
                convert_two_level_mixing_ratios_by_species_from_levels_to_layers(
                    species_mixing_ratio_values,
                    species_shallow_log_mixing_ratio,
                    species_deep_log_mixing_ratio,
                    species_boundary_log_pressure,
                    pressure_profile.log_pressures_by_level,
                )
            )

    gas_number_densities_by_layer: xr.Dataset = xr.apply_ufunc(
        mixing_ratios_to_number_densities_by_species,
        atmospheric_structure_by_layer["gas_mixing_ratios"],
        atmospheric_structure_by_layer["vertical_structure"].pressures_by_layer,
        atmospheric_structure_by_layer["vertical_structure"].temperatures_by_layer,
        dask="parallelized",
    )

    for species in gas_chemistry.data_vars:
        gas_number_densities_by_layer[species].attrs = gas_chemistry[species].attrs
        gas_number_densities_by_layer[species].attrs["units"] = "cm^-3"

    gas_species_calculated_from_reference_data: list[str] = list(
        gas_chemistry.data_vars
    )
    other_gas_species: list[str] = list(hminus_mixing_ratios_by_level.data_vars)

    gas_number_densities_by_layer_node: xr.DataTree = xr.DataTree(
        name="gas_number_densities",
        dataset=gas_number_densities_by_layer.get(
            gas_species_calculated_from_reference_data
        ),
    )

    hminus_number_densities_by_layer_node: xr.DataTree = xr.DataTree(
        name="hminus_number_densities",
        dataset=gas_number_densities_by_layer.get(other_gas_species),
    )

    atmospheric_structure_by_layer: xr.DataTree = atmospheric_structure_by_layer.assign(
        {
            "gas_number_densities": gas_number_densities_by_layer_node,
            "hminus_number_densities": hminus_number_densities_by_layer_node,
        }
    )

    path_lengths_by_layer_in_cm: xr.DataArray = altitudes_by_level_to_path_lengths(
        altitudes_by_level_in_cm
    )

    atmospheric_structure_by_layer["vertical_structure"] = (
        atmospheric_structure_by_layer["vertical_structure"].assign(
            {"path_lengths": path_lengths_by_layer_in_cm}
        )
    )

    return atmospheric_structure_by_layer


def compile_vertical_structure_with_power_law_clouds(
    fundamental_parameters: xr.Dataset,
    pressure_profile: xr.Dataset,
    temperature_profile: xr.DataArray,
    gas_chemistry: xr.Dataset,
    cloud_profile: xr.Dataset,
    additional_attributes: Optional[AttrsType] = None,
) -> xr.DataTree:
    planet_mass_in_g: float = calculate_mass_from_radius_and_surface_gravity(
        fundamental_parameters.planet_radius,  # assumed to be pre-processed into cgs
        fundamental_parameters.planet_gravity,  # assumed to be pre-processed into cgs
    )

    total_number_density: np.ndarray[np.float64] = calculate_total_number_density(
        pressure_profile.pressure, temperature_profile
    )

    if "h2" in gas_chemistry.data_vars:
        h2_filler_species: str = "h2"
        h2_filler_fraction: float = 1.00

    elif "h2he" in gas_chemistry.data_vars:
        h2_filler_species: str = "h2he"
        h2_filler_fraction: float = 0.83

    else:
        raise NotImplementedError(
            "h2 or h2he are currently the only filler species implemented for H minus opacities."
        )

    hminus_mixing_ratios_by_level: xr.Dataset = calculate_h_minus_mixing_ratios(
        gas_chemistry,
        temperature_profile,
        total_number_density,
        h2_filler_species,
        h2_filler_fraction,
    )

    gas_chemistry_with_non_reference_table_species: xr.Dataset = xr.merge(
        [gas_chemistry, hminus_mixing_ratios_by_level]
    )

    mean_molecular_weight_in_g: float = (
        calculate_mean_molecular_weight(gas_chemistry_with_non_reference_table_species)
        * AMU_IN_GRAMS
    )

    altitudes_by_level_in_cm: xr.DataArray = calculate_altitude_profile(
        pressure_profile.log_pressures_by_level,
        temperature_profile,
        mean_molecular_weight_in_g,
        fundamental_parameters.planet_radius,
        planet_mass_in_g,
    ).assign_coords(pressure=pressure_profile.pressure)

    additional_attributes: AttrsType = (
        additional_attributes if additional_attributes is not None else {}
    )

    vertical_structure_by_level: xr.Dataset = xr.Dataset(
        data_vars={
            "planet_radius": fundamental_parameters.planet_radius,
            "planet_gravity": fundamental_parameters.planet_gravity,
            "pressures_by_level": pressure_profile.pressure,
            "log_pressures_by_level": pressure_profile.log_pressures_by_level,
            "temperatures_by_level": temperature_profile,
            "altitudes_by_level": altitudes_by_level_in_cm,
        },
        coords={
            "pressure": pressure_profile.pressure,
        },
        attrs=additional_attributes,
    )
    vertical_structure_by_level_node: xr.DataTree = xr.DataTree(
        name="vertical_structure", dataset=vertical_structure_by_level
    )

    mixing_ratios_by_level: xr.Dataset = gas_chemistry_with_non_reference_table_species

    mixing_ratios_by_level_node: xr.DataTree = xr.DataTree(
        name="mixing_ratios", dataset=mixing_ratios_by_level
    )

    cloud_reference_optical_depths_by_level: xr.Dataset = (
        build_power_law_cloud_profiles(
            multiple_power_law_slab_cloud_inputs=cloud_profile,
            pressure_profile=pressure_profile,
        )
    )

    cloud_reference_optical_depths_by_level_node: xr.DataTree = xr.DataTree(
        name="cloud_reference_optical_depths",
        dataset=cloud_reference_optical_depths_by_level,
    )

    atmospheric_structure_by_level: xr.DataTree = xr.DataTree(
        name="atmospheric_structure",
        children={
            "vertical_structure": vertical_structure_by_level_node,
            "gas_mixing_ratios": mixing_ratios_by_level_node,
            "cloud_reference_optical_depths": cloud_reference_optical_depths_by_level_node,
        },
    )

    atmospheric_structure_by_layer: xr.DataTree = (
        convert_datatree_by_pressure_levels_to_pressure_layers(
            atmospheric_structure_by_level
        )
    )

    species_with_discrete_boundaries: list[str] = [
        species_name
        for species_name in gas_chemistry.data_vars
        if "boundary_log_pressure" in gas_chemistry[species_name].attrs
    ]

    if species_with_discrete_boundaries:
        for species in species_with_discrete_boundaries:
            species_mixing_ratios: xr.DataArray = atmospheric_structure_by_layer[
                "gas_mixing_ratios"
            ][species]

            species_boundary_log_pressure: float = species_mixing_ratios.attrs[
                "boundary_log_pressure"
            ]
            species_shallow_log_mixing_ratio: float = species_mixing_ratios.attrs[
                "shallow_log_mixing_ratio"
            ]
            species_deep_log_mixing_ratio: float = species_mixing_ratios.attrs[
                "deep_log_mixing_ratio"
            ]

            species_mixing_ratio_values: np.ndarray = species_mixing_ratios.values

            # using the values property means the operation is done in place on the underlying array,
            # so no need to reassign
            species_mixing_ratio_values: np.ndarray = (
                convert_two_level_mixing_ratios_by_species_from_levels_to_layers(
                    species_mixing_ratio_values,
                    species_shallow_log_mixing_ratio,
                    species_deep_log_mixing_ratio,
                    species_boundary_log_pressure,
                    pressure_profile.log_pressures_by_level,
                )
            )

    gas_number_densities_by_layer: xr.Dataset = xr.apply_ufunc(
        mixing_ratios_to_number_densities_by_species,
        atmospheric_structure_by_layer["gas_mixing_ratios"],
        atmospheric_structure_by_layer["vertical_structure"].pressures_by_layer,
        atmospheric_structure_by_layer["vertical_structure"].temperatures_by_layer,
        dask="parallelized",
    )

    for species in gas_chemistry.data_vars:
        gas_number_densities_by_layer[species].attrs = gas_chemistry[species].attrs
        gas_number_densities_by_layer[species].attrs["units"] = "cm^-3"

    gas_species_calculated_from_reference_data: list[str] = list(
        gas_chemistry.data_vars
    )
    other_gas_species: list[str] = list(hminus_mixing_ratios_by_level.data_vars)

    gas_number_densities_by_layer_node: xr.DataTree = xr.DataTree(
        name="gas_number_densities",
        dataset=gas_number_densities_by_layer.get(
            gas_species_calculated_from_reference_data
        ),
    )

    hminus_number_densities_by_layer_node: xr.DataTree = xr.DataTree(
        name="hminus_number_densities",
        dataset=gas_number_densities_by_layer.get(other_gas_species),
    )

    atmospheric_structure_by_layer: xr.DataTree = atmospheric_structure_by_layer.assign(
        {
            "gas_number_densities": gas_number_densities_by_layer_node,
            "hminus_number_densities": hminus_number_densities_by_layer_node,
        }
    )

    path_lengths_by_layer_in_cm: xr.DataArray = altitudes_by_level_to_path_lengths(
        altitudes_by_level_in_cm
    )

    atmospheric_structure_by_layer["vertical_structure"] = (
        atmospheric_structure_by_layer["vertical_structure"].assign(
            {"path_lengths": path_lengths_by_layer_in_cm}
        )
    )

    return atmospheric_structure_by_layer


def compile_vertical_structure_with_clouds(
    fundamental_parameters: xr.Dataset,
    pressure_profile: xr.Dataset,
    temperature_profile: xr.DataArray,
    gas_chemistry: xr.Dataset,
    cloud_chemistry: xr.Dataset,
    additional_attributes: Optional[AttrsType] = None,
) -> xr.DataTree:
    planet_mass_in_g: float = calculate_mass_from_radius_and_surface_gravity(
        fundamental_parameters.planet_radius,  # assumed to be pre-processed into cgs
        fundamental_parameters.planet_gravity,  # assumed to be pre-processed into cgs
    )

    total_number_density: np.ndarray[np.float64] = calculate_total_number_density(
        pressure_profile.pressure, temperature_profile
    )

    if "h2" in gas_chemistry.data_vars:
        h2_filler_species: str = "h2"
        h2_filler_fraction: float = 1.00

    elif "h2he" in gas_chemistry.data_vars:
        h2_filler_species: str = "h2he"
        h2_filler_fraction: float = 0.83

    else:
        raise NotImplementedError(
            "h2 or h2he are currently the only filler species implemented for H minus opacities."
        )

    hminus_mixing_ratios_by_level: xr.Dataset = calculate_h_minus_mixing_ratios(
        gas_chemistry,
        temperature_profile,
        total_number_density,
        h2_filler_species,
        h2_filler_fraction,
    )

    gas_chemistry_with_non_reference_table_species: xr.Dataset = xr.merge(
        [gas_chemistry, hminus_mixing_ratios_by_level]
    )

    mean_molecular_weight_in_g: float = (
        calculate_mean_molecular_weight(gas_chemistry_with_non_reference_table_species)
        * AMU_IN_GRAMS
    )

    altitudes_by_level_in_cm: xr.DataArray = calculate_altitude_profile(
        pressure_profile.log_pressures_by_level,
        temperature_profile,
        mean_molecular_weight_in_g,
        fundamental_parameters.planet_radius,
        planet_mass_in_g,
    ).assign_coords(pressure=pressure_profile.pressure)

    additional_attributes: AttrsType = (
        additional_attributes if additional_attributes is not None else {}
    )

    vertical_structure_by_level: xr.Dataset = xr.Dataset(
        data_vars={
            "planet_radius": fundamental_parameters.planet_radius,
            "planet_gravity": fundamental_parameters.planet_gravity,
            "pressures_by_level": pressure_profile.pressure,
            "log_pressures_by_level": pressure_profile.log_pressures_by_level,
            "temperatures_by_level": temperature_profile,
            "altitudes_by_level": altitudes_by_level_in_cm,
        },
        coords={
            "pressure": pressure_profile.pressure,
        },
        attrs=additional_attributes,
    )
    vertical_structure_by_level_node: xr.DataTree = xr.DataTree(
        name="vertical_structure", dataset=vertical_structure_by_level
    )

    atmospheric_structure_by_level: xr.DataTree = xr.DataTree(
        name="atmospheric_structure",
        children={"vertical_structure": vertical_structure_by_level_node},
    )

    atmospheric_structure_by_layer: xr.DataTree = (
        convert_datatree_by_pressure_levels_to_pressure_layers(
            atmospheric_structure_by_level
        )
    )

    path_lengths_by_layer_in_cm: xr.DataArray = altitudes_by_level_to_path_lengths(
        altitudes_by_level_in_cm
    )

    total_number_densities_by_layer: xr.Dataset = calculate_total_number_density(
        pressure_in_cgs=atmospheric_structure_by_layer[
            "vertical_structure"
        ].pressures_by_layer,
        temperatures_in_K=atmospheric_structure_by_layer[
            "vertical_structure"
        ].temperatures_by_layer,
    )

    gas_number_densities_by_layer: xr.Dataset = (
        total_number_densities_by_layer * gas_chemistry_with_non_reference_table_species
    )

    for gas_species in gas_number_densities_by_layer.data_vars:
        gas_number_densities_by_layer[gas_species].attrs = gas_chemistry[
            gas_species
        ].attrs
        gas_number_densities_by_layer[gas_species].attrs["units"] = "cm^-3"

    gas_species_calculated_from_reference_data: list[str] = list(
        gas_chemistry.data_vars
    )
    other_gas_species: list[str] = list(hminus_mixing_ratios_by_level.data_vars)

    gas_number_densities_by_layer_node: xr.DataTree = xr.DataTree(
        name="gas_number_densities",
        dataset=gas_number_densities_by_layer.get(
            gas_species_calculated_from_reference_data
        ),
    )

    hminus_number_densities_by_layer_node: xr.DataTree = xr.DataTree(
        name="hminus_number_densities",
        dataset=gas_number_densities_by_layer.get(other_gas_species),
    )

    cloud_number_densities_by_layer: xr.Dataset = (
        total_number_densities_by_layer * cloud_chemistry
    )

    for cloud_species in cloud_number_densities_by_layer.data_vars.keys():
        cloud_number_densities_by_layer[cloud_species].attrs = cloud_chemistry[
            cloud_species
        ].attrs
        cloud_number_densities_by_layer[cloud_species].attrs["units"] = "cm^-3"

    cloud_number_densities_by_layer_node: xr.DataTree = xr.DataTree(
        name="cloud_number_densities", dataset=cloud_number_densities_by_layer
    )

    atmospheric_structure_by_layer: xr.DataTree = atmospheric_structure_by_layer.assign(
        {
            "gas_number_densities": gas_number_densities_by_layer_node,
            "hminus_number_densities": hminus_number_densities_by_layer_node,
            "cloud_number_densities": cloud_number_densities_by_layer_node,
        }
    )

    atmospheric_structure_by_layer["gas_number_densities"] = (
        renormalize_chemistry_profile_with_clouds(
            atmospheric_structure_by_layer["gas_number_densities"].to_dataset(),
            cloud_number_densities_by_layer,
            total_number_densities_by_layer,
        )
    )

    atmospheric_structure_by_layer["vertical_structure"] = (
        atmospheric_structure_by_layer["vertical_structure"].assign(
            {"path_lengths": path_lengths_by_layer_in_cm}
        )
    )

    return atmospheric_structure_by_layer


class DefaultObservableInputs(msgspec.Struct):
    distance_to_system: PositiveValue
    stellar_radius: PositiveValue
    distance_units: str
    radius_units: str


# @cache
def build_default_observable_inputs(
    observable_inputs: DefaultObservableInputs,
    output_wavelengths: xr.DataArray,
    additional_attributes: Optional[AttrsType] = None,
) -> xr.Dataset:
    distance_to_system_in_cm_as_xarray: xr.DataArray = xr.DataArray(
        data=observable_inputs.distance_to_system,
        dims=set_dimensionless_quantity(),
        attrs={"units": observable_inputs.distance_units},
        name="distance_to_system",
    )

    stellar_radius_in_cm_as_xarray: xr.DataArray = xr.DataArray(
        data=observable_inputs.stellar_radius,
        dims=set_dimensionless_quantity(),
        attrs={"units": observable_inputs.radius_units},
        name="stellar_radius",
    )

    observable_input_dataarrays: tuple[XarrayStructure, ...] = (
        convert_units(distance_to_system_in_cm_as_xarray, {"distance_to_system": "cm"}),
        convert_units(stellar_radius_in_cm_as_xarray, {"stellar_radius": "cm"}),
    )

    output_wavelengths_in_cm: xr.Variable = convert_units(
        output_wavelengths, {"wavelength": "cm"}
    )

    additional_attributes: AttrsType = (
        additional_attributes if additional_attributes is not None else {}
    )

    observable_input_dataset: xr.Dataset = (
        xr.merge(observable_input_dataarrays)
        .assign_coords(wavelength=xr.as_variable(output_wavelengths_in_cm))
        .assign_attrs(**additional_attributes)
    )

    return observable_input_dataset


# @cache
def build_observable_inputs_by_spectral_groups(
    observable_inputs: DefaultObservableInputs,
    spectral_groups: xr.DataArray,
    additional_attributes: Optional[AttrsType] = None,
) -> xr.Dataset:
    distance_to_system_in_cm_as_xarray: xr.DataArray = xr.DataArray(
        data=observable_inputs.distance_to_system,
        dims=set_dimensionless_quantity(),
        attrs={"units": observable_inputs.distance_units},
        name="distance_to_system",
    )

    stellar_radius_in_cm_as_xarray: xr.DataArray = xr.DataArray(
        data=observable_inputs.stellar_radius,
        dims=set_dimensionless_quantity(),
        attrs={"units": observable_inputs.radius_units},
        name="stellar_radius",
    )

    observable_input_dataarrays: tuple[XarrayStructure, ...] = (
        convert_units(distance_to_system_in_cm_as_xarray, {"distance_to_system": "cm"}),
        convert_units(stellar_radius_in_cm_as_xarray, {"stellar_radius": "cm"}),
    )

    spectral_groups_in_cm: xr.DataArray = convert_units(
        spectral_groups, {"wavelength": "cm"}
    )

    additional_attributes: AttrsType = (
        additional_attributes if additional_attributes is not None else {}
    )

    observable_input_dataset: xr.Dataset = (
        xr.merge(observable_input_dataarrays)
        .assign_coords(wavelength=xr.as_variable(spectral_groups_in_cm.wavelength))
        .assign_coords(
            spectral_group=xr.as_variable(spectral_groups_in_cm.spectral_group)
        )
        .assign_attrs(**additional_attributes)
    )

    return observable_input_dataset


def build_forward_model(
    atmospheric_structure_by_layer: xr.DataTree,
    temperature_profile: xr.DataArray,
    crosssection_catalog: xr.Dataset,
    observable_inputs: xr.Dataset,
    crosssection_catalog_ready_to_use: bool = False,  # i.e. it has been pre-interpolated, for example if the temperature profile is fixed
    filler_species: Optional[str] = "h2he",
    h2_filler_fraction: Optional[float] = 0.83,
) -> xr.DataTree:
    observable_parameter_node: xr.DataTree = xr.DataTree(
        name="observable_parameters", dataset=observable_inputs
    )

    crosssection_catalog_with_wavelengths_in_cm = convert_units(
        crosssection_catalog, {"wavelength": "cm", "pressure": "barye"}
    )

    # TODO: in principle, this check can be done automatically. If its pressure and temperature
    # coordinates match the pressure and temperature coordinates of the atmospheric structure,
    # then the crosssection catalog is ready to use.
    if crosssection_catalog_ready_to_use:
        crosssection_catalog_interpolated_to_model: xr.Dataset = (
            crosssection_catalog_with_wavelengths_in_cm
        )

    else:
        pressures_by_layer: xr.DataArray = (
            atmospheric_structure_by_layer.vertical_structure.pressures_by_layer
        )

        temperatures_by_layer: xr.DataArray = (
            atmospheric_structure_by_layer.vertical_structure.temperatures_by_layer
        )

        species_present_in_model: Iterable[str] = (
            atmospheric_structure_by_layer.gas_number_densities.data_vars.keys()
        )

        crosssection_catalog_interpolated_to_model: xr.Dataset = (
            curate_gas_crosssection_catalog(
                crosssection_catalog=crosssection_catalog_with_wavelengths_in_cm,
                temperatures_by_layer=temperatures_by_layer,
                pressures_by_layer=pressures_by_layer,
                species_present_in_model=species_present_in_model,
            )
        )

    if "hminus_number_densities" in atmospheric_structure_by_layer:
        h_minus_opacity_factors: xr.Dataset = compute_h_minus_opacity_factors(
            atmospheric_structure_by_layer["hminus_number_densities"].to_dataset(),
            atmospheric_structure_by_layer["gas_number_densities"].get(filler_species),
            h2_filler_fraction,
        )

        h_minus_opacity_crosssections: xr.Dataset = (
            compile_anionic_opacity_crosssections(
                h_minus_opacity_factors,
                temperatures_by_layer,
                crosssection_catalog.wavelength,
            )
        ).assign_coords(
            wavelength=crosssection_catalog_with_wavelengths_in_cm.wavelength
        )

        crosssection_catalog_interpolated_to_model[filler_species] += (
            h_minus_opacity_crosssections
        )

    crosssection_catalog_node: xr.DataTree = xr.DataTree(
        name="crosssection_catalog",
        dataset=crosssection_catalog_interpolated_to_model,
    )

    reference_data_node: xr.DataTree = xr.DataTree(
        name="reference_data",
        children={"gas_crosssection_catalog": crosssection_catalog_node},
    )

    temperature_profile_node: xr.DataTree = xr.DataTree(
        name="temperature_profile_by_level",
        dataset=temperature_profile.to_dataset(name="temperature"),
    )

    atmospheric_model: xr.DataTree = xr.DataTree(
        name="atmospheric_model",
        children={
            "atmospheric_structure_by_layer": atmospheric_structure_by_layer,
            "temperature_profile_by_level": temperature_profile_node,
            "observable_parameters": observable_parameter_node,
            "reference_data": reference_data_node,
        },
    )

    return atmospheric_model


def build_forward_model_with_clouds(
    atmospheric_structure_by_layer: xr.DataTree,
    temperature_profile: xr.DataArray,
    gas_crosssection_catalog: xr.Dataset,
    cloud_crosssection_catalog: xr.DataTree,
    observable_inputs: xr.Dataset,
) -> xr.DataTree:
    observable_parameter_node: xr.DataTree = xr.DataTree(
        name="observable_parameters", dataset=observable_inputs
    )

    pressures_by_layer: xr.DataArray = (
        atmospheric_structure_by_layer.vertical_structure.pressures_by_layer
    )

    temperatures_by_layer: xr.DataArray = (
        atmospheric_structure_by_layer.vertical_structure.temperatures_by_layer
    )

    gas_species_present_in_model: Iterable[str] = (
        atmospheric_structure_by_layer.gas_number_densities.data_vars.keys()
    )

    gas_crosssection_catalog_with_wavelengths_in_cm = convert_units(
        gas_crosssection_catalog, {"wavelength": "cm", "pressure": "barye"}
    )

    gas_crosssection_catalog_interpolated_to_model: xr.Dataset = (
        curate_gas_crosssection_catalog(
            crosssection_catalog=gas_crosssection_catalog_with_wavelengths_in_cm,
            temperatures_by_layer=temperatures_by_layer,
            pressures_by_layer=pressures_by_layer,
            species_present_in_model=gas_species_present_in_model,
        )
    )

    cloud_species_present_in_model: Iterable[str] = (
        atmospheric_structure_by_layer.cloud_number_densities.data_vars.keys()
    )

    for cloud_species in cloud_species_present_in_model:
        cloud_crosssection_catalog[cloud_species] = cloud_crosssection_catalog[
            cloud_species
        ].map_over_datasets(convert_units, {"wavelength": "cm"})

    cloud_crosssection_catalog_interpolated_to_model: xr.Dataset = (
        curate_cloud_crosssection_catalog(
            crosssection_catalog=cloud_crosssection_catalog,
            species_present_in_model=cloud_species_present_in_model,
        )
    )

    gas_crosssection_catalog_node: xr.DataTree = xr.DataTree(
        name="gas_crosssection_catalog",
        dataset=gas_crosssection_catalog_interpolated_to_model,
    )

    reference_data_node: xr.DataTree = xr.DataTree(
        name="reference_data",
        children={
            "gas_crosssection_catalog": gas_crosssection_catalog_node,
            "cloud_crosssection_catalog": cloud_crosssection_catalog_interpolated_to_model,
        },
    )

    temperature_profile_node: xr.DataTree = xr.DataTree(
        name="temperature_profile_by_level",
        dataset=temperature_profile.to_dataset(name="temperature"),
    )

    atmospheric_model: xr.DataTree = xr.DataTree(
        name="atmospheric_model",
        children={
            "atmospheric_structure_by_layer": atmospheric_structure_by_layer,
            "temperature_profile_by_level": temperature_profile_node,
            "observable_parameters": observable_parameter_node,
            "reference_data": reference_data_node,
        },
    )

    return atmospheric_model


def calculate_emission_at_surface_without_clouds(
    forward_model_inputs: xr.DataTree, resampling_fwhm_fraction: float
) -> float:
    emission_fluxes = calculate_emitted_fluxes_without_clouds_via_two_stream(
        forward_model_inputs=forward_model_inputs
    )

    reference_model_wavelengths: xr.DataArray = forward_model_inputs[
        "observable_parameters"
    ].wavelength

    emission_fluxes_sampled_to_data: xr.DataArray = (
        resample_spectral_quantity_to_new_wavelengths(
            reference_model_wavelengths,
            emission_fluxes.wavelength,
            emission_fluxes,
            fwhm=resampling_fwhm_fraction,
        )
    ).rename("resampled_emission_flux")

    return emission_fluxes_sampled_to_data


def calculate_emission_model_without_clouds(
    forward_model_inputs: xr.DataTree, resampling_fwhm_fraction: float
) -> np.ndarray:
    emission_fluxes = calculate_observed_fluxes_without_clouds_via_two_stream(
        forward_model_inputs=forward_model_inputs
    )

    reference_model_wavelengths: xr.DataArray = forward_model_inputs[
        "observable_parameters"
    ].wavelength

    emission_fluxes_sampled_to_data: xr.DataArray = (
        resample_spectral_quantity_to_new_wavelengths(
            reference_model_wavelengths,
            emission_fluxes.wavelength,
            emission_fluxes,
            fwhm=resampling_fwhm_fraction,
            # * (
            #    reference_model_wavelengths.to_numpy()[-1]
            #    - reference_model_wavelengths.to_numpy()[-2]
            # ),
        )
    ).rename("resampled_emission_flux")

    return emission_fluxes_sampled_to_data


def calculate_emission_model_with_clouds(
    forward_model_inputs: xr.DataTree, resampling_fwhm_fraction: float
) -> np.ndarray:
    emission_fluxes = calculate_observed_fluxes_with_clouds_via_two_stream(
        forward_model_inputs=forward_model_inputs
    )

    reference_model_wavelengths: xr.DataArray = forward_model_inputs[
        "observable_parameters"
    ].wavelength

    emission_fluxes_sampled_to_data: xr.DataArray = (
        resample_spectral_quantity_to_new_wavelengths(
            reference_model_wavelengths,
            emission_fluxes.wavelength,
            emission_fluxes,
            fwhm=resampling_fwhm_fraction,
        )
    ).rename("resampled_emission_flux")

    return emission_fluxes_sampled_to_data


def calculate_cloudy_emission_model_with_spectral_groups(
    forward_model_inputs: xr.DataTree, resampling_fwhm_fraction: float
) -> xr.DataArray:
    emission_fluxes = calculate_observed_fluxes_with_clouds_via_two_stream(
        forward_model_inputs=forward_model_inputs
    )

    observable_parameters: xr.Dataset = forward_model_inputs[
        "observable_parameters"
    ].to_dataset()

    reference_model_wavelengths: xr.DataArray = (
        observable_parameters.wavelength.groupby("spectral_group")
    )

    def resample_spectrum_per_group(
        spectral_group_wavelengths: xr.DataArray,
        model_wavelengths: xr.DataArray = emission_fluxes.wavelength,
        model_fluxes: xr.DataArray = emission_fluxes,
        resampling_fwhm_fraction: float = resampling_fwhm_fraction,
    ):
        return resample_spectral_quantity_to_new_wavelengths(
            spectral_group_wavelengths,
            model_wavelengths,
            model_fluxes,
            fwhm=resampling_fwhm_fraction,
        )

    emission_fluxes_sampled_to_data: xr.DataArray = (
        reference_model_wavelengths.map(resample_spectrum_per_group)
    ).rename("resampled_emission_flux")

    return emission_fluxes_sampled_to_data


def calculate_power_law_cloudy_emission_model_with_spectral_groups(
    forward_model_inputs: xr.DataTree, resampling_fwhm_fraction: float
) -> xr.DataArray:
    emission_fluxes = calculate_observed_fluxes_with_power_law_clouds_via_two_stream(
        forward_model_inputs=forward_model_inputs
    )

    observable_parameters: xr.Dataset = forward_model_inputs[
        "observable_parameters"
    ].to_dataset()

    reference_model_wavelengths: xr.DataArray = (
        observable_parameters.wavelength.groupby("spectral_group")
    )

    def resample_spectrum_per_group(
        spectral_group_wavelengths: xr.DataArray,
        model_wavelengths: xr.DataArray = emission_fluxes.wavelength,
        model_fluxes: xr.DataArray = emission_fluxes,
        resampling_fwhm_fraction: float = resampling_fwhm_fraction,
    ) -> xr.DataArray:
        return resample_spectral_quantity_to_new_wavelengths(
            spectral_group_wavelengths,
            model_wavelengths,
            model_fluxes,
            fwhm=resampling_fwhm_fraction,
        )

    emission_fluxes_sampled_to_data: xr.DataArray = (
        reference_model_wavelengths.map(resample_spectrum_per_group)
    ).rename("resampled_emission_flux")

    return emission_fluxes_sampled_to_data


def calculate_cloudfree_emission_model_with_spectral_groups(
    forward_model_inputs: xr.DataTree, resampling_fwhm_fraction: float
) -> xr.DataArray:
    emission_fluxes = calculate_observed_fluxes_without_clouds_via_two_stream(
        forward_model_inputs=forward_model_inputs
    )

    observable_parameters: xr.Dataset = forward_model_inputs[
        "observable_parameters"
    ].to_dataset()

    reference_model_wavelengths: xr.DataArray = (
        observable_parameters.wavelength.groupby("spectral_group")
    )

    def resample_spectrum_per_group(
        spectral_group_wavelengths: xr.DataArray,
        model_wavelengths: xr.DataArray = emission_fluxes.wavelength,
        model_fluxes: xr.DataArray = emission_fluxes,
        resampling_fwhm_fraction: float = resampling_fwhm_fraction,
    ):
        return resample_spectral_quantity_to_new_wavelengths(
            spectral_group_wavelengths,
            model_wavelengths,
            model_fluxes,
            fwhm=resampling_fwhm_fraction,
        )

    emission_fluxes_sampled_to_data: xr.DataArray = (
        reference_model_wavelengths.map(resample_spectrum_per_group)
    ).rename("resampled_emission_flux")

    return emission_fluxes_sampled_to_data


def calculate_transmission_model(
    forward_model_inputs: xr.DataTree, resampling_fwhm_fraction: float
) -> float:
    transit_depths: xr.DataArray = calculate_observed_transmission_spectrum(
        forward_model_inputs=forward_model_inputs
    )

    reference_model_wavelengths: xr.DataArray = forward_model_inputs[
        "observable_parameters"
    ].wavelength

    transit_depths_sampled_to_data: xr.DataArray = (
        resample_spectral_quantity_to_new_wavelengths(
            reference_model_wavelengths,
            transit_depths.wavelength,
            transit_depths,
            fwhm=resampling_fwhm_fraction,
        )
    ).rename("resampled_transmission_flux")

    return transit_depths_sampled_to_data
