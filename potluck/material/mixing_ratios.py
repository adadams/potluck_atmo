from collections.abc import Mapping
from typing import Final, Optional, Protocol, TypeAlias, Union

import msgspec
import numpy as np

from potluck.basic_types import (
    LogMixingRatioValue,
    LogPressureValue,
    MixingRatioValue,
    NormalizedValue,
)

UniformLogMixingRatios: TypeAlias = Mapping[str, LogMixingRatioValue]
LogMixingRatios: TypeAlias = Union[
    UniformLogMixingRatios, Mapping[str, np.ndarray[LogMixingRatioValue]]
]
MixingRatios: TypeAlias = Union[
    Mapping[str, MixingRatioValue], Mapping[str, np.ndarray[MixingRatioValue]]
]


class TwoLevelGasChemistryInput(msgspec.Struct):
    shallow_log_mixing_ratio: LogMixingRatioValue
    deep_log_mixing_ratio: LogMixingRatioValue
    boundary_log_pressure: LogPressureValue


TwoLevelGasChemistryInputs: TypeAlias = Mapping[str, TwoLevelGasChemistryInput]


class MixingRatioPipeline(Protocol):
    def __call__(self, *args, **kwargs) -> MixingRatios: ...


SMALL_VALUE: Final[float] = 1e-6


def calculate_single_filler_species_mixing_ratios_by_level(
    existing_mixing_ratios: MixingRatios,
) -> MixingRatios:
    return 1 - np.sum(np.stack(list(existing_mixing_ratios.values())), axis=0)


def add_filler_to_mixing_ratios(
    mixing_ratios: MixingRatios, filler_species: str = "h2"
) -> MixingRatios:
    return mixing_ratios | {filler_species: 1 - np.sum(list(mixing_ratios.values()))}


def add_filler_to_mixing_ratios_by_level(
    mixing_ratios: MixingRatios, filler_species: str = "h2"
) -> MixingRatios:
    return mixing_ratios | {
        filler_species: 1 - np.sum(np.stack(list(mixing_ratios.values())), axis=0)
    }


def convert_log_abundances_to_mixing_ratios(
    log_abundances: LogMixingRatios, filler_species: str = "h2"
) -> dict[str, float]:
    abundances_without_filler: dict[str, float] = {
        species: 10 ** log_abundances[species] for species in log_abundances
    }

    return (
        abundances_without_filler
        if filler_species is None
        else add_filler_to_mixing_ratios(abundances_without_filler, filler_species)
    )


def convert_uniform_values_to_values_by_level(
    uniform_values: UniformLogMixingRatios, number_of_pressure_levels: int
) -> LogMixingRatios:
    return {
        species: np.full(number_of_pressure_levels, value)
        for species, value in uniform_values.items()
    }


def generate_uniform_mixing_ratios(
    uniform_log_abundances: LogMixingRatioValue,
    number_of_pressure_levels: int,
    filler_species: Optional[str] = "h2",
) -> MixingRatios:
    return convert_uniform_values_to_values_by_level(
        uniform_values=convert_log_abundances_to_mixing_ratios(
            log_abundances=uniform_log_abundances, filler_species=filler_species
        ),
        number_of_pressure_levels=number_of_pressure_levels,
    )


def convert_boundary_layer_to_level_index(
    boundary_log_pressure: LogPressureValue,
    log10_pressures_by_level: np.ndarray[LogPressureValue],
) -> float:
    integer_part_of_boundary_level_index: int = max(
        np.searchsorted(log10_pressures_by_level, boundary_log_pressure).item() - 1, 0
    )

    shallow_side_boundary_layer_log_pressure: LogPressureValue = (
        log10_pressures_by_level[integer_part_of_boundary_level_index]
    )
    deep_side_boundary_layer_log_pressure: LogPressureValue = log10_pressures_by_level[
        integer_part_of_boundary_level_index + 1
    ]

    # this should only over be exactly one if the bottom of the layer containing the boundary
    # is at the bottom of the model atmosphere, so we nudge it back slightly in that case
    fractional_part_of_boundary_level_index: NormalizedValue = np.clip(
        (boundary_log_pressure - shallow_side_boundary_layer_log_pressure)
        / (
            deep_side_boundary_layer_log_pressure
            - shallow_side_boundary_layer_log_pressure
        ),
        0,
        1 - SMALL_VALUE,
    )

    return (
        integer_part_of_boundary_level_index + fractional_part_of_boundary_level_index
    )


def calculate_two_level_mixing_ratios_by_species(
    shallow_log_mixing_ratio: LogMixingRatioValue,
    deep_log_mixing_ratio: LogMixingRatioValue,
    boundary_log_pressure: LogPressureValue,
    log10_pressures_by_level: np.ndarray[LogPressureValue],
) -> np.ndarray[LogMixingRatioValue]:
    boundary_level_index: float = convert_boundary_layer_to_level_index(
        boundary_log_pressure, log10_pressures_by_level
    )

    number_of_pressure_levels: int = len(log10_pressures_by_level)

    boundary_layer_nearest_index: int = int(boundary_level_index)

    shallow_mixing_ratio: MixingRatioValue = 10**shallow_log_mixing_ratio
    deep_mixing_ratio: MixingRatioValue = 10**deep_log_mixing_ratio

    number_of_levels_with_shallow_mixing_ratio: int = boundary_layer_nearest_index + 1
    number_of_levels_with_deep_mixing_ratio: int = (
        number_of_pressure_levels - number_of_levels_with_shallow_mixing_ratio
    )

    return np.r_[
        np.full(number_of_levels_with_shallow_mixing_ratio, shallow_mixing_ratio),
        np.full(number_of_levels_with_deep_mixing_ratio, deep_mixing_ratio),
    ]


def convert_two_level_mixing_ratios_by_species_from_levels_to_layers(
    two_level_gas_mixing_ratios_by_layer: np.ndarray[MixingRatioValue],
    shallow_log_mixing_ratio: LogMixingRatioValue,
    deep_log_mixing_ratio: LogMixingRatioValue,
    boundary_log_pressure: LogPressureValue,
    log10_pressures_by_level: np.ndarray[LogPressureValue],
) -> np.ndarray[MixingRatioValue]:
    boundary_level_index: float = convert_boundary_layer_to_level_index(
        boundary_log_pressure, log10_pressures_by_level
    )

    boundary_layer_nearest_index: int = int(boundary_level_index)

    fraction_of_boundary_layer_with_deep_mixing_ratio: float = (
        boundary_level_index - boundary_layer_nearest_index
    )
    fraction_of_boundary_layer_with_shallow_mixing_ratio: float = (
        1 - fraction_of_boundary_layer_with_deep_mixing_ratio
    )

    intermediate_log_mixing_ratio: LogMixingRatioValue = (
        shallow_log_mixing_ratio * fraction_of_boundary_layer_with_shallow_mixing_ratio
        + deep_log_mixing_ratio * fraction_of_boundary_layer_with_deep_mixing_ratio
    )
    intermediate_mixing_ratio: MixingRatioValue = 10**intermediate_log_mixing_ratio

    np.put(
        two_level_gas_mixing_ratios_by_layer,
        boundary_layer_nearest_index,
        intermediate_mixing_ratio,
    )

    return two_level_gas_mixing_ratios_by_layer


def generate_two_level_mixing_ratios_by_level(
    two_level_gas_inputs: TwoLevelGasChemistryInputs,
    log10_pressures_by_level: np.ndarray[LogPressureValue],
) -> MixingRatios:
    return {
        species: calculate_two_level_mixing_ratios_by_species(
            shallow_log_mixing_ratio=two_level_gas_input.shallow_log_mixing_ratio,
            deep_log_mixing_ratio=two_level_gas_input.deep_log_mixing_ratio,
            boundary_log_pressure=two_level_gas_input.boundary_log_pressure,
            log10_pressures_by_level=log10_pressures_by_level,
        )
        for species, two_level_gas_input in two_level_gas_inputs.items()
    }


def calculate_uniform_mixing_ratios_in_slab_single_layer(
    slab_top_level_fractional_index: float,
    slab_bottom_level_fractional_index: float,
    uniform_mixing_ratio: float,
    number_of_pressure_layers: int,
) -> np.ndarray[np.float64]:
    slab_level_index: int = int(slab_top_level_fractional_index)

    slab_mixing_ratio_fraction: np.ndarray = np.zeros(number_of_pressure_layers)

    slab_mixing_ratio_fraction[slab_level_index] = (
        slab_bottom_level_fractional_index - slab_top_level_fractional_index
    )

    return uniform_mixing_ratio * slab_mixing_ratio_fraction


def calculate_uniform_mixing_ratios_in_slab_multi_level(
    slab_top_level_index: int,
    slab_bottom_level_index: int,
    uniform_mixing_ratio: float,
    number_of_pressure_levels: int,
) -> np.ndarray[np.float64]:
    return (
        uniform_mixing_ratio
        * np.r_[
            np.zeros(slab_top_level_index + 1),
            np.ones(slab_bottom_level_index - slab_top_level_index - 1),
            np.zeros(number_of_pressure_levels - slab_bottom_level_index),
        ]
    )


def convert_uniform_mixing_ratios_in_slab_from_levels_to_layers(
    slab_top_level_fractional_index: float,
    slab_top_level_index: int,
    slab_bottom_level_fractional_index: float,
    slab_bottom_level_index: int,
    uniform_mixing_ratio: float,
) -> np.ndarray[np.float64]:
    slab_top_layer_filling_fraction: float = (
        slab_top_level_index + 1
    ) - slab_top_level_fractional_index

    slab_bottom_layer_filling_fraction: float = (
        slab_bottom_level_fractional_index - slab_bottom_level_index
    )

    np.put(
        uniform_mixing_ratio,
        [slab_top_level_index, slab_bottom_level_index],
        [slab_top_layer_filling_fraction, slab_bottom_layer_filling_fraction],
    )


def calculate_uniform_mixing_ratios_in_slab_multi_layer(
    slab_top_level_fractional_index: float,
    slab_top_level_index: int,
    slab_bottom_level_fractional_index: float,
    slab_bottom_level_index: int,
    uniform_mixing_ratio: float,
    number_of_pressure_layers: int,
) -> np.ndarray[np.float64]:
    slab_top_layer_filling_fraction: float = (
        slab_top_level_index + 1
    ) - slab_top_level_fractional_index
    slab_bottom_layer_filling_fraction: float = (
        slab_bottom_level_fractional_index - slab_bottom_level_index
    )

    return (
        uniform_mixing_ratio
        * np.r_[
            np.zeros(slab_top_level_index),
            slab_top_layer_filling_fraction,
            np.ones(slab_bottom_level_index - slab_top_level_index - 1),
            slab_bottom_layer_filling_fraction,
            np.zeros(number_of_pressure_layers - slab_bottom_level_index - 1),
        ]
    )


def convert_mass_mixing_ratios_to_volume_mixing_ratios(
    mass_mixing_ratios: MixingRatios, molecular_densities: dict[str, float]
) -> MixingRatios:
    return {
        species: mixing_ratio / molecular_densities[species]
        for species, mixing_ratio in mass_mixing_ratios.items()
    }
