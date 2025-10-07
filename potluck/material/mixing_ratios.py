from collections.abc import Mapping
from typing import Optional, Protocol, TypeAlias, Union

import numpy as np

from potluck.basic_types import LogMixingRatioValue, MixingRatioValue

UniformLogMixingRatios: TypeAlias = Mapping[str, LogMixingRatioValue]
LogMixingRatios: TypeAlias = Union[
    UniformLogMixingRatios, Mapping[str, np.ndarray[LogMixingRatioValue]]
]
MixingRatios: TypeAlias = Union[
    Mapping[str, MixingRatioValue], Mapping[str, np.ndarray[MixingRatioValue]]
]


class MixingRatioPipeline(Protocol):
    def __call__(self, *args, **kwargs) -> MixingRatios: ...


def add_filler_to_mixing_ratios(
    mixing_ratios: MixingRatios, filler_species: str = "h2"
) -> MixingRatios:
    return mixing_ratios | {filler_species: 1 - np.sum(list(mixing_ratios.values()))}


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
