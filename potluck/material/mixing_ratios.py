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
