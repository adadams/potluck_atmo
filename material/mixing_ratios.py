from collections.abc import Mapping
from typing import Protocol, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

MixingRatios: TypeAlias = Union[Mapping[str, float], Mapping[str, NDArray[np.float64]]]


def check_mixing_ratio_type(mixing_ratios: MixingRatios) -> bool:
    mixing_ratio_values: list[float] | list[NDArray[np.float64]] = list(
        mixing_ratios.values()
    )

    if isinstance(mixing_ratio_values[0], np.ndarray):
        return


class MixingRatioPipeline(Protocol):
    def __call__(self, *args, **kwargs) -> MixingRatios: ...


def add_filler_to_mixing_ratios(
    mixing_ratios: dict[str, float], filler_species: str = "h2"
) -> dict[str, float]:
    return mixing_ratios | {filler_species: 1 - np.sum(list(mixing_ratios.values()))}


def log_abundances_to_mixing_ratios(
    log_abundances: dict[str, float], filler_species: str = "h2"
) -> dict[str, float]:
    abundances_without_filler: dict[str, float] = {
        species: 10 ** log_abundances[species] for species in log_abundances
    }

    return add_filler_to_mixing_ratios(abundances_without_filler, filler_species)


def uniform_values_to_values_by_level(
    uniform_values: dict[str, float], number_of_pressure_levels: int
) -> dict[str, np.ndarray]:
    return {
        species: np.ones(number_of_pressure_levels) * value
        for species, value in uniform_values.items()
    }


def uniform_log_mixing_ratios(
    uniform_log_abundances: dict[str, float],
    number_of_pressure_levels: int,
    filler_species: str = "h2",
) -> dict[str, NDArray[np.float64]]:
    return uniform_values_to_values_by_level(
        uniform_values=log_abundances_to_mixing_ratios(
            log_abundances=uniform_log_abundances, filler_species=filler_species
        ),
        number_of_pressure_levels=number_of_pressure_levels,
    )
