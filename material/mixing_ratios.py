from collections.abc import Callable, Mapping
from typing import Protocol, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

from basic_functional_tools import compose

MixingRatios: TypeAlias = Union[Mapping[str, float], Mapping[str, NDArray[np.float64]]]


def check_mixing_ratio_type(mixing_ratios: MixingRatios) -> bool:
    mixing_ratio_values: list[float] | list[NDArray[np.float64]] = list(
        mixing_ratios.values()
    )

    if isinstance(mixing_ratio_values[0], np.ndarray):
        return


class MixingRatioPipeline(Protocol):
    def __call__(self, *args, **kwargs) -> MixingRatios: ...


def log_abundances_to_mixing_ratios(
    log_abundances: dict[str, float], filler_species: str = "h2"
) -> dict[str, float]:
    abundances_without_filler: dict[str, float] = {
        species: 10 ** log_abundances[species] for species in log_abundances
    }

    return abundances_without_filler | {
        filler_species: 1 - np.sum(list(abundances_without_filler.values()))
    }


def uniform_log_abundances_to_log_abundances_by_level(
    uniform_log_abundances: dict[str, float], number_of_pressure_levels: int
) -> dict[str, np.ndarray]:
    return {
        species: np.ones(number_of_pressure_levels) * log_abundance
        for species, log_abundance in uniform_log_abundances.items()
    }


uniform_log_mixing_ratios: Callable[
    [dict[str, float]], dict[str, NDArray[np.float64]]
] = compose(
    uniform_log_abundances_to_log_abundances_by_level, log_abundances_to_mixing_ratios
)
