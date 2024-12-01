from typing import Final

import msgspec
import numpy as np
from numpy.typing import NDArray

from constants_and_conversions import BOLTZMANN_CONSTANT_IN_CGS

SOLAR_METAL_FRACTION: Final[float] = 0.0196

NONMETAL_MOLECULAR_WEIGHTS: Final[dict[str, float]] = {
    "h": 1.0,
    "h2": 2.0,
    "he": 4.0,
}

METAL_MOLECULAR_WEIGHTS: Final[dict[str, float]] = {
    "h2o": 16.0,
    "ch4": 12.0,
    "co": 28.0,
    "co2": 44.0,
    "nh3": 14.0,
    "h2s": 32.0,
    "Burrows_alk": 24.0,
    "Lupu_alk": 24.0,
    "na": 23.0,
    "k": 39.0,
    "crh": 52.0,
    "feh": 56.0,
    "tio": 64.0,
    "vo": 67.0,
    "hcn": 26.0,
    "n2": 28.0,
    "ph3": 31.0,
}

MOLECULAR_WEIGHTS: Final[dict[str, float]] = (
    METAL_MOLECULAR_WEIGHTS | NONMETAL_MOLECULAR_WEIGHTS
)

CARBON_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {
    "ch4": 1,
    "co": 1,
    "co2": 1,
    "hcn": 1,
}

OXYGEN_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {
    "h2o": 1,
    "co": 1,
    "co2": 2,
    "tio": 1,
    "vo": 1,
}


class MoleculeMetrics(msgspec.Struct):
    CtoO_ratio: float | NDArray[np.float64]
    mean_molecular_weight: float | NDArray[np.float64]
    metallicity: float | NDArray[np.float64]


def calculate_CtoO_ratio(
    mixing_ratios: dict[str, float | NDArray[np.float64]],
) -> float | NDArray[np.float64]:
    carbon = 0.0
    oxygen = 0.0

    carbon_compounds = CARBON_ATOMS_PER_MOLECULE.keys()
    oxygen_compounds = OXYGEN_ATOMS_PER_MOLECULE.keys()

    for gas_species, mixing_ratio in mixing_ratios.items():
        if gas_species in carbon_compounds:
            carbon = carbon + (CARBON_ATOMS_PER_MOLECULE[gas_species] * mixing_ratio)

        if gas_species in oxygen_compounds:
            oxygen = oxygen + (OXYGEN_ATOMS_PER_MOLECULE[gas_species] * mixing_ratio)

    return carbon / oxygen


def calculate_metallicity(
    mixing_ratios: dict[str, float | NDArray[np.float64]],
    reference_metal_fraction: float = SOLAR_METAL_FRACTION,
) -> float | NDArray[np.float64]:
    metal_compounds = METAL_MOLECULAR_WEIGHTS.keys()

    metals = 0.0

    for gas_species, mixing_ratio in mixing_ratios.items():
        if gas_species in metal_compounds:
            metals = metals + (METAL_MOLECULAR_WEIGHTS[gas_species] * mixing_ratio)

    return np.log10(metals / reference_metal_fraction)


def calculate_weighted_molecular_weights(
    mixing_ratios: dict[str, float | NDArray[np.float64]],
    molecular_weights: dict[str, float] = MOLECULAR_WEIGHTS,
) -> NDArray[np.float64]:
    return {
        compound: molecular_weights[compound] * mixing_ratios[compound]
        for compound in mixing_ratios
    }


def calculate_mean_molecular_weight(
    mixing_ratios: dict[str, float | NDArray[np.float64]],
) -> float | NDArray[np.float64]:
    weighted_molecular_weights: NDArray[np.float64] = (
        calculate_weighted_molecular_weights(mixing_ratios)
    )

    return np.sum(list(weighted_molecular_weights.values()), axis=0)


def calculate_molecular_metrics(
    mixing_ratios: dict[str, float | NDArray[np.float64]],
) -> MoleculeMetrics:
    ctoo_ratio: float | NDArray[np.float64] = calculate_CtoO_ratio(mixing_ratios)

    mean_molecular_weight: float | NDArray[np.float64] = (
        calculate_mean_molecular_weight(mixing_ratios)
    )

    metallicity: float | NDArray[np.float64] = calculate_metallicity(mixing_ratios)

    return MoleculeMetrics(ctoo_ratio, mean_molecular_weight, metallicity)


def mixing_ratios_to_partial_pressures_by_species(
    mixing_ratio_by_level: NDArray[np.float64],
    molecular_weight_by_level: NDArray[np.float64],
    mean_molecular_weight_by_level: NDArray[np.float64],
):
    return (
        mixing_ratio_by_level
        * molecular_weight_by_level
        / mean_molecular_weight_by_level
    )


def mixing_ratios_to_number_densities_by_species(
    mixing_ratios_by_level: NDArray[np.float64],
    molecular_weights_by_level: NDArray[np.float64],
    mean_molecular_weights_by_level: NDArray[np.float64],
    pressure_in_cgs: NDArray[np.float64],
    temperatures_in_K: NDArray[np.float64],
) -> NDArray[np.float64]:
    partial_pressure_fraction = mixing_ratios_to_partial_pressures_by_species(
        mixing_ratios_by_level,
        molecular_weights_by_level,
        mean_molecular_weights_by_level,
    )

    return (
        partial_pressure_fraction
        * pressure_in_cgs
        / (BOLTZMANN_CONSTANT_IN_CGS * temperatures_in_K)
    )


def mixing_ratios_to_number_densities(
    mixing_ratios_by_level: dict[str, NDArray[np.float64]],
    molecular_weights_by_level: dict[str, NDArray[np.float64]],
    mean_molecular_weight_by_level: NDArray[np.float64],
    pressure_in_cgs: NDArray[np.float64],
    temperatures_in_K: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    return {
        species: mixing_ratios_to_number_densities_by_species(
            mixing_ratios_by_level[species],
            molecular_weights_by_level[species],
            mean_molecular_weight_by_level,
            pressure_in_cgs,
            temperatures_in_K,
        )
        for species in mixing_ratios_by_level
    }


def calculate_cumulative_molecular_metrics(
    mixing_ratios_by_level: dict[str, NDArray[np.float64]],
    molecular_weights_by_level: dict[str, NDArray[np.float64]],
    mean_molecular_weight_by_level: NDArray[np.float64],
    pressure_in_cgs: NDArray[np.float64],
    temperatures_in_K: NDArray[np.float64],
):
    # Probably should be implemented with xarray structures with species as a coordinate,
    # rather than as dictionaries. Okay for now.
    number_densities_by_level: dict[str, NDArray[np.float64]] = (
        mixing_ratios_to_number_densities(
            mixing_ratios_by_level,
            molecular_weights_by_level,
            mean_molecular_weight_by_level,
            pressure_in_cgs,
            temperatures_in_K,
        )
    )

    mean_mixing_ratios: dict[str, NDArray[np.float64]] = {
        species: np.sum(
            mixing_ratios_by_level[species]
            * number_densities_by_level[species]
            / np.sum(number_densities_by_level[species], axis=0),
            axis=0,
        )
        for species in mixing_ratios_by_level
    }

    return calculate_molecular_metrics(mean_mixing_ratios)
