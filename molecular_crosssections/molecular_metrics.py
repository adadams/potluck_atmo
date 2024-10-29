from typing import Final

import msgspec
import numpy as np
from numpy.typing import NDArray

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

MOLECULAR_WEIGHTS: Final[dict[str, float]] = {
    **METAL_MOLECULAR_WEIGHTS,
    **NONMETAL_MOLECULAR_WEIGHTS,
}

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
    CtoO_ratio: NDArray[np.float64]
    mean_molecular_weight: float
    metallicity: NDArray[np.float64]


def calculate_molecular_metrics(
    gas_abundances: dict[str, float],
) -> NDArray[np.float64]:
    carbon = 0.0
    oxygen = 0.0
    metals = 0.0

    weighted_molecular_weights: NDArray[np.float64] = np.array(
        [
            MOLECULAR_WEIGHTS[compound] * abundance
            for compound, abundance in gas_abundances.items()
        ]
    )
    mean_molecular_weight: float = np.sum(weighted_molecular_weights) / np.sum(
        list(gas_abundances.values())
    )  # sum of gas_abundances should be 1

    carbon_compounds = CARBON_ATOMS_PER_MOLECULE.keys()
    oxygen_compounds = OXYGEN_ATOMS_PER_MOLECULE.keys()
    metal_compounds = METAL_MOLECULAR_WEIGHTS.keys()

    for gas_species, gas_abundance in gas_abundances.items():
        if gas_species in carbon_compounds:
            carbon = carbon + (CARBON_ATOMS_PER_MOLECULE[gas_species] * gas_abundance)

        if gas_species in oxygen_compounds:
            oxygen = oxygen + (OXYGEN_ATOMS_PER_MOLECULE[gas_species] * gas_abundance)

        if gas_species in metal_compounds:
            metals = metals + (METAL_MOLECULAR_WEIGHTS[gas_species] * gas_abundance)

    ctoo_ratio = carbon / oxygen

    SOLAR_METAL_FRACTION: Final[float] = 0.0196
    metallicity = np.log10(metals / SOLAR_METAL_FRACTION)

    return MoleculeMetrics(ctoo_ratio, mean_molecular_weight, metallicity)
