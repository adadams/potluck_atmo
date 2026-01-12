from typing import Final

import msgspec
import numpy as np
import xarray as xr

from potluck.basic_types import NonnegativeValue
from potluck.constants_and_conversions import BOLTZMANN_CONSTANT_IN_CGS

SOLAR_METAL_FRACTION: Final[float] = 0.0196

NONMETAL_MOLECULAR_WEIGHTS: Final[dict[str, float]] = {
    "h": 1.01,
    "h-": 1.01,
    "h2": 2.02,
    "h2only": 2.02,
    "he": 4.00,
    "h2he": 2.55,  # 83% H2, 17% He
    "h2heh-": 2.55,  # 83% H2, 17% He
    "e-": 0.0005485799094,
}

METAL_MOLECULAR_WEIGHTS: Final[dict[str, float]] = {
    "h2o": 18.02,
    "ch4": 16.04,
    "co": 28.01,
    "12co": 28.01,
    "13co": 29.00,
    "co2": 44.01,
    "nh3": 17.03,
    "h2s": 34.07,
    "Burrows_alk": 23.94,  # ~1 part potassium for 16 parts sodium
    "Lupu_alk": 23.94,
    "na": 22.99,
    "k": 39.098,
    "crh": 53.00,
    "feh": 56.85,
    "tio": 63.87,
    "vo": 66.94,
    "hcn": 27.03,
    "n2": 28.01,
    "ph3": 34.00,
}

CONDENSATE_MOLECULAR_WEIGHTS: Final[dict[str, float]] = {
    "mgsio3": 100.387,
    "mg2sio4": 140.69,
    "na2s": 78.04,
    "al2o3": 101.961,
    "fe": 55.845,
}

MOLECULAR_WEIGHTS: Final[dict[str, float]] = (
    METAL_MOLECULAR_WEIGHTS | NONMETAL_MOLECULAR_WEIGHTS | CONDENSATE_MOLECULAR_WEIGHTS
)

HYDROGEN_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {
    "h": 1,
    "h-": 1,
    "h2": 2,
    "h2only": 2,
    "h2he": 2,
    "h2heh-": 3,
    "e-": 0,
    "h2o": 2,
    "h2s": 2,
    "ch4": 4,
    "nh3": 3,
    "crh": 1,
    "feh": 1,
    "hcn": 1,
    "ph3": 3,
}

HELIUM_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"he": 1, "h2he": 1, "h2heh-": 1}

CARBON_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {
    "ch4": 1,
    "co": 1,
    "12co": 1,
    "13co": 1,
    "co2": 1,
    "hcn": 1,
}

NITROGEN_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"nh3": 1, "n2": 2, "hcn": 1}

OXYGEN_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {
    "h2o": 1,
    "co": 1,
    "12co": 1,
    "13co": 1,
    "co2": 2,
    "tio": 1,
    "vo": 1,
    "al2o3": 3,
    "mgsio3": 3,
    "mg2sio4": 4,
}

SODIUM_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {
    "na": 1,
    "Lupu_alk": 1 / (1 + 10 ** (5.07 - 6.93)),
    "Burrows_alk": 1 / (1 + 10 ** (5.07 - 6.93)),
    "na2s": 2,
}


MAGNESIUM_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"mgsio3": 1, "mg2sio4": 2}

ALUMINUM_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"al2o3": 2}

SILICON_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"mgsio3": 1, "mg2sio4": 1}

PHOSPHORUS_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"ph3": 1}

SULFUR_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"h2s": 1, "na2s": 1}

CHLORINE_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"kcl": 1}

POTASSIUM_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {
    "k": 1,
    "Lupu_alk": 1 / (1 + 10 ** (6.93 - 5.07)),
    "Burrows_alk": 1 / (1 + 10 ** (6.93 - 5.07)),
    "kcl": 1,
}

# TODO: for if we end up using a calcium species
CALCIUM_ATOMS_PER_MOLECULE: Final[dict[str, int]] = dict()

TITANIUM_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"tio": 1}

VANADIUM_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"vo": 1}

CHROMIUM_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"crh": 1}

IRON_ATOMS_PER_MOLECULE: Final[dict[str, int]] = {"feh": 1, "fe": 1}

ATOMS_PER_MOLECULE_BY_ELEMENT: Final[dict[str, dict[str, int]]] = {
    "h": HYDROGEN_ATOMS_PER_MOLECULE,
    "he": HELIUM_ATOMS_PER_MOLECULE,
    "c": CARBON_ATOMS_PER_MOLECULE,
    "n": NITROGEN_ATOMS_PER_MOLECULE,
    "o": OXYGEN_ATOMS_PER_MOLECULE,
    "na": SODIUM_ATOMS_PER_MOLECULE,
    "mg": MAGNESIUM_ATOMS_PER_MOLECULE,
    "al": ALUMINUM_ATOMS_PER_MOLECULE,
    "si": SILICON_ATOMS_PER_MOLECULE,
    "p": PHOSPHORUS_ATOMS_PER_MOLECULE,
    "s": SULFUR_ATOMS_PER_MOLECULE,
    "cl": CHLORINE_ATOMS_PER_MOLECULE,
    "k": POTASSIUM_ATOMS_PER_MOLECULE,
    "ca": CALCIUM_ATOMS_PER_MOLECULE,
    "ti": TITANIUM_ATOMS_PER_MOLECULE,
    "v": VANADIUM_ATOMS_PER_MOLECULE,
    "cr": CHROMIUM_ATOMS_PER_MOLECULE,
    "fe": IRON_ATOMS_PER_MOLECULE,
}

ASPLUND_SOLAR_ELEMENTAL_ABUNDANCES_RELATIVE_TO_H: Final[dict[str, float]] = {
    "h": 1,
    "he": 10 ** (10.914 - 12),
    "c": 10 ** (8.46 - 12),
    "n": 10 ** (7.83 - 12),
    "o": 10 ** (8.69 - 12),
    "na": 10 ** (6.93 - 12),
    "mg": 10 ** (7.55 - 12),
    "al": 10 ** (6.43 - 12),
    "si": 10 ** (7.51 - 12),
    "p": 10 ** (5.41 - 12),
    "s": 10 ** (7.12 - 12),
    "cl": 10 ** (5.31 - 12),
    "k": 10 ** (5.07 - 12),
    "ca": 10 ** (6.30 - 12),
    "ti": 10 ** (4.97 - 12),
    "v": 10 ** (3.90 - 12),
    "cr": 10 ** (5.62 - 12),
    "fe": 10 ** (7.46 - 12),
}


class MoleculeMetrics(msgspec.Struct):
    CtoO_ratio: float | np.ndarray[np.float64]
    mean_molecular_weight: float | np.ndarray[np.float64]
    metallicity: float | np.ndarray[np.float64]


def calculate_CtoO_ratio(
    mixing_ratios: dict[str, float | np.ndarray[np.float64]],
) -> float | np.ndarray[np.float64]:
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


def calculate_elemental_ratios_by_xarray(
    number_density_by_level: xr.Dataset,
    reference_molecular_data: xr.Dataset,
    numerator_elements: str | list[str] = ["c", "c", "n"],
    denominator_elements: str | list[str] = ["o", "h", "h"],
) -> NonnegativeValue:
    numerator_elemental_set: list[str] = list(set(numerator_elements))
    denominator_elemental_set: list[str] = list(set(denominator_elements))

    number_column_density: xr.DataArray = number_density_by_level.to_dataarray(
        dim="molecular_species"
    ).sum("pressure")

    reference_numerator_element_count: xr.DataArray = (
        reference_molecular_data.atoms_per_molecule.sel(element=numerator_elemental_set)
    )

    reference_denominator_element_count: xr.DataArray = (
        reference_molecular_data.atoms_per_molecule.sel(
            element=denominator_elemental_set
        )
    )

    elemental_ratios: dict[str, NonnegativeValue] = {
        f"{numerator_element}_to_{denominator_element}": (
            reference_numerator_element_count.sel(element=numerator_element)
            * number_column_density
        )
        .sum("molecular_species")
        .item()
        / (
            reference_denominator_element_count.sel(element=denominator_element)
            * number_column_density
        )
        .sum("molecular_species")
        .item()
        for numerator_element, denominator_element in zip(
            numerator_elements, denominator_elements
        )
    }

    return elemental_ratios


def calculate_C_to_O_ratio_by_xarray(
    number_density_by_level: xr.Dataset, reference_molecular_data: xr.Dataset
) -> NonnegativeValue:
    return calculate_elemental_ratios_by_xarray(
        number_density_by_level,
        reference_molecular_data,
        numerator_elements="c",
        denominator_elements="o",
    )["c_to_o"]


def calculate_metallicity_by_molecular_weight_ratio(
    mixing_ratios: dict[str, float | np.ndarray[np.float64]],
    reference_metal_fraction: float = SOLAR_METAL_FRACTION,
) -> float | np.ndarray[np.float64]:
    metal_compounds = METAL_MOLECULAR_WEIGHTS.keys()

    metals = 0.0

    for gas_species, mixing_ratio in mixing_ratios.items():
        if gas_species in metal_compounds:
            metals = metals + (METAL_MOLECULAR_WEIGHTS[gas_species] * mixing_ratio)

    return np.log10(metals / reference_metal_fraction)


def calculate_metallicity(
    mixing_ratios: dict[str, float | np.ndarray[np.float64]],
    reference_solar_abundances: dict[
        str, float
    ] = ASPLUND_SOLAR_ELEMENTAL_ABUNDANCES_RELATIVE_TO_H,
):
    all_elemental_ratios: dict[str, float] = {
        element_name: sum(
            [
                atoms_per_molecule * mixing_ratios[species_name]
                for species_name, atoms_per_molecule in ATOMS_PER_MOLECULE_BY_ELEMENT[
                    element_name
                ]
                if species_name in mixing_ratios
            ]
        )
        for element_name in reference_solar_abundances
    }

    H_elemental_ratio: float = all_elemental_ratios["h"]
    metal_elemental_ratios: list[float] = [
        all_elemental_ratios[element_name]
        for element_name in reference_solar_abundances
        if element_name not in ["h", "he"]
    ]

    metallicity_ratio: float = sum(metal_elemental_ratios) / H_elemental_ratio

    H_solar_elemental_ratio: float = ASPLUND_SOLAR_ELEMENTAL_ABUNDANCES_RELATIVE_TO_H[
        "h"
    ]
    metal_solar_elemental_ratios: list[float] = [
        reference_solar_abundances[element_name]
        for element_name in reference_solar_abundances
        if element_name not in ["h", "he"]
    ]

    solar_metallicity_ratio: float = (
        sum(metal_solar_elemental_ratios) / H_solar_elemental_ratio
    )

    return np.log10(metallicity_ratio / solar_metallicity_ratio)


def calculate_metallicity_by_xarray(
    number_density_by_level: xr.Dataset, reference_molecular_data: xr.Dataset
) -> float:
    number_density_by_level: xr.DataArray = number_density_by_level.to_dataarray(
        dim="molecular_species"
    )

    total_number_density: xr.DataArray = number_density_by_level.sum(
        ("molecular_species", "pressure")
    )

    reference_molecular_data_for_model = reference_molecular_data.sel(
        molecular_species=number_density_by_level.molecular_species
    )

    is_H_or_He: xr.DataArray = reference_molecular_data_for_model.element.isin(
        ["h", "he"]
    )
    is_a_metal: xr.DataArray = ~is_H_or_He

    reference_H_data_for_model: xr.Dataset = reference_molecular_data_for_model.sel(
        element="h"
    )

    reference_molecular_data_for_model_by_metals: xr.Dataset = (
        reference_molecular_data_for_model.where(is_a_metal, drop=True)
    )

    model_metal_ratio: NonnegativeValue = (
        (
            number_density_by_level
            * reference_molecular_data_for_model_by_metals.atoms_per_molecule
        ).sum(("molecular_species", "pressure", "element"))
        / total_number_density
    ).item()

    model_H_ratio: NonnegativeValue = (
        (number_density_by_level * reference_H_data_for_model.atoms_per_molecule).sum(
            ("molecular_species", "pressure")
        )
        / total_number_density
    ).item()

    solar_metal_ratio_by_element: NonnegativeValue = (
        reference_molecular_data_for_model_by_metals.solar_elemental_abundance_relative_to_H.sum(
            "element"
        )
    ).item()

    solar_H_ratio: NonnegativeValue = (
        reference_H_data_for_model.solar_elemental_abundance_relative_to_H
    ).item()

    return np.log10(
        (model_metal_ratio / model_H_ratio)
        / (solar_metal_ratio_by_element / solar_H_ratio)
    )


def curate_molecular_weights(
    mixing_ratios: dict[str, float | np.ndarray[np.float64]],
    molecular_weights: dict[str, float] = MOLECULAR_WEIGHTS,
) -> np.ndarray[np.float64]:
    # TODO: this is essentially just picking the compounds by key.

    return {compound: molecular_weights[compound] for compound in mixing_ratios}


def calculate_weighted_molecular_weights(
    mixing_ratios: dict[str, float | np.ndarray[np.float64]],
    molecular_weights: dict[str, float] = MOLECULAR_WEIGHTS,
) -> np.ndarray[np.float64]:
    return {
        compound: molecular_weights[compound] * mixing_ratios[compound]
        for compound in mixing_ratios
    }


def calculate_mean_molecular_weight(
    mixing_ratios: dict[str, float | np.ndarray[np.float64]],
) -> float | np.ndarray[np.float64]:
    weighted_molecular_weights: np.ndarray[np.float64] = (
        calculate_weighted_molecular_weights(mixing_ratios)
    )

    return np.sum(list(weighted_molecular_weights.values()), axis=0)


def calculate_molecular_metrics(
    mixing_ratios: dict[str, float | np.ndarray[np.float64]],
) -> MoleculeMetrics:
    ctoo_ratio: float | np.ndarray[np.float64] = calculate_CtoO_ratio(mixing_ratios)

    mean_molecular_weight: float | np.ndarray[np.float64] = (
        calculate_mean_molecular_weight(mixing_ratios)
    )

    metallicity: float | np.ndarray[np.float64] = calculate_metallicity(mixing_ratios)

    return MoleculeMetrics(ctoo_ratio, mean_molecular_weight, metallicity)


def mixing_ratios_to_partial_pressures_by_species(
    mixing_ratio_by_level: np.ndarray[np.float64],
    molecular_weight_by_level: np.ndarray[np.float64],
    mean_molecular_weight_by_level: np.ndarray[np.float64],
):
    return (
        mixing_ratio_by_level
        * molecular_weight_by_level
        / mean_molecular_weight_by_level
    )


def calculate_total_number_density(
    pressure_in_cgs: np.ndarray[np.float64],
    temperatures_in_K: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    total_number_density: np.ndarray[np.float64] = pressure_in_cgs / (
        BOLTZMANN_CONSTANT_IN_CGS * temperatures_in_K
    )

    return total_number_density


def mixing_ratios_to_number_densities_by_species(
    mixing_ratios_by_level: np.ndarray[np.float64],
    pressure_in_cgs: np.ndarray[np.float64],
    temperatures_in_K: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    total_number_density: np.ndarray[np.float64] = calculate_total_number_density(
        pressure_in_cgs, temperatures_in_K
    )

    return mixing_ratios_by_level * total_number_density


def mixing_ratios_to_number_densities(
    mixing_ratios_by_level: dict[str, np.ndarray[np.float64]],
    pressure_in_cgs: np.ndarray[np.float64],
    temperatures_in_K: np.ndarray[np.float64],
) -> dict[str, np.ndarray[np.float64]]:
    return {
        species: mixing_ratios_to_number_densities_by_species(
            mixing_ratios_by_level[species],
            pressure_in_cgs,
            temperatures_in_K,
        )
        for species in mixing_ratios_by_level
    }


def calculate_cumulative_molecular_metrics(
    mixing_ratios_by_level: dict[str, np.ndarray[np.float64]],
    pressure_in_cgs: np.ndarray[np.float64],
    temperatures_in_K: np.ndarray[np.float64],
):
    # Probably should be implemented with xarray structures with species as a coordinate,
    # rather than as dictionaries. Okay for now.
    number_densities_by_level: dict[str, np.ndarray[np.float64]] = (
        mixing_ratios_to_number_densities(
            mixing_ratios_by_level,
            pressure_in_cgs,
            temperatures_in_K,
        )
    )

    mean_mixing_ratios: dict[str, np.ndarray[np.float64]] = {
        species: np.sum(
            mixing_ratios_by_level[species]
            * number_densities_by_level[species]
            / np.sum(number_densities_by_level[species], axis=0),
            axis=0,
        )
        for species in mixing_ratios_by_level
    }

    return calculate_molecular_metrics(mean_mixing_ratios)
