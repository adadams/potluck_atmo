from typing import Final

import numpy as np
import xarray as xr

from potluck.basic_types import PressureDimension, WavelengthDimension
from potluck.xarray_functional_wrappers import Dimensionalize, set_result_name_and_units


@set_result_name_and_units(result_names="H-_bound-free", units="cm^-2")
@Dimensionalize(
    argument_dimensions=((WavelengthDimension,),),
    result_dimensions=((WavelengthDimension,),),
)
def HminBoundFree(
    wavelengths_in_microns: np.ndarray[np.float64],
    cutoff_wavelength_in_microns: float = 1.6419,
) -> np.ndarray[np.float64]:
    def active_bound_free(
        wavelengths: np.ndarray[np.float64],
        cutoff_wavelength_in_microns: float = cutoff_wavelength_in_microns,
    ):
        x = (1.0 / wavelengths - 1.0 / cutoff_wavelength_in_microns) ** 0.5

        f = 4.982
        f = f * x - 34.194
        f = f * x + 92.536
        f = f * x - 118.858
        f = f * x + 49.534
        f = f * x + 152.519

        return (wavelengths * x) ** 3 * f * 1.0e-18

    wavelengths_within_cutoff: np.ndarray[np.float64] = wavelengths_in_microns[
        wavelengths_in_microns < cutoff_wavelength_in_microns
    ]

    wavelengths_beyond_cutoff: np.ndarray[np.float64] = wavelengths_in_microns[
        wavelengths_in_microns >= cutoff_wavelength_in_microns
    ]

    return np.r_[
        active_bound_free(wavelengths_within_cutoff),
        np.zeros_like(wavelengths_beyond_cutoff),
    ]


@set_result_name_and_units(result_names="H-_free-free", units="cm^-5")
@Dimensionalize(
    argument_dimensions=((PressureDimension,), (WavelengthDimension,)),
    result_dimensions=((WavelengthDimension, PressureDimension),),
)
def HminFreeFree(
    temperatures_in_K: np.ndarray[np.float64],
    wavelengths_in_microns: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    wavelengths_in_microns: np.ndarray[np.float64] = wavelengths_in_microns[
        :, np.newaxis
    ]

    _aj1 = np.array([0.0, 2483.346, -3449.889, 2200.040, -696.271, 88.283])
    _bj1 = np.array([0.0, 285.827, -1158.382, 2427.719, -1841.400, 444.517])
    _cj1 = np.array([0.0, -2054.291, 8746.523, -13651.105, 8624.970, -1863.864])
    _dj1 = np.array([0.0, 2827.776, -11485.632, 16755.524, -10051.530, 2095.288])
    _ej1 = np.array([0.0, -1341.537, 5303.609, -7510.494, 4400.067, -901.788])
    _fj1 = np.array([0.0, 208.952, -812.939, 1132.738, -655.020, 132.985])

    _aj2 = np.array([518.1021, 473.2636, -482.2089, 115.5291, 0.0, 0.0])
    _bj2 = np.array([-734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0])
    _cj2 = np.array([1021.1775, -1977.3395, 1096.8827, -245.6490, 0.0, 0.0])
    _dj2 = np.array([-479.0721, 922.3575, -521.1341, 114.2430, 0.0, 0.0])
    _ej2 = np.array([93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0])
    _fj2 = np.array([-6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0])

    # --- Branch 1: al > 0.3645 ---
    val1 = _fj1
    val1 = val1 / wavelengths_in_microns + _ej1
    val1 = val1 / wavelengths_in_microns + _dj1
    val1 = val1 / wavelengths_in_microns + _cj1
    val1 = val1 / wavelengths_in_microns + _bj1
    val1 = val1 + wavelengths_in_microns**2 * _aj1
    val1 = val1 / wavelengths_in_microns
    hj1 = 1.0e-29 * val1

    # --- Branch 2: al < 0.1823 ---
    val2 = _fj2
    val2 = val2 / wavelengths_in_microns + _ej2
    val2 = val2 / wavelengths_in_microns + _dj2
    val2 = val2 / wavelengths_in_microns + _cj2
    val2 = val2 / wavelengths_in_microns + _bj2
    val2 = val2 + wavelengths_in_microns**2 * _aj2
    val2 = val2 / wavelengths_in_microns
    hj2 = 1.0e-29 * val2

    _HMIN_EXPONENTS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    hj = np.zeros(6)
    hj = np.where(wavelengths_in_microns < 0.1823, hj2, hj)
    hj = np.where(wavelengths_in_microns > 0.3645, hj1, hj)

    tcoeff = (5040.0 / temperatures_in_K)[:, np.newaxis]
    t_vector = tcoeff**_HMIN_EXPONENTS
    sff_hm = np.sum(t_vector * hj[:, np.newaxis, :], axis=-1)

    k = 1.38064852e-16
    sff_hm = sff_hm * k * temperatures_in_K

    sff_hm = np.where(
        (temperatures_in_K < 800) | (wavelengths_in_microns > 20), 0.0, sff_hm
    )

    return sff_hm


@set_result_name_and_units(result_names="H2-_free-free", units="cm^-5")
@Dimensionalize(
    argument_dimensions=((PressureDimension,), (WavelengthDimension,)),
    result_dimensions=((WavelengthDimension, PressureDimension),),
)
def H2minFreeFree(
    temperatures_in_K: np.ndarray[np.float64],
    wavelengths_in_microns: np.ndarray[np.float64],
) -> np.ndarray[np.float64]:
    """
    Computes the H2- free-free absorption coefficient (cm^5).
    Based on Somerville (1964) polynomial fits.
    """
    wavelengths_in_microns: np.ndarray[np.float64] = wavelengths_in_microns[
        :, np.newaxis
    ]

    # Coefficients from Somerville (1964)
    c0: np.ndarray[np.float64] = np.array(
        [0.0, 1.1469, -1.3390, 0.6300, -0.1290, 0.0094]
    )
    c1: np.ndarray[np.float64] = np.array(
        [0.0, -0.1226, 0.1550, -0.0768, 0.0159, -0.0012]
    )

    # Horner's method for wavelength-dependent coefficients
    def evaluate_poly(coeffs, x):
        res = coeffs[-1]
        for c in reversed(coeffs[:-1]):
            res = res * x + c
        return res

    theta: np.ndarray[np.float64] = 5040.0 / temperatures_in_K

    f0: np.ndarray[np.float64] = evaluate_poly(c0, theta)
    f1: np.ndarray[np.float64] = evaluate_poly(c1, theta)

    k_B: Final[float] = 1.38064852e-16  # cgs

    val: np.ndarray[np.float64] = (1.0e-29) * (
        wavelengths_in_microns**2 * f0 + wavelengths_in_microns**3 * f1
    )
    sff_h2m: np.ndarray[np.float64] = val * k_B * temperatures_in_K

    sff_h2m: np.ndarray[np.float64] = np.where(
        (temperatures_in_K < 500) | (wavelengths_in_microns > 50), 0.0, sff_h2m
    )

    return sff_h2m


def calculate_h_minus_mixing_ratios(
    gas_chemistry: xr.Dataset,
    temperatures_by_level: xr.DataArray,
    total_number_densities: xr.DataArray,
    h2_filler_species: str = "h2he",
    h2_filler_fraction: float = 0.83,
) -> xr.Dataset:
    k_B = 1.38064852e-16  # cgs
    eV_in_ergs = 1.60218e-12
    h_planck = 6.626e-27  # cgs
    m_e = 9.109e-28  # grams
    m_H = 1.6735e-24  # grams

    gas_number_densities: xr.Dataset = gas_chemistry * total_number_densities

    h2_filler_original_number_density: xr.DataArray = gas_number_densities[
        h2_filler_species
    ]
    h2_original_number_density: xr.DataArray = (
        h2_filler_original_number_density * h2_filler_fraction
    )

    H2_dissociation_saha_component: xr.DataArray = (
        ((np.pi * m_H * k_B * temperatures_by_level) / h_planck**2) ** (3 / 2)
    ).assign_attrs(units="cm^-3")

    H2_binding_energy: float = 4.478 * eV_in_ergs

    H2_dissociation_constant: xr.DataArray = (
        H2_dissociation_saha_component
        * np.exp(-H2_binding_energy / (k_B * temperatures_by_level))
    ).rename("K_dis_H2")

    H_number_density: xr.DataArray = (
        np.sqrt(h2_original_number_density * H2_dissociation_constant)
        .rename("h")
        .assign_attrs(units="cm^-3")
    )

    partition_function_ratio: float = 1 / 2  # okay approximation for alkalis

    # "quantum concentration" for electrons
    saha_temperature_component: xr.DataArray = (
        ((2 * (2 * np.pi * m_e * k_B * temperatures_by_level) / h_planck**2) ** (3 / 2))
        .rename("saha_temperature_component")
        .assign_attrs(units="cm^-3")
    )

    alkali_binding_energies: dict[str, float] = {
        "na": 5.139 * eV_in_ergs,
        "k": 4.341 * eV_in_ergs,
    }

    alkali_saha_constants: dict[str, xr.DataArray] = {
        name: (
            partition_function_ratio
            * saha_temperature_component
            * np.exp(-binding_energy / (k_B * temperatures_by_level))
        )
        for name, binding_energy in alkali_binding_energies.items()
    }

    ionization_weighted_alkali_number_densities: xr.DataArray = xr.concat(
        [
            saha_constant * gas_number_densities[species_name]
            for species_name, saha_constant in alkali_saha_constants.items()
            if species_name in gas_number_densities.data_vars
        ],
        dim="alkali",
    )

    electron_number_density: xr.DataArray = (
        np.sqrt(ionization_weighted_alkali_number_densities.sum(dim="alkali"))
        .rename("e-")
        .assign_attrs(units="cm^-3")
    )

    H_to_Hminus_partition_function_ratio: float = 2
    Hminus_binding_energy: float = 0.754 * eV_in_ergs

    Hminus_saha_constants: xr.DataArray = (
        (
            H_to_Hminus_partition_function_ratio
            * saha_temperature_component
            * np.exp(-Hminus_binding_energy / (k_B * temperatures_by_level))
        )
        .rename("saha_const_H-")
        .assign_attrs(units="cm^-3")
    )

    Hminus_number_density: xr.DataArray = (
        ((H_number_density * electron_number_density) / Hminus_saha_constants)
        .rename("h-")
        .assign_attrs(units="cm^-3")
    )

    hminus_mixing_ratios: xr.Dataset = xr.Dataset(
        {
            "h": (H_number_density / total_number_densities).assign_attrs(
                units="dimensionless"
            ),
            "h-": (Hminus_number_density / total_number_densities).assign_attrs(
                units="dimensionless"
            ),
            "e-": (electron_number_density / total_number_densities).assign_attrs(
                units="dimensionless"
            ),
        }
    )

    return hminus_mixing_ratios


def compute_h_minus_opacity_factors(
    hminus_number_densities: xr.Dataset,
    h2_filler_number_densities: xr.DataArray,
    h2_filler_fraction: float = 0.83,
) -> xr.Dataset:
    h2_filler_original_number_density: xr.DataArray = h2_filler_number_densities
    h2_original_number_density: xr.DataArray = (
        h2_filler_original_number_density * h2_filler_fraction
    )

    hminus_ff_factor: xr.DataArray = (
        (
            (hminus_number_densities["h"] * hminus_number_densities["e-"])
            / h2_filler_original_number_density
        )
        .rename("hminus_ff_factor")
        .assign_attrs(units="cm^-3")
    )

    h2minus_ff_factor: xr.DataArray = (
        (
            (h2_original_number_density * hminus_number_densities["e-"])
            / h2_filler_original_number_density
        )
        .rename("h2minus_ff_factor")
        .assign_attrs(units="cm^-3")
    )

    hminus_bf_factor: xr.DataArray = (
        (hminus_number_densities["h-"] / h2_filler_original_number_density)
        .rename("hminus_bf_factor")
        .assign_attrs(units="dimensionless")
    )

    h_minus_factors: xr.Dataset = xr.Dataset(
        {
            "hminus_ff_factor": hminus_ff_factor,
            "h2minus_ff_factor": h2minus_ff_factor,
            "hminus_bf_factor": hminus_bf_factor,
        }
    )

    return h_minus_factors


def compile_anionic_opacity_crosssections(
    h_minus_opacity_factors: xr.Dataset,
    temperatures_by_layer: xr.DataArray,
    wavelengths: xr.DataArray,
) -> xr.DataArray:
    sigma_bf_hminus: xr.DataArray = HminBoundFree(wavelengths)
    sigma_ff_hminus: xr.DataArray = HminFreeFree(temperatures_by_layer, wavelengths)
    sigma_ff_h2minus: xr.DataArray = H2minFreeFree(temperatures_by_layer, wavelengths)

    opac_bf_hminus: xr.DataArray = (
        (h_minus_opacity_factors.hminus_bf_factor * sigma_bf_hminus)
        .rename("h-_bound-free")
        .assign_attrs(units="cm^2 / molecule")
    )
    opac_ff_hminus: xr.DataArray = (
        (h_minus_opacity_factors.hminus_ff_factor * sigma_ff_hminus)
        .rename("h-_free-free")
        .assign_attrs(units="cm^2 / molecule")
    )

    opac_ff_h2minus: xr.DataArray = (
        (h_minus_opacity_factors.h2minus_ff_factor * sigma_ff_h2minus)
        .rename("h2-_free-free")
        .assign_attrs(units="cm^2 / molecule")
    )

    total_anionic_opacity: xr.DataArray = (
        (opac_bf_hminus + opac_ff_hminus + opac_ff_h2minus)
        .rename("h-h2-")
        .assign_attrs(units="cm^2 / molecule")
    )

    return total_anionic_opacity
