from dataclasses import dataclass, field
from typing import Final

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from basic_functional_tools import interleave
from xarray_functional_wrappers import Dimensionalize, rename_and_unitize
from xarray_serialization import CosineAngleType, PressureType, WavelengthType

MAXIMUM_EXP_FLOAT: Final[float] = 5.0

STREAM_COSINE_ANGLES: Final[NDArray[np.float64]] = np.array(
    [
        0.0446339553,
        0.1443662570,
        0.2868247571,
        0.4548133152,
        0.6280678354,
        0.7856915206,
        0.9086763921,
        0.9822200849,
    ]
)

STREAM_WEIGHTS: Final[NDArray[np.float64]] = np.array(
    [
        0.0032951914,
        0.0178429027,
        0.0454393195,
        0.0791995995,
        0.1060473494,
        0.1125057995,
        0.0911190236,
        0.0445508044,
    ]
)

stream_cosine_angles_as_dataarray: xr.DataArray = xr.DataArray(
    data=STREAM_COSINE_ANGLES,
    dims=["cosine_angle"],
    coords={"cosine_angle": STREAM_COSINE_ANGLES},
    attrs={"units": "dimensionless"},
)

stream_weights_as_dataarray: xr.DataArray = xr.DataArray(
    data=STREAM_WEIGHTS,
    dims=["cosine_angle"],
    coords={"cosine_angle": STREAM_COSINE_ANGLES},
    attrs={"units": "dimensionless"},
)

bottom_layer: list[slice] = [slice(None, None), slice(-1, None)]
top_layer: list[slice] = [slice(None), slice(0, 1)]
upper_edges: list[slice] = [slice(None, None), slice(1, None)]
lower_edges: list[slice] = [slice(None, None), slice(None, -1)]


###############################################################################
############################ Main callable function. ##########################
###############################################################################


@dataclass
class RTToon1989Inputs:
    thermal_intensity: xr.DataArray  # (wavelength, pressure)
    delta_thermal_intensity: xr.DataArray  # (wavelength, pressure)
    scattering_asymmetry_parameter: xr.DataArray  # (wavelength, pressure)
    single_scattering_albedo: xr.DataArray  # (wavelength, pressure)
    optical_depth: xr.DataArray  # (wavelength, pressure)
    stream_cosine_angles: xr.DataArray = field(
        default_factory=lambda: stream_cosine_angles_as_dataarray.copy()
    )
    stream_weights: xr.DataArray = field(
        default_factory=lambda: stream_weights_as_dataarray.copy()
    )


@rename_and_unitize(new_name="emitted_twostream_flux", units="erg s^-1 cm^-3")
@Dimensionalize(
    argument_dimensions=(
        (WavelengthType, PressureType),
        (WavelengthType, PressureType),
        (WavelengthType, PressureType),
        (WavelengthType, PressureType),
        (WavelengthType, PressureType),
        (CosineAngleType,),
        (CosineAngleType,),
    ),
    result_dimensions=((WavelengthType,),),
)
def RT_Toon1989(
    thermal_intensity: NDArray[np.float64],
    delta_thermal_intensity: NDArray[np.float64],
    scattering_asymmetry_parameter: NDArray[np.float64],
    single_scattering_albedo: NDArray[np.float64],
    optical_depth: NDArray[np.float64],
    stream_cosine_angles: NDArray[np.float64],
    stream_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    terms_for_DSolver = calculate_terms_for_DSolver(
        optical_depth,
        single_scattering_albedo,
        scattering_asymmetry_parameter,
        thermal_intensity,
        delta_thermal_intensity,
    )

    terms_for_DTRIDGL = DSolver_subroutine(*terms_for_DSolver)

    xki_terms = DTRIDGL_subroutine(*terms_for_DTRIDGL)

    return calculate_flux(
        optical_depth,
        single_scattering_albedo,
        scattering_asymmetry_parameter,
        thermal_intensity,
        delta_thermal_intensity,
        xki_terms,
        stream_cosine_angles,
        stream_weights,
    )


###############################################################################


@dataclass
class DsolverInputs:
    cp: NDArray[np.float64]
    cpm1: NDArray[np.float64]
    cm: NDArray[np.float64]
    cmm1: NDArray[np.float64]
    ep: NDArray[np.float64]
    btop: NDArray[np.float64]
    bottom: NDArray[np.float64]
    gama: NDArray[np.float64]
    rsf: float = 0


def calculate_terms_for_DSolver(
    optical_depth: NDArray[np.float64],
    single_scattering_albedo: NDArray[np.float64],
    scattering_asymmetry_parameter: NDArray[np.float64],
    thermal_intensity: NDArray[np.float64],
    delta_thermal_intensity: NDArray[np.float64],
    mu_1: float = 0.5,  # This is mu_1 in Toon et al. 1989
) -> DsolverInputs:
    tau: NDArray[np.float64] = optical_depth
    w0: NDArray[np.float64] = single_scattering_albedo
    g: NDArray[np.float64] = scattering_asymmetry_parameter
    tbfrac: float = 1  # INCOMPLETE IMPLEMENTATION
    # tbase = getT(hmin)       # INCOMPLETE IMPLEMENTATION
    thermal_intensity_at_TOA: NDArray[np.float64] = (
        thermal_intensity[*top_layer] - delta_thermal_intensity[*top_layer] / 2
    )
    thermal_intensity_at_base: NDArray[np.float64] = (
        thermal_intensity[*bottom_layer] + delta_thermal_intensity[*bottom_layer] / 2
    )

    alpha: NDArray[np.float64] = np.sqrt((1 - w0) / (1 - w0 * g))
    lamda: NDArray[np.float64] = alpha * (1 - w0 * g) / mu_1
    gama: NDArray[np.float64] = (1 - alpha) / (1 + alpha)
    term: NDArray[np.float64] = 1 / 2 / (1 - w0 * g)

    dti_x_term: NDArray[np.float64] = delta_thermal_intensity * term
    dti_x_tau: NDArray[np.float64] = delta_thermal_intensity * tau

    cpm1: NDArray[np.float64] = thermal_intensity + dti_x_term
    cp: NDArray[np.float64] = cpm1 + dti_x_tau
    cmm1: NDArray[np.float64] = thermal_intensity - dti_x_term
    cm: NDArray[np.float64] = cmm1 + dti_x_tau

    lamda_x_tau: NDArray[np.float64] = np.clip(
        lamda * tau, a_min=None, a_max=MAXIMUM_EXP_FLOAT
    )
    ep: NDArray[np.float64] = np.exp(lamda_x_tau)

    tautop: NDArray[np.float64] = tau[*top_layer]
    btop: NDArray[np.float64] = (1 - np.exp(-tautop / mu_1)) * thermal_intensity_at_TOA
    bottom: NDArray[np.float64] = thermal_intensity_at_base + delta_thermal_intensity[
        *bottom_layer
    ] * (mu_1 / tbfrac)  # Equivalent to multiplying taulayer by tbfrac.

    return cp, cpm1, cm, cmm1, ep, btop, bottom, gama


@dataclass
class DSolverOutputs:
    afs: NDArray[np.float64]
    bfs: NDArray[np.float64]
    cfs: NDArray[np.float64]
    dfs: NDArray[np.float64]


def DSolver_subroutine(
    cp: NDArray[np.float64],
    cpm1: NDArray[np.float64],
    cm: NDArray[np.float64],
    cmm1: NDArray[np.float64],
    ep: NDArray[np.float64],
    btop: NDArray[np.float64],
    bottom: NDArray[np.float64],
    gama: NDArray[np.float64],
    rsf: float = 0,
) -> DSolverOutputs:
    """
    rsf is "surface" reflectivity, can set to zero
    Surface reflectivity should be zero for emission.
    (It is used in the tridiagonal matrix in the bottom layer.)

    DSolver subroutine to compute xk1 and xk2.
    Computes a,b,c,d coefficients first, top to bottom
    Then as and ds, *bottom to top*
    Then xk coefficients, top to bottom
    af, bd, cd, and df appear to be A_l, B_l, D_l, and E_l in Toon et al.
    xk1 and xk2 appear to be Y_1n and Y_2n in Toon et al.
    However, these do not match their formulae.
    """

    e1: NDArray[np.float64] = ep + gama / ep
    e2: NDArray[np.float64] = ep - gama / ep
    e3: NDArray[np.float64] = gama * ep + 1 / ep
    e4: NDArray[np.float64] = gama * ep - 1 / ep

    gama_top_layer: np.ndarray = gama[*top_layer]
    af_top: np.ndarray = np.zeros_like(gama_top_layer)
    bf_top: np.ndarray = gama_top_layer + 1
    cf_top: np.ndarray = gama_top_layer - 1
    df_top: np.ndarray = btop - cmm1[*top_layer]

    # odd indices
    odd_afs: np.ndarray = (e1[*upper_edges] + e3[*upper_edges]) * (
        gama[*lower_edges] - 1
    )
    odd_bfs: np.ndarray = (e2[*upper_edges] + e4[*upper_edges]) * (
        gama[*lower_edges] - 1
    )
    odd_cfs: np.ndarray = 2 * (1 - gama[*lower_edges] ** 2)
    odd_dfs: np.ndarray = (gama[*lower_edges] - 1) * (
        (cpm1[*lower_edges] - cp[*upper_edges])
        + (cmm1[*lower_edges] - cm[*upper_edges])
    )

    # even indices -- NOTE: even and odd have been switched from the
    # Fortran code and Toon et al. due to fencepost effects.
    even_afs: np.ndarray = 2 * (1 - gama[*upper_edges] ** 2)
    even_bfs: np.ndarray = (e1[*upper_edges] - e3[*upper_edges]) * (
        gama[*lower_edges] + 1
    )
    even_cfs: np.ndarray = odd_afs
    even_dfs: np.ndarray = e3[*upper_edges] * (
        cpm1[*lower_edges] - cp[*upper_edges]
    ) + e1[*upper_edges] * (cm[*upper_edges] - cmm1[*lower_edges])

    af_base: np.ndarray = e1[*bottom_layer] - rsf * e3[*bottom_layer]
    bf_base: np.ndarray = e2[*bottom_layer] - rsf * e4[*bottom_layer]
    cf_base: np.ndarray = np.zeros_like(af_base)
    # NOTE: original C++ version says bsurf, but was called with bottom
    df_base: np.ndarray = bottom - cp[*bottom_layer] + rsf * cm[*bottom_layer]

    interleaved_afs: np.ndarray = interleave(odd_afs, even_afs)
    interleaved_bfs: np.ndarray = interleave(odd_bfs, even_bfs)
    interleaved_cfs: np.ndarray = interleave(odd_cfs, even_cfs)
    interleaved_dfs: np.ndarray = interleave(odd_dfs, even_dfs)

    afs: NDArray[np.float64] = np.concatenate(
        [af_top, interleaved_afs, af_base], axis=-1
    )
    bfs: NDArray[np.float64] = np.concatenate(
        [bf_top, interleaved_bfs, bf_base], axis=-1
    )
    cfs: NDArray[np.float64] = np.concatenate(
        [cf_top, interleaved_cfs, cf_base], axis=-1
    )
    dfs: NDArray[np.float64] = np.concatenate(
        [df_top, interleaved_dfs, df_base], axis=-1
    )

    return afs, bfs, cfs, dfs


def DTRIDGL_subroutine(
    afs: NDArray[np.float64],
    bfs: NDArray[np.float64],
    cfs: NDArray[np.float64],
    dfs: NDArray[np.float64],
) -> NDArray[np.float64]:
    # DTRIDGL subroutine to compute the necessary xki array
    # This matches the algorithm in Toon et al.
    af_base: NDArray[np.float64] = afs[*bottom_layer]
    bf_base: NDArray[np.float64] = bfs[*bottom_layer]
    df_base: NDArray[np.float64] = dfs[*bottom_layer]

    as_base: NDArray[np.float64] = af_base / bf_base
    ds_base: NDArray[np.float64] = df_base / bf_base

    as_terms: NDArray[np.float64] = np.empty_like(afs, dtype=np.float64)
    as_terms[*bottom_layer] = as_base
    ds_terms: NDArray[np.float64] = np.empty_like(afs, dtype=np.float64)
    ds_terms[*bottom_layer] = ds_base

    twice_number_of_layers: int = np.shape(afs)[-1]

    for half_layer in reversed(range(twice_number_of_layers)):
        xx: NDArray[np.float64] = 1 / (
            bfs[:, half_layer] - cfs[:, half_layer] * as_terms[:, half_layer]
        )
        as_terms[:, half_layer - 1] = afs[:, half_layer] * xx
        ds_terms[:, half_layer - 1] = (
            dfs[:, half_layer] - cfs[:, half_layer] * ds_terms[:, half_layer]
        ) * xx

    xki_terms: NDArray[np.float64] = np.empty_like(ds_terms)
    xki_terms[0] = ds_terms[0]

    for half_layer, (as_term, ds_term) in enumerate(
        zip(as_terms[*upper_edges].T, ds_terms[*upper_edges].T)
    ):
        xki_terms[:, half_layer + 1] = ds_term - as_term * xki_terms[:, half_layer]

    return xki_terms


def calculate_flux(
    optical_depth: NDArray[np.float64],
    single_scattering_albedo: NDArray[np.float64],
    scattering_asymmetry_parameter: NDArray[np.float64],
    thermal_intensity: NDArray[np.float64],
    delta_thermal_intensity: NDArray[np.float64],
    xki_terms: NDArray[np.float64],
    stream_cosine_angles: NDArray[np.float64],
    stream_weights: NDArray[np.float64],
    mu_1: float = 0.5,  # This is mu_1 in Toon et al. 1989
) -> NDArray[np.float64]:
    tau: NDArray[np.float64] = optical_depth
    w0: NDArray[np.float64] = single_scattering_albedo
    g: NDArray[np.float64] = scattering_asymmetry_parameter
    thermal_intensity_at_base: NDArray[np.float64] = (
        thermal_intensity[*bottom_layer] + delta_thermal_intensity[*bottom_layer] / 2
    )

    number_of_layers: int = np.shape(tau)[-1]

    # NOTE: there was a line in the original C++ for loop (index n3):
    # if(xk2[n3]!=0. && fabs(xk2[n3]/xk[2*n3] < 1.e-30)) xk2[n3] = 0.;
    # but note xk was only initialized, so all xk would be zero at this step?
    even_xki_terms: NDArray[np.float64] = xki_terms[:, 0::2]
    odd_xki_terms: NDArray[np.float64] = xki_terms[:, 1::2]
    xk1_terms: NDArray[np.float64] = even_xki_terms + odd_xki_terms
    xk2_terms: NDArray[np.float64] = even_xki_terms - odd_xki_terms

    # These are calculated just as they are in the setup function.
    # My goal is to decouple the RT components as much as possible, which leads
    # to this bit of redundant calculation (there's probably a better way!).
    alpha: NDArray[np.float64] = np.sqrt(
        (1 - w0) / (1 - w0 * g)
    )  # sqrt( (1.-w0[i][j])/(1.-w0[i][j]*asym[i][j]) )
    lamda: NDArray[np.float64] = (
        alpha * (1 - w0 * g) / mu_1
    )  # alpha[j]*(1.-w0[i][j]*cosbar[j])/mu_1
    lamda_x_tau: NDArray[np.float64] = np.clip(
        lamda * tau, a_min=None, a_max=MAXIMUM_EXP_FLOAT
    )

    # These are the variables that are used to compute the flux.
    # They are all functions of the e-coefficient and blackbody fluxes via the matrix solver.
    gg_terms: NDArray[np.float64] = (
        xk1_terms * 2 * np.pi * w0 * (1 + (g * alpha)) / (1 + alpha)
    )
    hh_terms: NDArray[np.float64] = (
        xk2_terms * 2 * np.pi * w0 * (1 - (g * alpha)) / (1 + alpha)
    )

    blackbody_scattering_term: NDArray[np.float64] = delta_thermal_intensity * (
        mu_1 * (w0 * g) / (1 - w0 * g)
    )
    alpha1: NDArray[np.float64] = (
        2 * np.pi * (thermal_intensity + blackbody_scattering_term)
    )
    alpha2: NDArray[np.float64] = 2 * np.pi * delta_thermal_intensity

    stream_cosine_angles: NDArray[np.float64] = np.expand_dims(
        stream_cosine_angles, axis=tuple(range(1, w0.ndim + 1))
    )
    stream_weights: NDArray[np.float64] = np.expand_dims(
        stream_weights, axis=tuple(range(1, w0.ndim + 1))
    )

    epp_terms: NDArray[np.float64] = np.exp(lamda_x_tau)
    em1_terms: NDArray[np.float64] = np.exp(-lamda_x_tau)
    em2_terms: NDArray[np.float64] = np.exp(-tau / stream_cosine_angles)
    em3_terms: NDArray[np.float64] = em1_terms * em2_terms

    delta_thermal_intensity_by_angle: NDArray[np.float64] = (
        delta_thermal_intensity * stream_cosine_angles
    )

    delta_thermal_intensity_by_angle_at_base: NDArray[np.float64] = (
        delta_thermal_intensity_by_angle[..., -1]
    )

    fpt_base: NDArray[np.float64] = (
        2
        * np.pi
        * (
            thermal_intensity_at_base.squeeze()
            + delta_thermal_intensity_by_angle_at_base
        )
    )
    fpt_terms: NDArray[np.float64] = np.empty_like(
        delta_thermal_intensity_by_angle, dtype=np.float64
    )
    fpt_terms[..., -1] = fpt_base

    stream_cosine_angles_by_layer: NDArray[np.float64] = stream_cosine_angles[..., 0]
    stream_weights_by_layer: NDArray[np.float64] = stream_weights[..., 0]

    for layer in reversed(range(1, number_of_layers)):
        fpt_terms[..., layer - 1] = (
            fpt_terms[..., layer] * em2_terms[..., layer]
            + gg_terms[..., layer]
            / (lamda[..., layer] * stream_cosine_angles_by_layer - 1)
            * (epp_terms[..., layer] * em2_terms[..., layer] - 1)
            + hh_terms[..., layer]
            / (lamda[..., layer] * stream_cosine_angles_by_layer + 1)
            * (1 - em3_terms[..., layer])
            + alpha1[..., layer] * (1 - em2_terms[..., layer])
            + alpha2[..., layer]
            * (
                stream_cosine_angles_by_layer * (em2_terms[..., layer] - 1)
                + tau[..., layer]
            )
        )

    fpt_at_top: NDArray[np.float64] = fpt_terms[..., 0]

    total_flux: NDArray[np.float64] = np.sum(
        stream_weights_by_layer * fpt_at_top, axis=0
    )

    return total_flux
