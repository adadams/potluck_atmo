from dataclasses import dataclass, field
from typing import Final, Tuple

import jax
import jax.numpy as jnp
import xarray as xr
from jax import Array

from potluck.basic_functional_tools import interleave_with_jax as interleave
from potluck.basic_types import (
    CosineAngleDimension,
    PressureDimension,
    WavelengthDimension,
)
from potluck.xarray_functional_wrappers import Dimensionalize, set_result_name_and_units

jax.config.update("jax_enable_x64", True)

STREAM_COSINE_ANGLES: Final[Array] = jnp.array(
    [
        0.0446339553,
        0.1443662570,
        0.2868247571,
        0.4548133152,
        0.6280678354,
        0.7856915206,
        0.9086763921,
        0.9822200849,
    ],
    dtype=jnp.float64,
)

STREAM_WEIGHTS: Final[Array] = jnp.array(
    [
        0.0032951914,
        0.0178429027,
        0.0454393195,
        0.0791995995,
        0.1060473494,
        0.1125057995,
        0.0911190236,
        0.0445508044,
    ],
    dtype=jnp.float64,
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

bottom_layer: Tuple[slice, slice] = (slice(None, None), slice(-1, None))
top_layer: Tuple[slice, slice] = (slice(None), slice(0, 1))
upper_edges: Tuple[slice, slice] = (slice(None, None), slice(1, None))
lower_edges: Tuple[slice, slice] = (slice(None, None), slice(None, -1))


###############################################################################
############################ Main callable function. ##########################
###############################################################################


@dataclass
class RTToon1989Inputs:
    # These remain as xr.DataArray to bridge with the xarray_functional_wrappers.
    # The jax.jit'ed RT_Toon1989 function will receive Array directly from apply_ufunc.
    thermal_intensity: xr.DataArray
    delta_thermal_intensity: xr.DataArray
    scattering_asymmetry_parameter: xr.DataArray
    single_scattering_albedo: xr.DataArray
    optical_depth: xr.DataArray
    stream_cosine_angles: xr.DataArray = field(
        default_factory=lambda: stream_cosine_angles_as_dataarray.copy()
    )
    stream_weights: xr.DataArray = field(
        default_factory=lambda: stream_weights_as_dataarray.copy()
    )


@set_result_name_and_units(
    result_names="emitted_twostream_flux", units="erg s^-1 cm^-3"
)
@Dimensionalize(
    argument_dimensions=(
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
        (WavelengthDimension, PressureDimension),
        (CosineAngleDimension,),
        (CosineAngleDimension,),
    ),
    result_dimensions=((WavelengthDimension,),),
)
@jax.jit
def RT_Toon1989(
    thermal_intensity: Array,
    delta_thermal_intensity: Array,
    scattering_asymmetry_parameter: Array,
    single_scattering_albedo: Array,
    optical_depth: Array,
    stream_cosine_angles: Array,
    stream_weights: Array,
) -> Array:
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
"""
@dataclass
class DsolverInputs:
    cp: Array
    cpm1: Array
    cm: Array
    cmm1: Array
    ep: Array
    btop: Array
    bottom: Array
    gama: Array
    rsf: float = 0
"""


@jax.jit
def calculate_terms_for_DSolver(
    optical_depth: Array,
    single_scattering_albedo: Array,
    scattering_asymmetry_parameter: Array,
    thermal_intensity: Array,
    delta_thermal_intensity: Array,
    mu_1: float = 0.5,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    tau: Array = optical_depth
    w0: Array = single_scattering_albedo
    g: Array = scattering_asymmetry_parameter
    tbfrac: float = 1.0

    thermal_intensity_at_TOA: Array = (
        thermal_intensity[top_layer] - delta_thermal_intensity[top_layer] / 2.0
    )
    thermal_intensity_at_base: Array = (
        thermal_intensity[bottom_layer] + delta_thermal_intensity[bottom_layer] / 2.0
    )

    alpha: Array = jnp.sqrt((1.0 - w0) / (1.0 - w0 * g))
    lamda: Array = alpha * (1.0 - w0 * g) / mu_1
    gama: Array = (1.0 - alpha) / (1.0 + alpha)
    term: Array = 1.0 / (2.0 * (1.0 - w0 * g))

    dti_by_tau_x_term: Array = delta_thermal_intensity / tau * term

    prefactor: Array = 2 * jnp.pi * mu_1

    cpm1_without_prefactor: Array = thermal_intensity + dti_by_tau_x_term
    cp_without_prefactor: Array = cpm1_without_prefactor + delta_thermal_intensity

    cpm1: Array = cpm1_without_prefactor * prefactor
    cp: Array = cp_without_prefactor * prefactor

    cmm1_without_prefactor: Array = thermal_intensity - dti_by_tau_x_term
    cm_without_prefactor: Array = cmm1_without_prefactor + delta_thermal_intensity

    cmm1: Array = cmm1_without_prefactor * prefactor
    cm: Array = cm_without_prefactor * prefactor

    lamda_x_tau: Array = lamda * tau
    inverse_ep: Array = jnp.exp(-lamda_x_tau)

    tautop: Array = tau[top_layer]
    btop: Array = (1.0 - jnp.exp(-tautop / mu_1)) * thermal_intensity_at_TOA
    bottom: Array = thermal_intensity_at_base + delta_thermal_intensity[
        bottom_layer
    ] * (mu_1 / tbfrac)

    return (cp, cpm1, cm, cmm1, inverse_ep, btop, bottom, gama)


"""
@dataclass
class DSolverOutputs:
    # These remain as Array to bridge between JAX functions that produce these outputs
    afs: Array
    bfs: Array
    cfs: Array
    dfs: Array
"""


@jax.jit
def DSolver_subroutine(
    cp: Array,
    cpm1: Array,
    cm: Array,
    cmm1: Array,
    inverse_ep: Array,
    btop: Array,
    bottom: Array,
    gama: Array,
    rsf: float = 0,
) -> Tuple[Array, Array, Array, Array]:
    e1 = 1 + gama * inverse_ep
    e2 = 1 - gama * inverse_ep
    e3 = gama + inverse_ep
    e4 = gama - inverse_ep

    e1_top_layer: Array = e1[*top_layer]
    e2_top_layer: Array = e2[*top_layer]
    af_top: Array = jnp.empty_like(e1_top_layer)
    bf_top: Array = e1_top_layer
    cf_top: Array = -e2_top_layer
    df_top: Array = btop - cmm1[*top_layer]

    gama_top_layer: Array = gama[top_layer]
    af_top: Array = jnp.empty_like(gama_top_layer)
    bf_top: Array = gama_top_layer + 1.0
    cf_top: Array = gama_top_layer - 1.0
    df_top: Array = btop - cmm1[top_layer]

    even_afs: Array = (
        e2[lower_edges] * e3[lower_edges] - e4[lower_edges] * e1[lower_edges]
    )
    even_bfs: Array = (
        e1[lower_edges] * e1[upper_edges] - e3[lower_edges] * e3[upper_edges]
    )
    even_cfs: Array = (
        e3[lower_edges] * e4[lower_edges] - e1[upper_edges] * e2[lower_edges]
    )
    even_dfs: Array = e3[lower_edges] * (cpm1[upper_edges] - cp[lower_edges]) + e1[
        lower_edges
    ] * (cm[upper_edges] - cmm1[lower_edges])

    odd_afs: Array = (
        e2[upper_edges] * e1[lower_edges] - e3[lower_edges] * e4[upper_edges]
    )
    odd_bfs: Array = (
        e2[lower_edges] * e2[upper_edges] - e4[lower_edges] * e4[upper_edges]
    )
    odd_cfs: Array = (
        e1[upper_edges] * e4[upper_edges] - e2[upper_edges] * e3[upper_edges]
    )
    odd_dfs: Array = e2[upper_edges] * (cpm1[upper_edges] - cp[lower_edges]) + e4[
        upper_edges
    ] * (cmm1[upper_edges] - cm[lower_edges])

    af_base: Array = e1[bottom_layer] - rsf * e3[bottom_layer]
    bf_base: Array = e2[bottom_layer] - rsf * e4[bottom_layer]
    cf_base: Array = jnp.empty_like(af_base)
    df_base: Array = bottom - cp[bottom_layer] + rsf * cm[bottom_layer]

    interleaved_afs: Array = interleave(odd_afs, even_afs)
    interleaved_bfs: Array = interleave(odd_bfs, even_bfs)
    interleaved_cfs: Array = interleave(odd_cfs, even_cfs)
    interleaved_dfs: Array = interleave(odd_dfs, even_dfs)

    afs: Array = jnp.concatenate([af_top, interleaved_afs, af_base], axis=-1)
    bfs: Array = jnp.concatenate([bf_top, interleaved_bfs, bf_base], axis=-1)
    cfs: Array = jnp.concatenate([cf_top, interleaved_cfs, cf_base], axis=-1)
    dfs: Array = jnp.concatenate([df_top, interleaved_dfs, df_base], axis=-1)

    return (afs, bfs, cfs, dfs)


@jax.jit
def DTRIDGL_subroutine(
    afs: Array,
    bfs: Array,
    cfs: Array,
    dfs: Array,
) -> Array:
    af_base: Array = afs[bottom_layer]
    bf_base: Array = bfs[bottom_layer]
    df_base: Array = dfs[bottom_layer]

    as_base: Array = af_base / bf_base
    ds_base: Array = df_base / bf_base

    as_terms: Array = jnp.empty_like(afs, dtype=jnp.float64)
    as_terms = as_terms.at[bottom_layer].set(as_base)

    ds_terms: Array = jnp.empty_like(afs, dtype=jnp.float64)
    ds_terms = ds_terms.at[bottom_layer].set(ds_base)

    twice_number_of_layers: int = jnp.shape(afs)[-1]

    def body_fn(carry, half_layer_val):
        as_terms_loop, ds_terms_loop = carry

        current_afs_col = afs[:, half_layer_val - 1]
        current_bfs_col = bfs[:, half_layer_val - 1]
        current_cfs_col = cfs[:, half_layer_val - 1]
        current_dfs_col = dfs[:, half_layer_val - 1]
        current_as_terms_col = as_terms_loop[:, half_layer_val]
        current_ds_terms_col = ds_terms_loop[:, half_layer_val]

        xx = 1.0 / (current_bfs_col - current_cfs_col * current_as_terms_col)

        new_as_val = current_afs_col * xx
        new_ds_val = (current_dfs_col - current_cfs_col * current_ds_terms_col) * xx

        as_terms_loop = as_terms_loop.at[:, half_layer_val - 1].set(new_as_val)
        ds_terms_loop = ds_terms_loop.at[:, half_layer_val - 1].set(new_ds_val)

        return (
            as_terms_loop,
            ds_terms_loop,
        ), None

    init_carry = (as_terms, ds_terms)
    (as_terms, ds_terms), _ = jax.lax.scan(
        body_fn, init_carry, jnp.arange(twice_number_of_layers - 1, -1, -1)
    )

    xki_terms: Array = jnp.empty_like(ds_terms)
    xki_terms = xki_terms.at[:, 0].set(ds_terms[:, 0])

    def body_fn_forward_scan(xki_terms, half_layer_val):
        as_term_col = as_terms[:, half_layer_val + 1]
        ds_term_col = ds_terms[:, half_layer_val + 1]

        current_xki_terms = xki_terms[:, half_layer_val]

        xki_terms = xki_terms.at[:, half_layer_val + 1].set(
            ds_term_col - as_term_col * current_xki_terms
        )

        return xki_terms, None

    xki_terms, _ = jax.lax.scan(
        body_fn_forward_scan, xki_terms, jnp.arange(0, twice_number_of_layers - 1)
    )

    return xki_terms


@jax.jit
def calculate_flux(
    optical_depth: Array,
    single_scattering_albedo: Array,
    scattering_asymmetry_parameter: Array,
    thermal_intensity: Array,
    delta_thermal_intensity: Array,
    xki_terms: Array,
    stream_cosine_angles: Array,
    stream_weights: Array,
    mu_1: float = 0.5,
) -> Array:
    tau: Array = optical_depth
    w0: Array = single_scattering_albedo
    g: Array = scattering_asymmetry_parameter
    thermal_intensity_at_base: Array = (
        thermal_intensity[bottom_layer] + delta_thermal_intensity[bottom_layer] / 2.0
    )

    number_of_layers: int = jnp.shape(tau)[-1]

    even_xki_terms: Array = xki_terms[:, 0::2]
    odd_xki_terms: Array = xki_terms[:, 1::2]

    xk1_terms: Array = even_xki_terms + odd_xki_terms
    xk2_terms: Array = even_xki_terms - odd_xki_terms

    alpha: Array = jnp.sqrt((1.0 - w0) / (1.0 - w0 * g))
    lamda: Array = alpha * (1.0 - w0 * g) / mu_1
    gama: Array = (1.0 - alpha) / (1.0 + alpha)
    lamda_x_tau: Array = lamda * tau

    term: Array = 1.0 / (2.0 * (1.0 - w0 * g))

    gg_terms: Array = xk1_terms * (1.0 / mu_1 - lamda)
    hh_terms: Array = xk2_terms * gama * (1.0 / mu_1 + lamda)

    blackbody_scattering_term: Array = term - mu_1
    alpha1: Array = (
        2.0
        * jnp.pi
        * (
            thermal_intensity
            + blackbody_scattering_term * delta_thermal_intensity / tau
        )
    )
    alpha2: Array = 2.0 * jnp.pi * delta_thermal_intensity / tau

    axis_elements = []
    for i in range(1, w0.ndim + 1):
        axis_elements.append(i)
    axis_tuple = tuple(axis_elements)

    stream_cosine_angles_expanded: Array = jnp.expand_dims(
        stream_cosine_angles, axis=axis_tuple
    )

    log_em1_terms: Array = -lamda_x_tau
    em1_terms: Array = jnp.exp(log_em1_terms)
    log_em2_terms: Array = -tau / stream_cosine_angles_expanded
    em2_terms: Array = jnp.exp(log_em2_terms)
    em3_terms: Array = jnp.exp(log_em1_terms + log_em2_terms)

    delta_thermal_intensity_expanded: Array = jnp.expand_dims(
        delta_thermal_intensity, axis=0
    )

    delta_thermal_intensity_by_angle: Array = (
        delta_thermal_intensity_expanded * stream_cosine_angles_expanded
    )

    delta_thermal_intensity_by_angle_at_base: Array = delta_thermal_intensity_by_angle[
        :, :, -1
    ]

    fpt_base: Array = (
        2.0
        * jnp.pi
        * (thermal_intensity_at_base[:, 0] + delta_thermal_intensity_by_angle_at_base)
    )

    fpt_terms_shape = (
        delta_thermal_intensity_by_angle.shape[0],
        delta_thermal_intensity_by_angle.shape[1],
        delta_thermal_intensity_by_angle.shape[2],
    )

    fpt_terms: Array = jnp.zeros(fpt_terms_shape, dtype=jnp.float64)
    fpt_terms = fpt_terms.at[:, :, -1].set(fpt_base)

    stream_cosine_angles_by_layer: Array = stream_cosine_angles_expanded[:, 0, 0]
    stream_weights_by_layer: Array = stream_weights

    def body_fn_flux_scan(fpt_terms_loop: Array, layer_val: int):
        current_fpt_term_slice = fpt_terms_loop[:, :, layer_val]
        tau_layer_expanded = tau[:, layer_val][None, :]
        lamda_layer_expanded = lamda[:, layer_val][None, :]

        em1_terms_layer_expanded = em1_terms[:, layer_val][None, :]
        em2_terms_layer_expanded = em2_terms[:, :, layer_val]
        em3_terms_layer_expanded = em3_terms[:, :, layer_val]

        gg_terms_layer_expanded = gg_terms[:, layer_val][None, :]
        hh_terms_layer_expanded = hh_terms[:, layer_val][None, :]
        alpha1_layer_expanded = alpha1[:, layer_val][None, :]
        alpha2_layer_expanded = alpha2[:, layer_val][None, :]

        cos_angle_val_for_broadcast = stream_cosine_angles_by_layer[:, None]

        term_1: Array = current_fpt_term_slice * em2_terms_layer_expanded
        term_2: Array = (
            gg_terms_layer_expanded
            / (lamda_layer_expanded * cos_angle_val_for_broadcast - 1.0)
            * (em2_terms_layer_expanded - em1_terms_layer_expanded)
        )
        term_3: Array = (
            hh_terms_layer_expanded
            / (lamda_layer_expanded * cos_angle_val_for_broadcast + 1.0)
            * (1.0 - em3_terms_layer_expanded)
        )
        term_4: Array = alpha1_layer_expanded * (
            1.0 - em2_terms_layer_expanded
        ) + alpha2_layer_expanded * (
            cos_angle_val_for_broadcast
            - em2_terms_layer_expanded
            * (tau_layer_expanded + cos_angle_val_for_broadcast)
        )

        new_fpt_value: Array = term_1 + term_2 + term_3 + term_4

        fpt_terms_loop = fpt_terms_loop.at[:, :, layer_val - 1].set(new_fpt_value)

        return fpt_terms_loop, None

    fpt_terms, _ = jax.lax.scan(
        body_fn_flux_scan, fpt_terms, jnp.arange(number_of_layers - 1, 0, -1)
    )

    fpt_at_top: Array = fpt_terms[:, :, 0]

    total_flux: Array = jnp.sum(stream_weights_by_layer[:, None] * fpt_at_top, axis=0)

    return total_flux
