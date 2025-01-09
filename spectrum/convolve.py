from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import scipy
from numpy.typing import NDArray


def conv_uniform_FWHM(modspec, obspec, fwhm):
    # (number_of_segments, number_of_model_wavelengths)
    wlmod = np.reshape(modspec[0, :], modspec[0, :].size)
    #  (number_of_segments, number_of_model_wavelengths)
    Fp = np.reshape(modspec[1, :], modspec[1, :].size)
    # (number_of_segments, number_of_observed_wavelengths)
    wlobs = np.reshape(obspec[0, :], obspec[0, :].size)

    szobs = wlobs.shape[0]  # total number of observed wavelengths across all segments

    Fratio_int = np.zeros(szobs)

    for i in range(szobs):
        # total number of observed wavelengths across all segments
        sigma = fwhm / 2.355

        gauss = np.exp(-((wlmod - wlobs[i]) ** 2) / (2 * sigma**2))

        gauss = gauss / np.sum(gauss)

        Fratio_int[i] = np.sum(gauss * Fp)

    # (total number of observed wavelengths across all segments, total number of model wavelengths across all segments)
    return Fratio_int


def convolve_my_attempt(modspec, fwhm):
    # (number_of_segments, number_of_model_wavelengths)
    wlmod = np.reshape(modspec[0, :], modspec[0, :].size)
    #  (number_of_segments, number_of_model_wavelengths)
    Fp = np.reshape(modspec[1, :], modspec[1, :].size)

    szmod = wlmod.shape[0]  # total number of observed wavelengths across all segments

    Fratio_int = np.zeros(szmod)

    for j in range(szmod):
        # total number of observed wavelengths across all segments
        sigma = fwhm / 2.355

        gauss = np.exp(-((wlmod - wlmod[j]) ** 2) / (2 * sigma**2))

        gauss = gauss / np.sum(gauss)

        Fratio_int[j] = np.sum(gauss * Fp)

    # (total number of observed wavelengths across all segments, total number of model wavelengths across all segments)
    return Fratio_int


UfuncLike: TypeAlias = (
    Callable[[float], float] | Callable[[NDArray[np.float64]], NDArray[np.float64]]
)


def convolve_with_constant_FWHM(
    model_wavelengths: NDArray[np.float64],
    model_spectral_quantity: NDArray[np.float64],
    fwhm: float,
) -> NDArray[np.float64]:
    sigma: np.ndarray = fwhm / 2.355

    Fratio_int: np.ndarray = np.zeros_like(model_wavelengths)

    for wavelength_index, model_wavelength in enumerate(model_wavelengths):
        unnormalized_gaussian_kernel: np.ndarray = np.exp(
            -((model_wavelength - model_wavelengths) ** 2) / (2 * sigma**2)
        )

        normalized_gaussian_kernel: np.ndarray = unnormalized_gaussian_kernel / np.sum(
            unnormalized_gaussian_kernel
        )

        Fratio_int[wavelength_index] = np.sum(
            normalized_gaussian_kernel * model_spectral_quantity
        )

    return np.asarray(Fratio_int)


def convolve_with_constant_bin_index_FWHM(
    model_wavelengths: NDArray[np.float64],
    model_spectral_quantity: NDArray[np.float64],
    fwhm: float,
):
    sigma: np.ndarray = fwhm / 2.355

    Fratio_int: np.ndarray = np.zeros_like(model_wavelengths)

    for wavelength_index, model_wavelength in enumerate(model_wavelengths):
        unnormalized_gaussian_kernel: np.ndarray = np.exp(
            -((model_wavelength - model_wavelengths) ** 2) / (2 * sigma**2)
        )

        normalized_gaussian_kernel: np.ndarray = unnormalized_gaussian_kernel / np.sum(
            unnormalized_gaussian_kernel
        )

        Fratio_int[wavelength_index] = np.sum(
            normalized_gaussian_kernel * model_spectral_quantity
        )

    return np.asarray(Fratio_int)


def numpy_convolve_with_constant_FWHM(
    model_wavelengths: NDArray[np.float64],
    model_spectral_quantity: NDArray[np.float64],
    fwhm: float,
    number_of_sigmas_to_sample: int = 3,
    kernel_wavelength_spacing_fraction: float = 1.0,
) -> NDArray[np.float64]:
    sigma: np.ndarray = fwhm / 2.355

    kernel_wavelength_spacing: float = kernel_wavelength_spacing_fraction * np.min(
        np.diff(model_wavelengths)
    )
    kernel_wavelength_minimum: float = -number_of_sigmas_to_sample * sigma
    kernel_wavelength_maximum: float = number_of_sigmas_to_sample * sigma

    number_of_kernel_samples: int = int(
        np.ceil(
            (kernel_wavelength_maximum - kernel_wavelength_minimum)
            / kernel_wavelength_spacing
        )
    )

    kernel_wavelengths: np.ndarray = np.linspace(
        -number_of_sigmas_to_sample * sigma,
        number_of_sigmas_to_sample * sigma,
        number_of_kernel_samples,
    )

    unnormalized_gaussian_kernel: np.ndarray = np.exp(
        -((kernel_wavelengths) ** 2) / (2 * sigma**2)
    )

    normalized_gaussian_kernel: np.ndarray = unnormalized_gaussian_kernel / np.sum(
        unnormalized_gaussian_kernel
    )

    return scipy.signal.oaconvolve(
        model_spectral_quantity, normalized_gaussian_kernel, mode="same"
    )
