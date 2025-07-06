import numpy as np
from numba import njit


@njit
def convolve_with_constant_FWHM(
    model_wavelengths: np.ndarray[np.float64],
    model_spectral_quantity: np.ndarray[np.float64],
    fwhm: float,
) -> np.ndarray[np.float64]:
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

    return Fratio_int


def convolve_with_constant_FWHM_using_FFT(
    model_wavelengths: np.ndarray,
    model_spectral_quantity: np.ndarray,
    fwhm: float,
) -> np.ndarray:
    sigma: np.ndarray = fwhm / 2.355

    # Create a Gaussian kernel with the same size as the input arrays
    kernel: np.ndarray = np.exp(
        -((model_wavelengths - model_wavelengths[0]) ** 2) / (2 * sigma**2)
    )

    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Perform circular convolution using FFT
    fft_conv: np.ndarray[np.complex64] = np.fft.ifft(
        np.fft.fft(model_spectral_quantity) * np.fft.fft(kernel)
    )

    # Shift the result to match the original function
    fft_conv: np.ndarray[np.complex64] = np.roll(fft_conv, -np.argmax(kernel))

    return np.real(fft_conv)
