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


def convolve_to_constant_R(
    model_wavelengths: np.ndarray, model_spectral_quantity: np.ndarray, R: float
) -> np.ndarray:
    """
    Convolves a spectrum on a log-uniform grid to a constant spectral resolution R.
    This is the ideal method for modeling instrumental or velocity broadening.

    Args:
        model_wavelengths: The log-uniform wavelength grid.
        model_spectral_quantity: The spectral data.
        R: The target spectral resolution (R = lambda / FWHM).
    """

    # 1. Get the properties of the log-uniform grid
    log_wav = np.log(model_wavelengths)
    # The step size in log-lambda space is constant
    delta_log_wav = log_wav[1] - log_wav[0]

    # 2. Convert the resolution R into a sigma in pixel units
    # FWHM in log-lambda space is approx. FWHM_wav / wav = 1/R
    fwhm_log_wav = 1.0 / R
    # Convert FWHM to sigma (sigma = FWHM / 2.355)
    sigma_log_wav = fwhm_log_wav / 2.355
    # Convert sigma from log-lambda units to pixel units
    sigma_pix = sigma_log_wav / delta_log_wav

    # 3. Create a standard Gaussian kernel in pixel space
    N = len(model_wavelengths)
    # Create a coordinate axis in pixels, centered at 0 and wrapped for FFT
    x_pix = np.arange(N)
    x_pix = np.where(x_pix >= N // 2, x_pix - N, x_pix)

    kernel = np.exp(-(x_pix**2) / (2 * sigma_pix**2))
    kernel /= np.sum(kernel)  # Normalize

    # 4. Perform the convolution using FFT
    convolved_spectrum = np.fft.ifft(
        np.fft.fft(model_spectral_quantity) * np.fft.fft(kernel)
    )

    return np.real(convolved_spectrum)
