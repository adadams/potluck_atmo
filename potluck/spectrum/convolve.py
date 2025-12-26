import numpy as np


def convolve_with_constant_FWHM_FFT(
    model_wavelengths: np.ndarray,
    model_spectral_quantity: np.ndarray,
    width_scale: float,
) -> np.ndarray:
    N = len(model_wavelengths)
    sigma_pix = width_scale / 2.355

    dist_grid = np.fft.fftfreq(N, d=1.0) * N

    kernel = np.exp(-(dist_grid**2) / (2 * sigma_pix**2))
    kernel /= np.sum(kernel)  # Normalize

    convolved_spectrum = np.fft.ifft(
        np.fft.fft(model_spectral_quantity) * np.fft.fft(kernel)
    )

    return np.real(convolved_spectrum)


def convolve_to_constant_R(
    model_wavelengths: np.ndarray, model_spectral_quantity: np.ndarray, R: float
) -> np.ndarray:
    log_wav = np.log(model_wavelengths)
    delta_log_wav = log_wav[1] - log_wav[0]

    fwhm_log_wav = 1.0 / R
    sigma_log_wav = fwhm_log_wav / 2.355
    sigma_pix = sigma_log_wav / delta_log_wav

    N = len(model_wavelengths)

    x_pix = np.arange(N)
    x_pix = np.where(x_pix >= N // 2, x_pix - N, x_pix)

    kernel = np.exp(-(x_pix**2) / (2 * sigma_pix**2))
    kernel /= np.sum(kernel)

    convolved_spectrum = np.fft.ifft(
        np.fft.fft(model_spectral_quantity) * np.fft.fft(kernel)
    )

    return np.real(convolved_spectrum)
