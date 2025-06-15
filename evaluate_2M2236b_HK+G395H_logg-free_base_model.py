from importlib import import_module
from pathlib import Path
from types import ModuleType

import xarray as xr
from matplotlib import pyplot as plt

from calculate_RT import calculate_observed_fluxes
from model_statistics.calculate_statistics import calculate_log_likelihood
from spectrum.bin import resample_spectral_quantity_to_new_wavelengths
from user.input_structs import UserForwardModelInputs

model_directory_label: str = "2M2236b_HK+G395H_logg-free"

project_directory: Path = Path.cwd()
user_directory: Path = project_directory / "user"
model_directory: Path = user_directory / f"{model_directory_label}_model"
input_file_directory: Path = model_directory / "input_files"
intermediate_output_directory: Path = model_directory / "intermediate_outputs"
output_file_directory: Path = model_directory / "output_files"

plt.style.use(project_directory / "arthur.mplstyle")

parent_directory: str = "user"

forward_model_module: ModuleType = import_module(
    f"{parent_directory}.{model_directory_label}_model.input_files.{model_directory_label}_forward_model_inputs"
)

default_forward_model_inputs: UserForwardModelInputs = (
    forward_model_module.default_forward_model_inputs
)

reference_model_wavelengths: xr.DataArray = (
    forward_model_module.reference_model_wavelengths
)

fraction_of_reddest_fwhm_to_convolve_with: float = 1.00  # 0.01


def calculate_emission_model(
    forward_model_inputs: UserForwardModelInputs,
    resampling_fwhm_fraction: float = fraction_of_reddest_fwhm_to_convolve_with,
) -> float:
    emission_fluxes = calculate_observed_fluxes(
        user_forward_model_inputs=forward_model_inputs,
        precalculated_crosssection_catalog=None,
    )

    reference_model_wavelengths: xr.DataArray = forward_model_inputs.output_wavelengths

    selected_emission_fluxes = emission_fluxes["observed_twostream_flux"]

    emission_fluxes_sampled_to_data: xr.DataArray = (
        resample_spectral_quantity_to_new_wavelengths(
            reference_model_wavelengths,
            selected_emission_fluxes.wavelength,
            selected_emission_fluxes,
            fwhm=resampling_fwhm_fraction
            * (
                reference_model_wavelengths.to_numpy()[-1]
                - reference_model_wavelengths.to_numpy()[-2]
            ),
        )
    ).rename("resampled_emission_flux")

    return emission_fluxes_sampled_to_data


def calculate_emission_model_log_likelihood(
    forward_model_inputs: UserForwardModelInputs,
    data: xr.DataArray,
    data_error: xr.DataArray,
    resampling_fwhm_fraction: float = fraction_of_reddest_fwhm_to_convolve_with,
) -> float:
    transit_depths_sampled_to_data: xr.DataArray = calculate_emission_model(
        forward_model_inputs=forward_model_inputs,
        resampling_fwhm_fraction=resampling_fwhm_fraction,
    )

    log_likelihood: float = calculate_log_likelihood(
        transit_depths_sampled_to_data, data, data_error
    )

    return log_likelihood


if __name__ == "__main__":
    apollo_model_filepath: Path = (
        input_file_directory / "2M2236b_HK+G395H_logg-free_apollo_model.nc"
    )
    apollo_model: xr.Dataset = xr.open_dataset(apollo_model_filepath)

    test_model_spectrum: xr.DataArray = calculate_emission_model(
        forward_model_inputs=default_forward_model_inputs,
        resampling_fwhm_fraction=fraction_of_reddest_fwhm_to_convolve_with,
    )

    # at wavelengths less than 4.1 microns, scale by 0.8663427867
    # at wavelengths greater than 4.1 microns, scale by 1.050878524
    model_spectrum_rescaled: xr.DataArray = xr.where(
        (test_model_spectrum.wavelength > 2.8) & (test_model_spectrum.wavelength < 4.1),
        test_model_spectrum * 1.13337269,
        test_model_spectrum,
    )

    model_spectrum_rescaled: xr.DataArray = xr.where(
        model_spectrum_rescaled.wavelength > 4.1,
        model_spectrum_rescaled * 0.9045477122,
        model_spectrum_rescaled,
    )

    data_wavelengths: xr.DataArray = forward_model_module.reference_model_wavelengths
    data_fluxes: xr.DataArray = forward_model_module.reference_model_flux_lambda
    data_flux_errors: xr.DataArray = (
        forward_model_module.reference_model_flux_lambda_errors
    )

    print(f"{test_model_spectrum=}")

    figure, axis = plt.subplots(1, 1, figsize=(12, 8))

    axis.plot(
        test_model_spectrum.wavelength,
        model_spectrum_rescaled,
        label="``New code'' model with Apollo-fit parameters",
        color="crimson",
    )

    axis.plot(
        apollo_model.wavelength,
        apollo_model.flux,
        label="Apollo model from original Apollo-fit parameters",
        color="dodgerblue",
    )

    axis.errorbar(
        data_wavelengths,
        data_fluxes,
        yerr=data_flux_errors,
        label="Data",
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=0.5,
        capsize=2,
        capthick=0.5,
        zorder=-1,
    )

    # text in upper right corner
    axis.text(
        0.95,
        0.95,
        "HK+G395H logg-free",
        color="black",
        fontsize=28,
        ha="right",
        va="top",
        transform=axis.transAxes,
    )

    axis.set_xlabel("Wavelength (microns)")
    axis.set_ylabel("Emission flux (erg cm$^{-2}$ s$^{-1}$ cm$^{-1}$)")

    axis.legend(fontsize=16, frameon=False)

    plt.show()

    figure.savefig(
        output_file_directory / "2M2236b_HK+G395H_logg-free_model_comparison.pdf",
        bbox_inches="tight",
    )
