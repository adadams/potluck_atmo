from importlib import import_module
from pathlib import Path
from types import ModuleType

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from calculate_RT import calculate_observed_fluxes
from spectrum.bin import resample_spectral_quantity_to_new_wavelengths
from user.input_importers import import_model_id
from user.input_structs import UserForwardModelInputs

model_directory_label: str = "R1b_retrieval"

project_directory: Path = Path.cwd()
user_directory: Path = project_directory / "user"
model_directory: Path = user_directory / f"{model_directory_label}_model"
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

model_id: str = import_model_id(
    model_directory_label=model_directory_label, parent_directory="user"
)

reference_model_wavelengths: xr.DataArray = (
    forward_model_module.reference_model_wavelengths
)

fraction_of_reddest_fwhm_to_convolve_with: float = 0.01


observed_fluxes: dict[str, xr.DataArray] = calculate_observed_fluxes(
    default_forward_model_inputs,
    precalculated_crosssection_catalog=forward_model_module.crosssection_catalog_dataset,
)

observed_fluxes_sampled_to_data: xr.DataArray = (
    resample_spectral_quantity_to_new_wavelengths(
        reference_model_wavelengths,
        observed_fluxes["transmission_flux"].wavelength,
        observed_fluxes["transmission_flux"],
        fwhm=fraction_of_reddest_fwhm_to_convolve_with
        * (
            reference_model_wavelengths.to_numpy()[-1]
            - reference_model_wavelengths.to_numpy()[-2]
        ),
    )
).rename("resampled_transmission_flux")

observed_onestream_emission_fluxes_sampled_to_data: xr.DataArray = (
    resample_spectral_quantity_to_new_wavelengths(
        reference_model_wavelengths,
        observed_fluxes["observed_onestream_flux"].wavelength,
        observed_fluxes["observed_onestream_flux"],
        fwhm=fraction_of_reddest_fwhm_to_convolve_with
        * (
            reference_model_wavelengths.to_numpy()[-1]
            - reference_model_wavelengths.to_numpy()[-2]
        ),
    )
).rename("resampled_onestream_flux")

observed_twostream_emission_fluxes_sampled_to_data: xr.DataArray = (
    resample_spectral_quantity_to_new_wavelengths(
        reference_model_wavelengths,
        observed_fluxes["observed_twostream_flux"].wavelength,
        observed_fluxes["observed_twostream_flux"],
        fwhm=fraction_of_reddest_fwhm_to_convolve_with
        * (
            reference_model_wavelengths.to_numpy()[-1]
            - reference_model_wavelengths.to_numpy()[-2]
        ),
    )
).rename("resampled_twostream_flux")

figure, axis = plt.subplots(1, 1, figsize=(15, 10))

axis.plot(
    reference_model_wavelengths,
    observed_fluxes_sampled_to_data,
    color="crimson",
    label="Resampled transit depths",
)

data_filepath: Path = model_directory / "T3B_APOLLO_test-truncated.dat"
data_wavelo, data_wavehi, data_flux, data_errors, *_ = np.loadtxt(data_filepath).T
data_wavelengths = 0.5 * (data_wavelo + data_wavehi)

axis.errorbar(
    data_wavelengths,
    data_flux,
    yerr=data_errors,
    fmt="x",
    color="black",
    capsize=5,
    label="R1b Input Data",
)

original_data_filepath: Path = model_directory / "T3B.Spectrum.binned.dat"
original_data_wavelo, original_data_wavehi, original_data_flux, *_ = np.loadtxt(
    original_data_filepath
).T
original_data_wavelengths = 0.5 * (original_data_wavelo + original_data_wavehi)
numpy_array_data_test_filepath: Path = (
    model_directory / "R1b_isothermal_spectrum_transmission.npy"
)
numpy_array_data_test = np.load(numpy_array_data_test_filepath)
original_data_flux = numpy_array_data_test

# axis.plot(
#    original_data_wavelengths,
#    original_data_flux,
#    color="cornflowerblue",
#    label="Original Data",
# )

axis.set_xlabel("Wavelength (microns)")
axis.set_ylabel("Transit depth")
# axis.set_yscale("log")
axis.legend(frameon=False, fontsize=16)

plt.savefig(output_file_directory / "test_plot_transmission.pdf", bbox_inches="tight")

figure, axis = plt.subplots(1, 1, figsize=(15, 10))

axis.plot(
    reference_model_wavelengths,
    observed_onestream_emission_fluxes_sampled_to_data,
    color="crimson",
    linewidth=4,
    label="Resampled 1-stream emission fluxes",
)

axis.plot(
    reference_model_wavelengths,
    observed_twostream_emission_fluxes_sampled_to_data,
    color="mediumseagreen",
    linewidth=6,
    label="Resampled 2-stream emission fluxes",
)

original_data_wavelengths = 0.5 * (original_data_wavelo + original_data_wavehi)
numpy_array_data_test_filepath: Path = model_directory / "R1b_spectrum_emission.npy"
numpy_array_data_test = np.load(numpy_array_data_test_filepath)
original_data_flux = numpy_array_data_test

axis.plot(
    original_data_wavelengths,
    original_data_flux,
    color="cornflowerblue",
    linewidth=2,
    label="Original Emission Data",
)

axis.set_xlabel("Wavelength (microns)")
axis.set_ylabel("Emission flux")
# axis.set_yscale("log")
axis.set_ylim(0, 1.1 * np.max(original_data_flux))
axis.legend(frameon=False, fontsize=16)

plt.savefig(output_file_directory / "test_plot_emission.pdf", bbox_inches="tight")
