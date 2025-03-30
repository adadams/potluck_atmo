from pathlib import Path

import xarray as xr
from matplotlib import pyplot as plt

from calculate_RT import calculate_observed_fluxes, resample_observed_fluxes
from user.input_importers import (
    UserForwardModelInputsPlusStuff,
    import_model_id,
    import_user_forward_model_inputs_plus_stuff,
)

model_directory_label: str = "example_isothermal"

current_directory: Path = Path.cwd()
user_directory: Path = current_directory / "user"
model_directory: Path = user_directory / f"{model_directory_label}_model"
intermediate_output_directory: Path = model_directory / "intermediate_outputs"
output_file_directory: Path = model_directory / "output_files"

plt.style.use(current_directory / "arthur.mplstyle")

user_model_inputs: UserForwardModelInputsPlusStuff = (
    import_user_forward_model_inputs_plus_stuff(
        model_directory_label=model_directory_label, parent_directory="user"
    )
)

model_id: str = import_model_id(
    model_directory_label=model_directory_label, parent_directory="user"
)

reference_model_wavelengths: xr.DataArray = xr.open_dataset(
    user_model_inputs.reference_model_filepath
).reference_wavelengths

observed_fluxes: dict[str, xr.DataArray] = calculate_observed_fluxes(user_model_inputs)

figure, axis = plt.subplots(1, 1, figsize=(15, 10))

# axis.plot(
#    observed_fluxes["observed_onestream_flux"].wavelength,
#    observed_fluxes["observed_onestream_flux"],
#    label="Observed fluxes (1-stream)",
# )
# axis.plot(
#    observed_fluxes["observed_twostream_flux"].wavelength,
#    observed_fluxes["observed_twostream_flux"],
#    label="Observed fluxes (2-stream)",
# )


observed_fluxes_sampled_to_data: xr.DataArray = resample_observed_fluxes(
    observed_fluxes,
    reference_model_wavelengths,
)

axis.plot(
    observed_fluxes_sampled_to_data["resampled_onestream_flux"].wavelength,
    observed_fluxes_sampled_to_data["resampled_onestream_flux"],
    label="Resampled fluxes (1-stream)",
)
axis.plot(
    observed_fluxes_sampled_to_data["resampled_twostream_flux"].wavelength,
    observed_fluxes_sampled_to_data["resampled_twostream_flux"],
    label="Resampled fluxes (2-stream)",
)

axis.set_xlabel("Wavelength (microns)")
axis.set_ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$)")
# axis.set_yscale("log")
axis.legend(frameon=False, fontsize=16)

plt.savefig(output_file_directory / "test_plot.pdf", bbox_inches="tight")
