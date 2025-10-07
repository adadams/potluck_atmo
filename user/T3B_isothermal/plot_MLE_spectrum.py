from inspect import get_annotations
from pathlib import Path

import msgspec
import xarray as xr
from build_model import ModelInputs, evaluate_transmission_spectrum
from matplotlib import pyplot as plt
from potluck.model_statistics.calculate_statistics import (
    calculate_reduced_chi_squared_statistic,
)
from potluck.xarray_functional_wrappers import convert_units
from run_retrieval import ModelFreeParameters

current_directory: Path = Path(__file__).parent
project_directory: Path = Path.cwd() / "potluck"

number_of_free_parameters: int = len(get_annotations(ModelFreeParameters))

plt.style.use(project_directory / "arthur.mplstyle")

if __name__ == "__main__":
    MLE_inputs_as_toml_filepath: Path = (
        current_directory
        / "RISOTTO_T3B_Full_higher_pressure_2025-09-25_MLE_inputs.toml"
    )

    with open(MLE_inputs_as_toml_filepath, "rb") as MLE_inputs_file:
        MLE_inputs: ModelInputs = msgspec.toml.decode(
            MLE_inputs_file.read(), type=ModelInputs
        )

    MLE_dataset_filepath: Path = (
        current_directory / "RISOTTO_T3B_Full_higher_pressure_2025-09-25_MLE.nc"
    )
    MLE_dataset: xr.Dataset = xr.open_dataset(MLE_dataset_filepath)

    run_name: str = (
        f"{MLE_inputs.metadata['model_ID']}_{MLE_inputs.metadata['run_date']}"
    )

    transmission_spectrum_in_cgs: xr.DataArray = evaluate_transmission_spectrum(
        MLE_inputs, resampling_fwhm_fraction=200.0
    )
    transmission_spectrum: xr.DataArray = convert_units(
        transmission_spectrum_in_cgs, {"wavelength": "micron"}
    )

    Planet0_data_filepath: Path = current_directory / "T3B_risotto_Full_data.nc"
    Planet0_data: xr.Dataset = xr.open_dataset(Planet0_data_filepath).assign_coords(
        wavelength=transmission_spectrum.wavelength
    )

    figure, axis = plt.subplots(figsize=(20, 5))

    reduced_chi_squared_statistic: float = calculate_reduced_chi_squared_statistic(
        transmission_spectrum,
        Planet0_data.transit_depth,
        Planet0_data.transit_depth_uncertainty,
        number_of_free_parameters,
    )
    print(
        f"{run_name} reduced chi-squared statistic: {reduced_chi_squared_statistic.item()}"
    )

    axis.errorbar(
        Planet0_data.wavelength,
        Planet0_data.transit_depth,
        yerr=Planet0_data.transit_depth_uncertainty,
        color="black",
        fmt="o",
        elinewidth=2,
        capsize=4,
    )

    axis.plot(
        transmission_spectrum.wavelength,
        transmission_spectrum,
        color="crimson",
        linewidth=2,
        label=run_name,
        zorder=10,
    )

    axis.set_xlabel("Wavelength (microns)")
    axis.set_ylabel("Transit depth")
    axis.text(
        0.95,
        0.05,
        run_name.replace("_", " "),
        color="crimson",
        fontsize=28,
        ha="right",
        va="bottom",
        transform=axis.transAxes,
    )

    # axis.text(
    #    0.95,
    #    0.25,
    #    f"Flux error scaling factor: "
    #    f"{MLE_dataset.flux_scaled_error_inflation_factor.item() * 100:.1f}\%",
    #    color="black",
    #    fontsize=19,
    #    ha="right",
    #    va="bottom",
    #    transform=axis.transAxes,
    # )

    plt.savefig(current_directory / f"{run_name}_MLE_spectra.pdf", bbox_inches="tight")
