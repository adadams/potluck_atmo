from collections.abc import Callable
from pathlib import Path
from typing import Any

import xarray as xr
from matplotlib import pyplot as plt

from model_statistics.calculate_statistics import (
    calculate_reduced_chi_squared_statistic,
)
from run_2M2236b_G395H_loggnormal_retrieval import (
    prepare_model_for_likelihood_evaluation as evaluate_2M2236_JWST_loggnormal_model,
)

project_directory: Path = Path.cwd()
user_directory: Path = project_directory / "user"

run_names: list[str] = ["2M2236b_G395H_logg-normal"]
run_times: list[str] = ["2025Jun09_12:22:30"]
run_prefixes: list[str] = [
    f"{run_name}_{run_time}" for run_name, run_time in zip(run_names, run_times)
]
run_model_spectrum_functions: list[Callable[[Any], xr.Dataset]] = [
    evaluate_2M2236_JWST_loggnormal_model
]
plot_colors: list[str] = ["darkorange"]

figure, (JWST_loggnormal_axis) = plt.subplots(
    len(run_prefixes), 1, figsize=(15, 5 * len(run_prefixes))
)

for run_name, run_prefix, model_function, axis, plot_color in zip(
    run_names,
    run_prefixes,
    run_model_spectrum_functions,
    [JWST_loggnormal_axis],
    plot_colors,
):
    print(f"{run_prefix=}")
    input_file_directory: Path = user_directory / f"{run_name}_model" / "input_files"
    output_file_directory: Path = user_directory / f"{run_name}_model" / "output_files"

    apollo_model_filepath: Path = (
        input_file_directory / "2M2236b_G395H_logg-normal_apollo_model.nc"
    )
    apollo_model: xr.Dataset = xr.open_dataset(apollo_model_filepath)

    axis.plot(
        apollo_model.wavelength,
        apollo_model.flux,
        label="Apollo model from original Apollo-fit parameters",
        color="dodgerblue",
    )

    MLE_parameter_filepath: Path = output_file_directory / f"{run_prefix}_MLE.nc"

    MLE_parameters: xr.Dataset = xr.open_dataset(MLE_parameter_filepath)
    print(f"{MLE_parameters.log10_constant_error_inflation_term.item()=}")

    MLE_parameters_as_inputs: dict[str, float] = {
        parameter_name: parameter_value["data"]
        for parameter_name, parameter_value in MLE_parameters.to_dict()[
            "data_vars"
        ].items()
    }

    MLE_parameters_as_inputs["fraction_of_reddest_fwhm_to_convolve_with"] = 1.00

    model_dataset: xr.Dataset = model_function(MLE_parameters_as_inputs)
    print(f"{model_dataset=}")

    # calculate reduced chi-squared statistic
    number_of_free_parameters: int = len(MLE_parameters.data_vars)

    reduced_chi_squared_statistic: float = calculate_reduced_chi_squared_statistic(
        model_dataset.emission_data,
        model_dataset.scaled_emission_data_error,
        model_dataset.emission_model,
        number_of_free_parameters,
    )
    print(
        f"{run_name} reduced chi-squared statistic: {reduced_chi_squared_statistic.item()}"
    )

    axis.errorbar(
        model_dataset.wavelength,
        model_dataset.emission_data,
        yerr=model_dataset.scaled_emission_data_error,
        color="black",
        fmt="o",
        elinewidth=2,
        capsize=4,
    )

    axis.plot(
        model_dataset.wavelength,
        model_dataset.emission_model,
        color=plot_color,
        linewidth=2,
        label=run_name,
        zorder=10,
    )

    axis.set_xlabel("Wavelength (microns)")
    axis.set_ylabel("emission depth")
    axis.text(
        0.95,
        0.95,
        run_name,
        color=plot_color,
        fontsize=28,
        ha="right",
        va="top",
        transform=axis.transAxes,
    )

    # axis.text(
    #    0.95,
    #    0.75,
    #    f"Flux error scaling factor: "
    #    f"{MLE_parameters.flux_scaled_error_inflation_factor.item() * 100:.1f}\%",
    #    color="black",
    #    fontsize=19,
    #    ha="right",
    #    va="top",
    #    transform=axis.transAxes,
    # )

plt.savefig(user_directory / "2M2236b_MLE_spectra_conv3.00.pdf", bbox_inches="tight")
