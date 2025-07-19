from pathlib import Path

import xarray as xr
from matplotlib import pyplot as plt

from evaluate_R1a_base_model import (
    prepare_model_for_likelihood_evaluation as evaluate_R1a_model,
)
from evaluate_R1b_base_model import (
    prepare_model_for_likelihood_evaluation as evaluate_R1b_model,
)
from evaluate_R1c_base_model import (
    prepare_model_for_likelihood_evaluation as evaluate_R1c_model,
)
from model_statistics.calculate_statistics import (
    calculate_reduced_chi_squared_statistic,
)

project_directory: Path = Path.cwd()
user_directory: Path = project_directory / "user"

figure, (R1a_axis, R1b_axis, R1c_axis) = plt.subplots(3, 1, figsize=(15, 15))
plot_colors = ["mediumseagreen", "cornflowerblue", "crimson"]

for run_name, model_function, axis, plot_color in zip(
    ["R1a", "R1b", "R1c"],
    [
        evaluate_R1a_model,
        evaluate_R1b_model,
        evaluate_R1c_model,
    ],
    [R1a_axis, R1b_axis, R1c_axis],
    plot_colors,
):
    print(f"{run_name=}")
    output_file_directory: Path = (
        user_directory / f"{run_name}_retrieval_model" / "output_files"
    )

    MLE_parameter_filepath: Path = (
        output_file_directory / f"{run_name}_retrieval_MLE.nc"
    )

    MLE_parameters: xr.Dataset = xr.open_dataset(MLE_parameter_filepath)
    print(f"{MLE_parameters=}")

    MLE_parameters_as_inputs: dict[str, float] = {
        parameter_name: parameter_value["data"]
        for parameter_name, parameter_value in MLE_parameters.to_dict()[
            "data_vars"
        ].items()
    }

    model_dataset: xr.Dataset = model_function(MLE_parameters_as_inputs)

    # calculate reduced chi-squared statistic
    number_of_free_parameters: int = len(MLE_parameters.data_vars)

    reduced_chi_squared_statistic: float = calculate_reduced_chi_squared_statistic(
        model_dataset.transit_data,
        model_dataset.scaled_transit_data_error,
        model_dataset.transit_model,
        number_of_free_parameters,
    )
    print(
        f"{run_name} reduced chi-squared statistic: {reduced_chi_squared_statistic.item()}"
    )

    axis.errorbar(
        model_dataset.wavelength,
        model_dataset.transit_data,
        yerr=model_dataset.scaled_transit_data_error,
        color="black",
        fmt="o",
        elinewidth=2,
        capsize=4,
    )

    axis.plot(
        model_dataset.wavelength,
        model_dataset.transit_model,
        color=plot_color,
        linewidth=2,
        label=run_name,
        zorder=10,
    )

    axis.set_xlabel("Wavelength (microns)")
    axis.set_ylabel("Transit depth")
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

    axis.text(
        0.95,
        0.75,
        f"Flux error scaling factor: "
        f"{MLE_parameters.flux_scaled_error_inflation_factor.item() * 100:.1f}\%",
        color="black",
        fontsize=19,
        ha="right",
        va="top",
        transform=axis.transAxes,
    )

plt.savefig(user_directory / "R1_MLE_spectra.pdf", bbox_inches="tight")
