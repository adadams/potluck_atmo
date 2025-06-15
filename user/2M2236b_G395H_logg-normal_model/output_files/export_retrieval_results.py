from inspect import get_annotations
from pathlib import Path
from typing import TypedDict

import numpy as np
import xarray as xr


class ModelFreeParameters(TypedDict):
    planet_radius_relative_to_earth: float
    planet_log_gravity: float
    uniform_h2o_log_abundance: float
    uniform_co_log_abundance: float
    uniform_co2_log_abundance: float
    uniform_ch4_log_abundance: float
    uniform_Lupu_alk_log_abundance: float
    uniform_h2s_log_abundance: float
    uniform_nh3_log_abundance: float
    photospheric_scaled_temperature: float
    shallow_scaled_temperature_0: float
    shallow_scaled_temperature_1: float
    shallow_scaled_temperature_2: float
    shallow_scaled_temperature_3: float
    shallow_scaled_temperature_4: float
    deep_scaled_temperature_0: float
    deep_scaled_temperature_1: float
    deep_scaled_temperature_2: float
    deep_scaled_temperature_3: float
    flux_scaled_error_inflation_factor: float
    log10_constant_error_inflation_term: float
    NRS1_flux_scale_factor: float
    NRS2_flux_scale_factor: float


project_directory: Path = Path.cwd()
current_directory: Path = Path(__file__).parent

retrieval_run_name: str = "2M2236b_G395H_logg-normal"
run_date_tag: str = "2025Jun09_12:22:30"
retrieval_run_prefix: str = f"{retrieval_run_name}_{run_date_tag}"

sampled_points_filepath: Path = current_directory / f"{retrieval_run_prefix}_points.npy"
log_likelihoods_filepath: Path = current_directory / f"{retrieval_run_prefix}_log_l.npy"
log_weights_filepath: Path = current_directory / f"{retrieval_run_prefix}_log_w.npy"
evidences_filepath: Path = current_directory / f"{retrieval_run_prefix}_log_z.npy"

sampled_points: np.ndarray = np.load(sampled_points_filepath)
log_likelihoods: np.ndarray = np.load(log_likelihoods_filepath)
log_weights: np.ndarray = np.load(log_weights_filepath)
evidence: float = float(np.load(evidences_filepath))

sample_coordinate: xr.DataArray = xr.DataArray(
    data=np.arange(len(sampled_points)),
    dims=("sample",),
    coords={"sample": np.arange(len(sampled_points))},
    name="sample",
)

sampled_points_by_parameter: dict = {
    parameter_name: {
        "dims": ("sample",),
        "data": parameter_samples.T,
        "name": get_annotations(ModelFreeParameters).keys(),
    }
    for parameter_name, parameter_samples in zip(
        get_annotations(ModelFreeParameters).keys(),
        sampled_points.T,
    )
}

log_likelihood_dataarray: xr.DataArray = xr.DataArray(
    data=log_likelihoods,
    dims=("sample",),
    coords={"sample": sample_coordinate},
    name="log_likelihood",
)

log_weight_dataarray: xr.DataArray = xr.DataArray(
    data=log_weights,
    dims=("sample",),
    coords={"sample": sample_coordinate},
    name="log_weight",
)

sample_dataset: xr.Dataset = xr.Dataset.from_dict(
    {
        "data_vars": sampled_points_by_parameter,
        "coords": {"sample": sample_coordinate.to_dict()},
        "dims": {"sample": "sample"},
        "attrs": {"evidence": evidence, "earth_radius_in_cm": 6.371e8},
    }
)

results_dataset: xr.Dataset = xr.merge(
    [sample_dataset, log_likelihood_dataarray, log_weight_dataarray]
)

results_dataset.to_netcdf(current_directory / f"{retrieval_run_prefix}_results.nc")

MLE_dataset: xr.Dataset = results_dataset.isel(
    sample=np.argmax(log_likelihoods)
).drop_vars(["log_likelihood", "log_weight"])
print(f"{MLE_dataset=}")

MLE_dataset.to_netcdf(current_directory / f"{retrieval_run_prefix}_MLE.nc")
