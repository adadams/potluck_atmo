from inspect import get_annotations
from pathlib import Path

import msgspec
import numpy as np
import xarray as xr
from build_model import ModelInputs
from run_retrieval import (
    ModelFreeParameters,
    replace_inputs_with_free_parameters,
)

project_directory: Path = Path.cwd() / "potluck"
current_directory: Path = Path(__file__).parent

retrieval_run_name: str = "RISOTTO_T3B_Full"
run_date_tag: str = "2025-09-24"
retrieval_run_prefix: str = f"{retrieval_run_name}_{run_date_tag}"

sampled_points_filepath: Path = current_directory / f"{retrieval_run_prefix}_points.npy"
log_likelihoods_filepath: Path = current_directory / f"{retrieval_run_prefix}_log_l.npy"
log_weights_filepath: Path = current_directory / f"{retrieval_run_prefix}_log_w.npy"
evidences_filepath: Path = current_directory / f"{retrieval_run_prefix}_log_z.npy"

if __name__ == "__main__":
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

    MLE_as_free_parameter_dict: ModelFreeParameters = ModelFreeParameters(
        **{
            parameter_name: float(MLE_dataset[parameter_name])
            for parameter_name in get_annotations(ModelFreeParameters).keys()
        }
    )

    original_input_toml_filepath: Path = current_directory / "model_inputs.toml"

    with open(original_input_toml_filepath, "rb") as input_toml_file:
        inputs_from_toml_file: dict = msgspec.toml.decode(
            input_toml_file.read(), type=ModelInputs
        )

    MLE_inputs: ModelInputs = replace_inputs_with_free_parameters(
        inputs_from_toml_file, MLE_as_free_parameter_dict
    )

    with open(
        current_directory / f"{retrieval_run_prefix}_MLE_inputs.toml", "wb"
    ) as MLE_inputs_file:
        MLE_inputs_file.write(msgspec.toml.encode(MLE_inputs))

    MLE_dataset.to_netcdf(current_directory / f"{retrieval_run_prefix}_MLE.nc")
