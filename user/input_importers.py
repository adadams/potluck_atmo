from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

from user.input_structs import UserForwardModelInputs, UserVerticalModelInputs


def import_model_id(model_directory_label: str, parent_directory: str = "user"):
    return import_module(
        f"{parent_directory}.{model_directory_label}_model.input_files.{model_directory_label}_id"
    ).model_case_name


def import_user_vertical_inputs(
    model_directory_label: str, parent_directory: str = "user"
) -> UserVerticalModelInputs:
    return import_module(
        f"{parent_directory}.{model_directory_label}_model.input_files.{model_directory_label}_vertical_inputs"
    ).user_vertical_inputs


class UserForwardModelInputsPlusStuff(NamedTuple):
    user_forward_model_inputs: UserForwardModelInputs
    reference_model_filepath: Path
    vertical_structure_datatree_path: Path


def import_user_forward_model_inputs_plus_stuff(
    model_directory_label: str, parent_directory: str = "user"
) -> UserForwardModelInputsPlusStuff:
    forward_model_module: ModuleType = import_module(
        f"{parent_directory}.{model_directory_label}_model.input_files.{model_directory_label}_forward_model_inputs"
    )

    return UserForwardModelInputsPlusStuff(
        user_forward_model_inputs=forward_model_module.user_forward_model_inputs,
        reference_model_filepath=forward_model_module.reference_model_filepath,
        vertical_structure_datatree_path=forward_model_module.vertical_structure_datatree_path,
    )
