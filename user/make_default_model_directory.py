from pathlib import Path
from shutil import copyfile

current_directory: Path = Path(__file__).parent

DEFAULT_MODEL_DIRECTORY_FOLDER_NAMES: list[str] = [
    "input_files",
    "intermediate_outputs",
    "output_files",
    "serial_outputs",
]

ID_FILE_TEMPLATE: str = '''model_case_name: str = "{model_case_name}"'''


def make_default_model_directory(
    model_directory_label: str, parent_directory: Path = current_directory
) -> None:
    model_directory: Path = parent_directory / (model_directory_label + "_model")

    model_directory.mkdir(exist_ok=True)

    for folder_name in DEFAULT_MODEL_DIRECTORY_FOLDER_NAMES:
        folder_path: Path = model_directory / folder_name
        folder_path.mkdir(exist_ok=True)

    return


def make_default_input_file_templates(
    model_directory_label: str,
    model_case_name: str = None,
    parent_directory: Path = current_directory,
    template_model_name: str = "example_isothermal",
) -> None:
    if model_case_name is None:
        model_case_name = model_directory_label

    input_directory_name: str = "input_files"

    model_directory: Path = parent_directory / (model_directory_label + "_model")
    model_inputs_directory: Path = (
        parent_directory / model_directory / input_directory_name
    )

    template_model_directory: Path = parent_directory / (template_model_name + "_model")
    template_model_inputs_directory: Path = (
        parent_directory / template_model_directory / input_directory_name
    )

    if not model_inputs_directory.exists():
        model_inputs_directory.mkdir()

    id_filename: str = f"{model_directory_label}_id.py"

    (model_inputs_directory / id_filename).write_text(
        ID_FILE_TEMPLATE.format(model_case_name=model_case_name)
    )

    model_filename_templates: list[str] = [
        "{model_name}_vertical_inputs.py",
        "{model_name}_forward_model_inputs.py",
    ]

    templated_model_filenames: list[str] = [
        filename.format(model_name=template_model_name)
        for filename in model_filename_templates
    ]

    new_model_filenames: list[str] = [
        filename.format(model_name=model_directory_label)
        for filename in model_filename_templates
    ]

    for template_filename, new_filename in zip(
        templated_model_filenames, new_model_filenames
    ):
        copyfile(
            src=template_model_inputs_directory / template_filename,
            dst=model_inputs_directory / new_filename,
        )

    return


if __name__ == "__main__":
    model_directory_label: str = "test_almost_isothermal"
    model_case_name: str = "isothermal_1300K_1ppt_H20_1ppt_CO"

    make_default_model_directory(model_directory_label=model_directory_label)
    make_default_input_file_templates(
        model_directory_label=model_directory_label, model_case_name=model_case_name
    )
