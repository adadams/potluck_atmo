import inspect
from collections.abc import Callable
from pathlib import Path

import msgspec

from model_builders import default_builders

current_directory: Path = Path(__file__).parent

test_model_inputs_filepath: Path = current_directory / "test_model_inputs.toml"

with open(test_model_inputs_filepath, "rb") as file:
    test_model_inputs: default_builders.FundamentalParameterInputs = (
        msgspec.toml.decode(
            file.read()  # , type=FundamentalParameterInputs
        )
    )

for (
    model_component_function_name,
    model_component_parameter_set,
) in test_model_inputs.items():
    model_component_function: Callable = getattr(
        default_builders, model_component_function_name
    )

    # get the type signature of the only argument of this function
    model_component_function_signature = inspect.signature(
        model_component_function
    ).parameters

    model_component_function_arguments_class: type = list(
        model_component_function_signature.values()
    )[0].annotation

    model_component_function_arguments = model_component_function_arguments_class(
        **model_component_parameter_set
    )
