from collections.abc import Callable, Sequence
from inspect import signature
from pathlib import Path
from typing import Any

import pint
from msgspec import Struct

DEFAULT_UNITS_SYSTEM: str = "cgs"

ureg = pint.UnitRegistry(system=DEFAULT_UNITS_SYSTEM)
ureg.load_definitions(Path.joinpath(Path(__file__).parent, "additional_units.txt"))


def type_check_arguments(
    function: Callable, argument_types: Sequence[Struct], *args
) -> dict[str, Any]:
    proposed_function_arguments: dict[str, str] = signature(function).parameters
    proposed_function_argument_names: list[str] = list(
        proposed_function_arguments.keys()
    )

    function_arguments: dict[str, str] = {
        argument_name: (
            getattr(argument_value, argument_name)
            * ureg(getattr(argument_value, "units"))
        ).to_base_units()
        if hasattr(argument_value, "units")
        else getattr(argument_value, argument_name)
        for argument_name, argument_value in zip(proposed_function_argument_names, args)
    }

    return function_arguments


def call_with_type_checks(function: Callable, *args) -> Callable:
    return function(**type_check_arguments(function, *args))
