from typing import Any
from warnings import warn

import msgspec


def check_if_all_headers_match(headers: tuple[tuple[Any, ...], ...]):
    set_of_unique_headers: set[tuple[Any, ...]] = set(
        [msgspec.structs.astuple(header) for header in headers]
    )

    if len(set_of_unique_headers) > 1:
        warn(f"Headers do not all match. Unique headers found: {set_of_unique_headers}")

    return set_of_unique_headers
