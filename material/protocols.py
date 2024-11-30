from typing import Protocol

from material.types import TwoStreamMaterial


class MaterialFunction(Protocol):
    def __call__(self, *args, **kwargs) -> TwoStreamMaterial: ...


class GasFunction(MaterialFunction): ...


class CloudFunction(MaterialFunction): ...
