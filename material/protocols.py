from typing import Protocol

from material.scattering.types import TwoStreamMaterial


class MaterialFunction(Protocol):
    def __call__(self) -> TwoStreamMaterial: ...


class GasFunction(MaterialFunction): ...


class CloudFunction(MaterialFunction): ...
