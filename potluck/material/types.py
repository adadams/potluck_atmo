from dataclasses import dataclass

import numpy as np

from potluck.material.scattering.types import TwoStreamScatteringCoefficients


@dataclass
class TwoStreamMaterial:
    name: str
    scattering_coefficients: TwoStreamScatteringCoefficients
    absorption_coefficients: np.ndarray[np.float64]


# NOTE: these are meant as a cumulative struct, encompassing contributions from multiple materials.
@dataclass
class TwoStreamScatteringParameters:
    scattering_asymmetry_parameter: np.ndarray[np.float64]  # g
    single_scattering_albedo: np.ndarray[np.float64]  # w0


@dataclass
class TwoStreamParameters:
    scattering_asymmetry_parameter: np.ndarray[np.float64]  # g
    single_scattering_albedo: np.ndarray[np.float64]  # w0
    optical_depth: np.ndarray[np.float64]  # tau
