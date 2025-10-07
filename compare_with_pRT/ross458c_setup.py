from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Final, TypeAlias

import msgspec
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, MultipleLocator
from petitRADTRANS.plotlib import plot_opacity_contributions
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.spectral_model import SpectralModel

pRT_directory: Path = Path(__file__).parent
analysis_code_directory: Path = pRT_directory.parent
project_directory: Path = analysis_code_directory.parent
dataset_directory: Path = project_directory / "datasets"
plots_directory: Path = project_directory / "plots"

plt.style.use(project_directory / "arthur.mplstyle")

ESTIMATED_COMPANION_RADIUS_VS_SOLAR: Final[float] = 0.14376
ESTIMATED_DISTANCE_TO_SYSTEM_IN_PARSECS: Final[float] = 64.5

LIST_OF_SPECIES: list[str] = ["H2O", "CO", "CO2", "CH4", "K", "H2S", "NH3"]


class FundamentalParameters(msgspec.Struct):
    radius: float
    log10_gravity: float


class ChemistryParameters(msgspec.Struct):
    metallicity: float
    CtoO_ratio: float
    log10_quenching_pressure: float


MixingRatio = Annotated[float, msgspec.Meta(le=0)]
CoveringFraction = Annotated[float, msgspec.Meta(ge=0, le=1)]


class CloudSpeciesParameters(msgspec.Struct):
    f_sed: float
    log10_base_mass_fraction: MixingRatio


class CloudComponentParameters(msgspec.Struct):
    covering_fraction: CoveringFraction
    components: dict[str, CloudSpeciesParameters]
    log10_Kzz: float
    sigma_g: float


CloudParameters: TypeAlias = tuple[CloudComponentParameters, ...]


class pRTParameters(msgspec.Struct):
    fundamental_parameters: FundamentalParameters
    chemistry_parameters: ChemistryParameters
    cloud_parameters: CloudParameters


@dataclass
class pRTSpectrum:
    wavelengths: np.ndarray
    flux: np.ndarray
    additional_returned_quantities: dict


@dataclass
class pRTFreeParameters:
    pass


def read_parameters(
    filepath: Path,
) -> tuple[FundamentalParameters, ChemistryParameters, CloudComponentParameters]:
    with open(filepath, "rb") as file:
        return msgspec.toml.decode(
            file.read(),
            type=pRTParameters,
        )


def convert_surface_quantity_to_observed_quantity(
    radius_in_solar_radii: float = ESTIMATED_COMPANION_RADIUS_VS_SOLAR,
    distance_in_parsecs: float = ESTIMATED_DISTANCE_TO_SYSTEM_IN_PARSECS,
):
    distance_in_solar_radii: float = distance_in_parsecs * 4.435e7

    return (radius_in_solar_radii / distance_in_solar_radii) ** 2


def setup_initial_pRT_atmosphere(
    pressures: np.ndarray,
    line_species: list[str] = LIST_OF_SPECIES,
    wavelength_boundaries: tuple[float, float] = (0.80, 3.15),
    **other_kwargs,
) -> Radtrans:
    # Load scattering version of pRT

    return Radtrans(
        pressures=pressures,
        line_species=line_species,
        # rayleigh_species=["H2", "He"],
        gas_continuum_contributors=["H2-H2", "H2-He"],
        cloud_species=[
            # "Mg2SiO4(s)_amorphous__Mie",
            # "Fe(s)_crystalline__Mie",
            "Na2S(s)_crystalline__Mie",
        ],
        line_opacity_mode="lbl",
        wavelength_boundaries=wavelength_boundaries,
        **other_kwargs,
        # scattering_in_emission=True,
    )


def get_mass_fractions_from_dataset(
    mass_fraction_dataset: xr.Dataset,
) -> dict[str, np.ndarray]:
    mass_fractions: dict[str, np.ndarray] = {
        absorber: mass_fraction_dataset[absorber].values
        for absorber in mass_fraction_dataset
    }

    if "CO-NatAbund" in mass_fractions:
        mass_fractions["CO"] = mass_fractions.pop("CO-NatAbund")

    return mass_fractions


def generate_pRT_spectrum_from_inputs(
    pRT_atmosphere: Radtrans,
    fundamental_parameters: FundamentalParameters,
    cloud_f_sed: dict[str, float],
    temperature: np.ndarray,
    mass_fraction_dataset: xr.Dataset,
) -> Callable[[pRTFreeParameters], pRTSpectrum]:
    temperature: np.ndarray = TP_dataset.temperature.values

    reference_gravity = 10**fundamental_parameters.log10_gravity

    mass_fractions: dict[str, np.ndarray] = get_mass_fractions_from_dataset(
        mass_fraction_dataset
    )

    mean_molar_masses = mass_fraction_dataset["mean_molecular_weight"].values

    return partial(
        pRT_atmosphere.calculate_flux,
        temperatures=temperature,
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=reference_gravity,
        cloud_f_sed=cloud_f_sed,
        return_contribution=True,
    )
