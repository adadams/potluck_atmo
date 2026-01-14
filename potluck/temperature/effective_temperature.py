import xarray as xr

from potluck.basic_types import TemperatureValue
from potluck.constants_and_conversions import STEFAN_BOLTZMANN_CONSTANT_IN_CGS
from potluck.spectrum.wavelength import calculate_spectrally_integrated_flux


def calculate_effective_temperature(
    emission_flux_density: xr.DataArray,
) -> TemperatureValue:
    # in F_lambda units (here assumes cgs for now: erg/s/cm^2/cm)

    bolometric_emission_flux: xr.DataArray = calculate_spectrally_integrated_flux(
        emission_flux_density
    ).sum("wavelength", skipna=True)

    effective_temperature: TemperatureValue = (
        (bolometric_emission_flux / STEFAN_BOLTZMANN_CONSTANT_IN_CGS) ** (1 / 4)
    ).item()

    return effective_temperature
