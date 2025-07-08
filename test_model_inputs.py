import xarray as xr

from temperature.models import PietteTemperatureModelInputs, generate_piette_model
from template_model_structs import (
    FundamentalParameterInputs,
    PressureProfileInputs,
    UniformGasChemistryInputs,
    build_default_fundamental_parameters,
    build_pressure_profile_from_log_pressures,
    build_temperature_profile,
    build_uniform_gas_chemistry,
)

fundamental_parameters: xr.Dataset = build_default_fundamental_parameters(
    FundamentalParameterInputs(
        planet_radius=7.776098627,
        planet_gravity=13613.2622,
        radius_units="Earth_radius",
        gravity_units="cm/s^2",
    )
)

pressure_profile: xr.Dataset = build_pressure_profile_from_log_pressures(
    PressureProfileInputs(
        shallowest_log10_pressure=-4.0,
        deepest_log10_pressure=2.5,
        number_of_levels=71,
        units="bar",
    )
)

temperature_profile: xr.Dataset = build_temperature_profile(
    temperature_model_constructor=generate_piette_model,
    temperature_model_inputs={
        "piette_parameters": PietteTemperatureModelInputs(
            photospheric_scaled_3bar_temperature=0.2889458091719745,
            scaled_1bar_temperature=0.11159102,
            scaled_0p1bar_temperature=0.02182628,
            scaled_0p01bar_temperature=0.12510834,
            scaled_0p001bar_temperature=0.10768672,
            scaled_0p0001bar_temperature=0.01539343,
            scaled_10bar_temperature=0.02514635,
            scaled_30bar_temperature=0.01982915,
            scaled_100bar_temperature=0.06249186,
            scaled_300bar_temperature=0.32445998,
        ),
        "lower_temperature_bound": 75.0,
        "upper_temperature_bound": 3975.0,
    },
    pressure_profile=pressure_profile,
)

gas_chemistry: xr.Dataset = build_uniform_gas_chemistry(
    UniformGasChemistryInputs(
        log_mixing_ratios={
            "h2o": -5.940043768,
            "co": -5.695578981,
            "co2": -8.884468544,
            "ch4": -7.663836048,
            "Lupu_alk": -4.953393893,
            "h2s": -11.42842546,
            "nh3": -10.14099491,
        },
        filler_species="h2",
    ),
    pressure_profile=pressure_profile,
)
