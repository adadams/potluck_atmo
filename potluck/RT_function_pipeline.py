from collections.abc import Callable

# interpolation_function(s): (1) take datasets/arrays defined on pressure levels and return datasets/arrays defined on pressure layers, (2)
# thermal_intensity_function(wavelength_grid_in_cm, temperature_grid_in_K) -> thermal_intensity, delta_thermal_intensity
# RT_function(RT_inputs: [
#   wavelengths_in_cm,
#   thermal_intensity,
#   delta_thermal_intensity,
#   crosssections_interpolated_to_TP_profile_pressures_and_temperatures,
#   number_density_with_species_as_dim_interpolated_to_pressure_layers,
#   path_length_by_layer_in_cm
#   ]) -> spectral_intensity

# NOTE: can this function be the one that is wrapped with the xarray decorator?
# It defines the function pipeline, outlines the dimensions via types of the arguments and results,
# and accounts for those for the ultimate result.
# I think the type annotation system should outline this for the user, and then the xarray decorator can
# actually do the application on the macro level.


def RT_function_pipeline(RT_function: Callable) -> ...: ...
