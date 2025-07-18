import xarray as xr


def curate_crosssection_catalog(
    crosssection_catalog: xr.Dataset,
    temperatures_by_layer: xr.DataArray,
    pressures_by_layer: xr.DataArray,
    species_present_in_model: list[str],
):
    interpolated_crosssection_catalog: xr.Dataset = crosssection_catalog.interp(
        pressure=pressures_by_layer, temperature=temperatures_by_layer, method="linear"
    )

    # a "trick" to make sure the species are in the same order
    interpolated_crosssection_catalog_with_model_species: xr.Dataset = (
        interpolated_crosssection_catalog.get(species_present_in_model)
    )

    return interpolated_crosssection_catalog_with_model_species


def compile_crosssection_data_for_forward_model(
    vertical_inputs_by_layer: xr.Dataset,
    crosssection_catalog: xr.Dataset,
) -> xr.Dataset:
    return curate_crosssection_catalog(
        crosssection_catalog=crosssection_catalog,
        temperatures_by_layer=vertical_inputs_by_layer.temperature,
        pressures_by_layer=vertical_inputs_by_layer.pressure,
        species_present_in_model=vertical_inputs_by_layer.species,
    )
