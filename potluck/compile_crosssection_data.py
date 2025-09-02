import xarray as xr


def curate_gas_crosssection_catalog(
    crosssection_catalog: xr.Dataset,
    temperatures_by_layer: xr.DataArray,
    pressures_by_layer: xr.DataArray,
    species_present_in_model: list[str],
):
    # a "trick" to make sure the species are in the same order
    crosssection_catalog_with_model_species: xr.Dataset = crosssection_catalog.get(
        species_present_in_model
    )

    interpolated_crosssection_catalog: xr.Dataset = (
        crosssection_catalog_with_model_species.interp(
            pressure=pressures_by_layer,
            temperature=temperatures_by_layer,
            method="linear",
        )
    )

    return interpolated_crosssection_catalog


def compile_crosssection_data_for_forward_model(
    vertical_inputs_by_layer: xr.Dataset,
    crosssection_catalog: xr.Dataset,
) -> xr.Dataset:
    return curate_gas_crosssection_catalog(
        crosssection_catalog=crosssection_catalog,
        temperatures_by_layer=vertical_inputs_by_layer.temperature,
        pressures_by_layer=vertical_inputs_by_layer.pressure,
        species_present_in_model=vertical_inputs_by_layer.species,
    )


def curate_cloud_crosssection_catalog(
    crosssection_catalog: xr.Dataset, species_present_in_model: list[str]
):
    # a "trick" to make sure the species are in the same order
    crosssection_catalog_with_model_species: xr.Dataset = crosssection_catalog.filter(
        lambda species: species.name in species_present_in_model
    )

    return crosssection_catalog_with_model_species
