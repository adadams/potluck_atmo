import xarray as xr


def curate_crosssection_catalog(
    crosssection_catalog_dataset: xr.Dataset,
    temperatures_by_layer: xr.DataArray,
    pressures_by_layer: xr.DataArray,
    species_present_in_model: list[str],
):
    interpolated_crosssection_catalog: xr.Dataset = crosssection_catalog_dataset.interp(
        pressure=pressures_by_layer, temperature=temperatures_by_layer, method="linear"
    )

    interpolated_crosssection_catalog_with_model_species: xr.Dataset = (
        interpolated_crosssection_catalog.get(species_present_in_model)
    )

    interpolated_crosssection_catalog_as_dataarray: xr.DataArray = (
        interpolated_crosssection_catalog_with_model_species.to_array(
            dim="species", name="crosssections"
        ).sortby("species")
    )

    return interpolated_crosssection_catalog_as_dataarray


def compile_crosssection_data_for_forward_model(
    vertical_inputs_by_layer: xr.Dataset,
    crosssection_catalog_dataset: xr.Dataset,
) -> xr.Dataset:
    return curate_crosssection_catalog(
        crosssection_catalog_dataset=crosssection_catalog_dataset,
        temperatures_by_layer=vertical_inputs_by_layer.temperature,
        pressures_by_layer=vertical_inputs_by_layer.pressure,
        species_present_in_model=vertical_inputs_by_layer.species,
    )
