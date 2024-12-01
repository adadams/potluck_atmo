import xarray as xr


def curate_crosssection_catalog(
    crosssection_catalog_dataset: xr.Dataset,
    temperatures_by_layer: xr.DataArray,
    pressures_by_layer: xr.DataArray,
    species_present_in_model: list[str],
):
    return (
        (
            crosssection_catalog_dataset.interp(
                temperature=temperatures_by_layer,
                pressure=pressures_by_layer,
            )
        )
        .get(species_present_in_model)
        .to_array(dim="species", name="crosssections")
    )


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
