import xarray as xr


def correct_data_for_flux_calibration(
    data: xr.Dataset, flux_calibration_factors: dict[str, float]
) -> xr.Dataset:
    for spectral_group, flux_calibration_factor in flux_calibration_factors.items():
        data["emission_flux"] = xr.where(
            data.spectral_group == spectral_group,
            data.emission_flux * (1 + flux_calibration_factor),
            data.emission_flux,
        )

    return data
