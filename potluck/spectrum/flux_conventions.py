import astropy.units as u
import xarray as xr


def to_Flambda(
    wavelength_dataarray: xr.DataArray,
    data_array: xr.DataArray,
    Flambda_units: str = "erg/s/cm^2/AA",
) -> xr.DataArray:
    wavelength_unit: str = wavelength_dataarray.attrs["units"]

    unit_converted_data_array: xr.DataArray = (
        data_array.to_numpy() * getattr(u, data_array.units)
    ).to(
        Flambda_units,
        equivalencies=u.spectral_density(
            wav=wavelength_dataarray.values * getattr(u, wavelength_unit)
        ),
    )

    return xr.DataArray(
        data=unit_converted_data_array,
        dims=data_array.dims,
        coords=data_array.coords,
        attrs={
            "units": Flambda_units,
            "long_name": rf"$F_\lambda$ ({Flambda_units})",
        },
    )
