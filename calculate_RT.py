from pathlib import Path
from typing import Final

import msgspec
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from compile_crosssection_data import curate_crosssection_catalog
from compile_thermal_structure import compile_thermal_structure_for_forward_model
from compile_vertical_structure import (
    ForwardModelXarrayInputs,
    compile_vertical_structure_for_forward_model,
)
from constants_and_conversions import MICRONS_TO_CM
from material.two_stream import compile_composite_two_stream_parameters
from material.types import TwoStreamParameters
from radiative_transfer.RT_one_stream import calculate_spectral_intensity_at_surface
from radiative_transfer.RT_Toon1989 import RT_Toon1989, RTToon1989Inputs
from spectrum.bin import resample_spectral_quantity_to_new_wavelengths
from spectrum.observe import convert_surface_quantity_to_observed_quantity
from user.input_importers import (
    UserForwardModelInputsPlusStuff,
    import_model_id,
    import_user_forward_model_inputs_plus_stuff,
)
from xarray_functional_wrappers import (
    XarrayOutputs,
    rename_and_unitize,
    save_xarray_outputs_to_file,
)

current_directory: Path = Path(__file__).parent

plt.style.use(Path.cwd() / "arthur.mplstyle")


def calculate_observed_fluxes(
    user_model_inputs: UserForwardModelInputsPlusStuff,
) -> XarrayOutputs:
    compiled_vertical_dataset: ForwardModelXarrayInputs = compile_vertical_structure_for_forward_model(
        user_vertical_inputs=user_model_inputs.user_forward_model_inputs.vertical_inputs
    )
    vertical_structure_dataset_by_level: xr.Dataset = compiled_vertical_dataset.by_level
    vertical_structure_dataset_by_layer: xr.Dataset = compiled_vertical_dataset.by_layer

    temperatures_by_level: xr.DataArray = (
        vertical_structure_dataset_by_level.temperature
    )
    temperatures_by_layer: xr.DataArray = (
        vertical_structure_dataset_by_layer.temperature
    )
    pressures_by_layer: xr.DataArray = vertical_structure_dataset_by_layer.pressure

    number_densities_by_layer: xr.DataArray = vertical_structure_dataset_by_layer[
        "number_density"
    ]

    species_present_in_model: list[str] = number_densities_by_layer.species.values

    crosssection_catalog_dataset_interpolated_to_model: xr.Dataset = curate_crosssection_catalog(
        crosssection_catalog_dataset=user_model_inputs.user_forward_model_inputs.crosssection_catalog,
        temperatures_by_layer=temperatures_by_layer,
        pressures_by_layer=pressures_by_layer,
        species_present_in_model=species_present_in_model,
    )

    model_wavelengths_in_microns: xr.DataArray = (
        crosssection_catalog_dataset_interpolated_to_model.wavelength
    )
    model_wavelengths_in_cm = model_wavelengths_in_microns * MICRONS_TO_CM

    thermal_intensities_by_layer: xr.Dataset = (
        compile_thermal_structure_for_forward_model(
            temperatures_by_level=temperatures_by_level,
            pressures_by_layer=pressures_by_layer,
            model_wavelengths_in_microns=model_wavelengths_in_microns,
        )
    )

    two_stream_parameters: TwoStreamParameters = compile_composite_two_stream_parameters(
        wavelengths_in_cm=model_wavelengths_in_cm,
        crosssections=crosssection_catalog_dataset_interpolated_to_model,
        number_density=number_densities_by_layer,
        path_lengths=user_model_inputs.user_forward_model_inputs.path_lengths_by_level,
    )

    RT_Toon1989_inputs: RTToon1989Inputs = RTToon1989Inputs(
        thermal_intensity=thermal_intensities_by_layer.thermal_intensity,
        delta_thermal_intensity=thermal_intensities_by_layer.delta_thermal_intensity,
        **msgspec.structs.asdict(two_stream_parameters),
    )

    RT_Toon1989_inputs.optical_depth.rename("optical_depth").to_netcdf(
        output_file_directory / "test_RT_inputs_optical_depth.nc"
    )

    emitted_twostream_flux: xr.DataArray = rename_and_unitize(
        RT_Toon1989(*RT_Toon1989_inputs[:-2]),
        name="emitted_flux",
        units="erg s^-1 cm^-3",
    )

    cumulative_optical_depth: xr.DataArray = (
        RT_Toon1989_inputs.optical_depth.cumulative("pressure").sum()
    )

    emitted_onestream_flux: xr.DataArray = rename_and_unitize(
        calculate_spectral_intensity_at_surface(
            thermal_intensities_by_layer.thermal_intensity,
            cumulative_optical_depth,
        ),
        name="emitted_flux",
        units="erg s^-1 cm^-3",
    )

    observed_onestream_flux: xr.DataArray = (
        emitted_onestream_flux
        * (
            user_model_inputs.user_forward_model_inputs.vertical_inputs.planet_radius_in_cm
            / user_model_inputs.user_forward_model_inputs.distance_to_system_in_cm
        )
        ** 2
    ).rename("observed_flux")

    observed_twostream_flux: xr.DataArray = (
        emitted_twostream_flux
        * (
            user_model_inputs.user_forward_model_inputs.vertical_inputs.planet_radius_in_cm
            / user_model_inputs.user_forward_model_inputs.distance_to_system_in_cm
        )
        ** 2
    ).rename("observed_flux")

    return {
        "observed_onestream_flux": observed_onestream_flux,
        "observed_twostream_flux": observed_twostream_flux,
    }


@save_xarray_outputs_to_file
def resample_observed_fluxes(
    observed_fluxes: dict[str, xr.DataArray],
    reference_model_wavelengths: xr.DataArray,
) -> dict[str, xr.DataArray]:
    return {
        observed_flux_name.replace(
            "observed", "resampled"
        ): resample_spectral_quantity_to_new_wavelengths(
            reference_model_wavelengths,
            observed_flux.wavelength,
            observed_flux,
        )
        for observed_flux_name, observed_flux in observed_fluxes.items()
    }


if __name__ == "__main__":
    model_directory_label: str = "test_almost_isothermal"

    current_directory: Path = Path(__file__).parent
    user_directory: Path = current_directory / "user"
    model_directory: Path = user_directory / f"{model_directory_label}_model"
    intermediate_output_directory: Path = model_directory / "intermediate_outputs"
    output_file_directory: Path = model_directory / "output_files"

    user_model_inputs: UserForwardModelInputsPlusStuff = (
        import_user_forward_model_inputs_plus_stuff(
            model_directory_label=model_directory_label, parent_directory="user"
        )
    )

    model_id: str = import_model_id(
        model_directory_label=model_directory_label, parent_directory="user"
    )

    observed_fluxes: dict[str, xr.DataArray] = calculate_observed_fluxes(
        user_model_inputs
    )

    reference_model: xr.Dataset = xr.open_dataset(
        user_model_inputs.reference_model_filepath
    )
    reference_model_wavelengths: xr.DataArray = reference_model.wavelength

    resampled_fluxes: dict[str, xr.DataArray] = resample_observed_fluxes(
        observed_fluxes,
        reference_model_wavelengths,
        output_directory=output_file_directory,
        filename_prefix=model_id,
    )
    resampled_onestream_flux = resampled_fluxes["resampled_onestream_flux"]
    resampled_twostream_flux = resampled_fluxes["resampled_twostream_flux"]

    apollo_model_spectrum_filepath: Path = (
        model_directory / "isothermal_1300K.Spectrum.binned.dat"
    )

    apollo_model_spectral_output = np.loadtxt(apollo_model_spectrum_filepath).T
    tamso_wavelo, tamso_wavehi, tamso_flux, *_ = apollo_model_spectral_output
    tamso_wave = (tamso_wavelo + tamso_wavehi) / 2

    apollo_onestream_model_spectrum_filepath: Path = (
        model_directory / "isothermal_1300K.one-stream.Spectrum.binned.dat"
    )

    apollo_onestream_spectral_output = np.loadtxt(
        apollo_onestream_model_spectrum_filepath
    ).T
    tamso_onestream_wavelo, tamso_onestream_wavehi, tamso_onestream_flux, *_ = (
        apollo_onestream_spectral_output
    )
    tamso_onestream_wave = (tamso_onestream_wavelo + tamso_onestream_wavehi) / 2

    picaso_model_spectrum_filepath: Path = model_directory / "isothermal_spectra.nc"

    ESTIMATED_COMPANION_RADIUS_VS_SOLAR: Final[float] = 0.116
    ESTIMATED_DISTANCE_TO_SYSTEM_IN_PARSECS: Final[float] = 64.5

    picaso_model_spectral_output = xr.open_dataset(picaso_model_spectrum_filepath)
    picaso_wave = picaso_model_spectral_output.wavelength
    picaso_flux = (
        picaso_model_spectral_output.flux_emission
        * convert_surface_quantity_to_observed_quantity(
            distance_in_parsecs=ESTIMATED_DISTANCE_TO_SYSTEM_IN_PARSECS,
            radius_in_solar_radii=ESTIMATED_COMPANION_RADIUS_VS_SOLAR,
        )
    )

    figure, axis = plt.subplots(1, 1, figsize=(15, 10))

    axis.plot(
        reference_model_wavelengths,
        resampled_onestream_flux,
        linewidth=2,
        color="cornflowerblue",
        label="(Current attempt) Potluck isothermal model, 1300 K, 1-stream",
    )

    axis.plot(
        reference_model_wavelengths,
        resampled_twostream_flux,
        linewidth=5,
        color="indigo",
        label="(Current attempt) Potluck isothermal model, 1300 K",
    )

    axis.plot(
        tamso_onestream_wave,
        tamso_onestream_flux,
        label="APOLLO isothermal model, 1300 K, 1-stream",
        color="palegreen",
        linewidth=2,
    )

    axis.plot(
        tamso_wave,
        tamso_flux,
        label="APOLLO isothermal model, 1300 K",
        color="mediumseagreen",
        linewidth=2,
    )

    axis.plot(
        picaso_wave,
        picaso_flux,
        label="PICASO isothermal model, 1300 K",
        color="crimson",
        linewidth=2,
    )

    axis.set_xlim((1.0, 5.0))
    # axis.set_ylim((0, axis.get_ylim()[1]))
    axis.set_xlabel("Wavelength (microns)")
    axis.set_ylabel(rf"Flux (erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$)")
    axis.legend(frameon=False, fontsize=16)

    for filetype in ["png", "pdf"]:
        plt.savefig(
            output_file_directory / f"{model_directory_label}_RT_comparison.{filetype}",
            bbox_inches="tight",
        )
