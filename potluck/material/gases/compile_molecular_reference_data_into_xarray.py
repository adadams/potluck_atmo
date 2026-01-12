import sys
from pathlib import Path

import xarray as xr

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from molecular_metrics import (
    ASPLUND_SOLAR_ELEMENTAL_ABUNDANCES_RELATIVE_TO_H,
    ATOMS_PER_MOLECULE_BY_ELEMENT,
    MOLECULAR_WEIGHTS,
)

current_directory: Path = Path(__file__).parent

molecular_species_coordinate: xr.Variable = xr.Variable(
    dims=("molecular_species",),
    data=list(MOLECULAR_WEIGHTS.keys()),
    attrs={"units": "dimensionless"},
)
print(f"{molecular_species_coordinate=}")

elements_coordinate: xr.Variable = xr.Variable(
    dims=("element",),
    data=list(ASPLUND_SOLAR_ELEMENTAL_ABUNDANCES_RELATIVE_TO_H.keys()),
    attrs={"units": "dimensionless"},
)
print(f"{elements_coordinate=}")

solar_elemental_abundance_dataarray: xr.DataArray = xr.DataArray(
    data=list(ASPLUND_SOLAR_ELEMENTAL_ABUNDANCES_RELATIVE_TO_H.values()),
    dims=("element",),
    coords={"element": elements_coordinate},
    attrs={"units": "dimensionless"},
    name="solar_elemental_abundance_relative_to_H",
)

print(f"{solar_elemental_abundance_dataarray=}")

molecular_weight_dataarray: xr.DataArray = xr.DataArray(
    data=list(MOLECULAR_WEIGHTS.values()),
    dims=("molecular_species",),
    coords={"molecular_species": molecular_species_coordinate},
    attrs={"units": "g/mol"},
    name="molecular_weight",
)

print(f"{molecular_weight_dataarray=}")

atoms_per_molecule_dataarray: xr.DataArray = xr.DataArray(
    data=[
        [
            atoms_per_molecule[molecular_species]
            if molecular_species in atoms_per_molecule
            else 0
            for element_name, atoms_per_molecule in ATOMS_PER_MOLECULE_BY_ELEMENT.items()
        ]
        for molecular_species in MOLECULAR_WEIGHTS
    ],
    dims=("molecular_species", "element"),
    coords={
        "molecular_species": molecular_species_coordinate,
        "element": elements_coordinate,
    },
    attrs={"units": "dimensionless"},
    name="atoms_per_molecule",
)

print(f"{atoms_per_molecule_dataarray.sel(molecular_species="h2s", element="h")=}")


molecular_reference_data: xr.Dataset = xr.Dataset(
    data_vars={
        "solar_elemental_abundance_relative_to_H": solar_elemental_abundance_dataarray,
        "molecular_weight": molecular_weight_dataarray,
        "atoms_per_molecule": atoms_per_molecule_dataarray,
    },
    coords={
        "molecular_species": molecular_species_coordinate,
        "element": elements_coordinate,
    },
)

molecular_reference_data.to_netcdf(
    current_directory / "reference_data" / "molecular_reference_data.nc"
)
