import numpy as np
from numpy.typing import NDArray

from xarray_functional_wrappers import Dimensionalize, rename_and_unitize
from xarray_serialization import PressureType, WavelengthType

"""
// computes the flux ratio received at each wavelength IN TRANSIT
vector<double> Planet::transFlux(double rs, vector<double> wavens, string table)
{
  if(table=="hires"){
    vector<double> fluxes(wavens.size(),0);

    for(int i=0; i<wavens.size(); i++){
      double dflux = pi*rp*rp;

      for(int j=nlayer-1; j>=0; j--){
        // Entire layer below cloud deck
        if(hprof[j]<hmin){
          dflux += (hprof[j]-hprof[j+1]) * 2*pi*(rp+hprof[j]);
        }

        // Entire layer above cloud deck
        else if(hprof[j+1]>=hmin){
          dflux += (1 - exp(-tauprof[i][j])) * (hprof[j]-hprof[j+1]) * 2*pi*(rp+hprof[j]);
        }

        // Layer overlaps cloud deck boundary
        else{
          double fvisible = (hprof[j]-hmin)/(hprof[j]-hprof[j+1]);
          dflux += (1 - exp(-tauprof[i][j])) * (hprof[j]-hprof[j+1]) * 2*pi*(rp+hprof[j]) * fvisible;
        }
      }

      fluxes[i] = dflux / (pi*rs*rs);
    }
    return fluxes;
  }
"""


@rename_and_unitize(new_name="transit_depth", units="dimensionless")
@Dimensionalize(
    argument_dimensions=(
        (WavelengthType, PressureType),
        (PressureType,),
        (PressureType,),
        None,
        None,
    ),
    result_dimensions=((WavelengthType,),),
)
def calculate_transmission_spectrum(
    cumulative_optical_depth: NDArray[np.float64],
    path_lengths: NDArray[np.float64],
    altitudes: NDArray[np.float64],
    stellar_radius_in_cm: float,
    planet_radius_in_cm: float,
) -> NDArray[np.float64]:
    solid_planet_disk_area: float = np.pi * planet_radius_in_cm**2
    solid_stellar_disk_area: float = np.pi * stellar_radius_in_cm**2

    planet_area_with_atmosphere: float = solid_planet_disk_area + np.sum(
        (1 - np.exp(-cumulative_optical_depth))
        * path_lengths
        * 2
        * np.pi
        * (planet_radius_in_cm + altitudes),
        axis=-1,
    )

    return planet_area_with_atmosphere / solid_stellar_disk_area
