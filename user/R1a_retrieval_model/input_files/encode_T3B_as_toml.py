from pathlib import Path

import msgspec
import pandas as pd

from xarray_serialization import (
    AltitudeValue,
    MixingRatioValue,
    PressureValue,
    TemperatureValue,
)


class RISOTTOVerticalInputs(msgspec.Struct):
    pressures_by_level: list[PressureValue]
    temperatures_by_level: list[TemperatureValue]
    mixing_ratios_by_level: dict[str, list[MixingRatioValue]]


current_directory: Path = Path(__file__).parent

# Read the CSV file
input_data_filepath: Path = current_directory / "T3B_malbec_TP.txt"
df = pd.read_csv(
    input_data_filepath,
    sep="\s+",
    header=None,
    names=["pressure", "temperature", "mmw", "altitude", "H2", "He", "H2O", "CH4"],
)

# Create a dictionary to store the data
data = {}

# Iterate over the columns and add them to the dictionary
for column in df.columns:
    data[column] = df[column].tolist()


# Write the data to a TOML file
with open(current_directory / "output.toml", "wb") as file:
    file.write(msgspec.toml.encode(data))


"""
temperatures_by_level,
    mmw_by_level,
    altitude_by_level,
    H2_by_level,
    He_by_level,
    H2O_by_level,
    CH4_by_level
    """
