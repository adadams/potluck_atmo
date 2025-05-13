#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
  echo "Error: Conda is not installed. Please install conda before running this script."
  exit 1
fi

# Create a new conda environment with Python
if ! conda create -n potluck python; then
  echo "Error: Failed to create conda environment. Please check the conda installation and try again."
  exit 1
fi

# Activate the new environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate potluck

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
  echo "Error: requirements.txt not found. Please make sure it's in the same directory as this script."
  exit 1
fi

# Install dependencies from requirements.txt
if ! pip install -r requirements.txt; then
  echo "Error: Failed to install dependencies. Please check the requirements.txt file and try again."
  exit 1
fi

# Apply hotfix to package
PACKAGE_DIR=$(python -c "import os; print(os.path.dirname(os.path.abspath(__import__('spectres').__file__)))")
perl -pi -e "s/from \.spectral_resampling_numba import spectres/from \.spectral_resampling_numba import spectres_numba as spectres/" "$PACKAGE_DIR/__init__.py"

echo "Environment setup complete!"