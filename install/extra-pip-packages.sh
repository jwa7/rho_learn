#!/usr/bin/env bash

set -eux

# ===== PyTorch
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2

# ===== ASE 
# ===== A fork is needed, as this contains a bug fix for the FHI-aims ASE calculator
pip uninstall -y ase
pip cache remove "ase*"
pip install git+https://gitlab.com/jwa7/ase@977968238d011a795a417f9791fa816a80d20d87

# ===== 'Cube-Toolz' for reading and manipulating cube files
pip uninstall -y cube_tools
pip cache remove "cube_tools*"
pip install git+https://github.com/funkymunkycool/Cube-Toolz.git

# ===== chemiscope
pip cache remove "chemiscope*"
pip install chemiscope

# ===== rascaline
# pip uninstall -y rascaline rascaline-torch
# pip cache remove "rascaline*"
# pip install --no-build-isolation git+https://github.com/luthaf/rascaline@96cee2a34f8b091fbd7be539ace3ed529216f0ef
# pip install --no-build-isolation git+https://github.com/luthaf/rascaline@96cee2a34f8b091fbd7be539ace3ed529216f0ef#subdirectory=python/rascaline-torch


# ===== metatensor
pip uninstall -y metatensor metatensor-core metatensor-operations metatensor-torch metatensor-learn
pip cache remove "metatensor*"
pip install --no-build-isolation git+https://github.com/lab-cosmo/metatensor@015d6d6eeef03ed4ab27f9d8a3a77e2f54356da9
pip install --no-build-isolation git+https://github.com/lab-cosmo/metatensor@015d6d6eeef03ed4ab27f9d8a3a77e2f54356da9#subdirectory=python/metatensor-torch
