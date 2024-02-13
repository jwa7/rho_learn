#!/usr/bin/env bash

set -eux

# PyTorch
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch

# A fork of ASE is needed, as this contains a bug fix for the FHI-aims ASE
# calculator
pip uninstall -y ase
pip cache remove "ase*"
pip install git+https://gitlab.com/jwa7/ase@977968238d011a795a417f9791fa816a80d20d87

# 'Cube-Toolz' for reading and manipulating cube files
pip uninstall -y cube_tools
pip cache remove "cube_tools*"
pip install git+https://github.com/funkymunkycool/Cube-Toolz.git

# Software from the COSMO stack
pip install chemiscope

pip uninstall -y metatensor metatensor-core metatensor-operations metatensor-torch metatensor-learn
pip uninstall -y rascaline rascaline-torch
pip cache remove "metatensor*"
pip cache remove "rascaline*"

# Has to be in this order, such that the recent changes in metatensor overrides
# the older metatensor version installed by rascaline
pip install --no-build-isolation git+https://github.com/luthaf/rascaline@02fff56f327f1fd8b31f0437638e7af4b34127bd
pip install --no-build-isolation git+https://github.com/lab-cosmo/metatensor@bec4561db93ed9a8037db6d39d264ebccb62c25d
