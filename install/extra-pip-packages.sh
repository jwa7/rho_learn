#!/usr/bin/env bash

set -eux

pip install --extra-index-url https://download.pytorch.org/whl/cpu torch

pip uninstall -y metatensor metatensor-core metatensor-operations metatensor-torch
pip uninstall -y rascaline rascaline-torch

pip cache remove "metatensor*"
pip cache remove "rascaline*"

# Has to be in this order, such that the recent changes in metatensor overrides
# the older metatensor version installed by rascaline
pip install --no-build-isolation git+https://github.com/luthaf/rascaline@148ed002074de6c1cd2130fbbb1bbe0ddd43ed4d
pip install --no-build-isolation git+https://github.com/lab-cosmo/metatensor@659ce8dc6cab85cbea95371a4f841cd3bd686b54
