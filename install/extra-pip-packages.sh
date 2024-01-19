#!/usr/bin/env bash

set -eux

pip install --extra-index-url https://download.pytorch.org/whl/cpu torch

pip uninstall -y metatensor metatensor-core metatensor-operations metatensor-torch
pip uninstall -y rascaline rascaline-torch

pip cache remove "metatensor*"
pip cache remove "rascaline*"

# Has to be in this order, such that the recent changes in metatensor overrides
# the older metatensor version installed by rascaline
pip install --no-build-isolation git+https://github.com/luthaf/rascaline@1c0694096bc6832dc059f4f382627c246b4895f9
pip install --no-build-isolation git+https://github.com/lab-cosmo/metatensor@00cb4c5b4b1f5f00c6424259448be1e65631d3a2
