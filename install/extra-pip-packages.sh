#!/usr/bin/env bash

set -eux

pip install --extra-index-url https://download.pytorch.org/whl/cpu torch

pip uninstall -y metatensor metatensor-core metatensor-operations metatensor-torch
pip uninstall -y rascaline rascaline-torch

pip cache remove "metatensor*"
pip cache remove "rascaline*"

pip install --no-build-isolation git+https://github.com/lab-cosmo/metatensor@659ce8dc6cab85cbea95371a4f841cd3bd686b54
pip install --no-build-isolation git+https://github.com/luthaf/rascaline@6dc73889a6adcf2e34fb24967cbd039f45055bbe