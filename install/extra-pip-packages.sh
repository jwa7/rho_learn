#!/usr/bin/env bash

set -eux

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

# ===== Uninstalls, clear caches
pip uninstall -y torch rascaline rascaline-torch metatensor metatensor-core metatensor-operations metatensor-torch metatensor-learn wigners
pip cache remove "torch*" 
pip cache remove "rascaline*" 
pip cache remove "metatensor*"

# ===== PyTorch, rascaline, rascaline-torch with CPU-only torch
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.2
pip install metatensor-torch
pip install wigners
pip install --no-build-isolation --no-deps "rascaline @ git+https://github.com/luthaf/rascaline@1f879f9d54d1b4e24bbe4c089b3c749a7c8ba7d8"
pip install --no-build-isolation --no-deps "rascaline-torch @ git+https://github.com/luthaf/rascaline@1f879f9d54d1b4e24bbe4c089b3c749a7c8ba7d8#subdirectory=python/rascaline-torch"

# ===== metatensor-operations release version
pip install --no-build-isolation --no-deps metatensor-operations

# ===== metatensor-learn latest version
pip install --no-build-isolation --no-deps "metatensor-learn @ git+https://github.com/lab-cosmo/metatensor@2f60f48798a2c34b0fbd91d40a1d85420c456f26#subdirectory=python/metatensor-learn"
