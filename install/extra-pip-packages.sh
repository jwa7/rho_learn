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
pip uninstall -y pytorch rascaline rascaline-torch
pip cache remove "torch*" 
pip cache remove "rascaline*"
pip uninstall -y metatensor metatensor-core metatensor-operations metatensor-torch metatensor-learn
pip cache remove "metatensor*"

# ===== PyTorch, rascaline, rascaline-torch with CPU-only torch
pip install --extra-index-url https://download.pytorch.org/whl/cpu "rascaline-torch @ git+https://github.com/luthaf/rascaline@ddd880286fa570ee38ab1698f8569c02db484eb8#subdirectory=python/rascaline-torch"

# ===== metatensor-operations and metatensor-learn. Just use the release version for now.
pip install --no-build-isolation --no-deps metatensor-operations metatensor-learn
