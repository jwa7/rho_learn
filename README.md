# _rho\_learn_

Author: Joseph W. Abbott, PhD Student @ Lab COSMO, EPFL



## About

A proof-of-concept framework for torch-based equivariant learning of scalar
fields and tensorial properties expanded in the angular basis. This package
provides the building blocks for end-to-end `torch`-based learning and
prediction pipelines interfaced with `metatensor`, a storage format for
atomistic data. 

The subpackage `rholearn` contains modules for loss functions, datasets,
dataloaders, models, and training, allowing gradient-based workflows with
minibatching to be built. 

The subpackage `rhocalc` contains infrastructure to interface the core
functionality in `rholearn` with Quantum Chemistry codes. This involves routines
to generate learning targets and parse outputs into `metatensor` format for
input in the ML workflow. Currently, only a simple interface (via `ase`
calculators) with the electronic structure code `FHI-aims` is implemented, with
a focus on the generation of scalar fields expanded onto a fitted RI basis.


Some of the software modules `rho_learn` conbines into a workflow are described
below:

* **``rascaline``**: [Luthaf/rascaline](https://github.com/Luthaf/rascaline).
  This is used to transform xyz coordinates of systems into a suitable
  (equivariant) structural representation for input into a model. `rascaline`
  features a series of calculators, with a utility suite for performing clebsch
  gordan iterations coming soon.


* **``metatensor``**:
  [lab-cosmo/metatensor](https://github.com/lab-cosmo/metatensor). This is a
  storage format for atomistic machine learning, allowing an efficient way to
  track data and associated metadata for a wide range of atomistic systems and
  objects. In `rho_learn`, custom `torch` modules are built to allow for ML
  workflows based on `TensorMap` objects.


* **``equisolve``**:
  [lab-cosmo/equisolve](https://github.com/lab-cosmo/equisolve). Concerned with
  higher-level functions and classes built on top of **metatensor**, this
  package is used to prepare data and build models for machine learning. It can be
  used for sample and feature selection prior to model training.


* **``chemiscope``**:
  [lab-cosmo/chemiscope](https://github.com/lab-cosmo/chemiscope). This package
  is used a an interactive visualizer and property explorer for the molecular
  data from which the structural representations are built.


# Set up

## Requirements

The only requirements to begin installation are ``git``, ``conda`` and ``rustc``:

* ``git >= 2.37``
* ``conda >= 22.9``
* ``rustc >= 1.65``

**``conda``**
 
Is used as a package and environment manager. It allows a virtual environment
will be created within which the appropriate version of Python (``== 3.10``) and
required packages will be installed.

If you don't already have ``conda``, the latest version of the lightweight
[``miniforge``](https://github.com/conda-forge/miniforge/releases/) can be
downloaded from [here](https://github.com/conda-forge/miniforge/releases/), for
your specific operating system. After downloading, change the execute
permissions and run the installer, for instance as follows:

```
chmod +x Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh
```

and follow the installation instructions.

When starting up a terminal, if the ``(base)`` label on the terminal user prompt
is not seen, the command ``bash`` might have to be run to activate ``conda``.

**``rustc``**

Is used to compile code in ``rascaline`` and ``metatensor``. To install
``rustc``, run the following command, taken from the ['Install
Rust'](https://www.rust-lang.org/tools/install) webpage:

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

and follow the installation instructions.


## Installation


Clone this repo and create a ``conda`` environment using the ``environment.yml``
file. This will install all the required base packages, such as ase,
numpy, torch and matplotlib, into an environment called ``rho``.

1. Clone the **``rho_learn``** repo and create a virtual environment.

```
git clone https://github.com/jwa7/rho_learn.git
cd rho_learn
conda env create -f environment.yml
conda activate rho
```

Then, some atomistic ML packages can be installed in the ``rho`` environment.
Ensure you install these **in the order shown below** (this is very important)
and with the exact commands, as some development branches are required for this
setup.

  2. **chemiscope**: ``pip install chemiscope``
  
  2. **metatensor**: ``pip install metatensor``
  
  2. **rascaline**: ``pip install git+https://github.com/luthaf/rascaline.git@b2cedfe870541e6d037357db58de1901eb116c41``

  2. **rho_learn**: ensure you're in the ``rho_learn/`` directory then ``pip install .``


## Jupyter Notebooks

In order to run the example notebooks (see Examples section below), you'll need
to work within the ``rho`` conda environment. If the env doesn't show up in your
jupyter IDE, you can run the terminal command ``ipython kernel install --user
--name=rho`` and restart the jupyter session.

Then, working within the ``rho`` environment, **``rho_learn``** and other modules
can then be imported in a Python script as follows:

```py
import rascaline
from rascaline.utils import clebsch_gordan

import metatensor
from metatensor import Labels, TensorMap

from rholearn.loss import L2Loss
from rholearn.models import RhoModel
```


# Examples

**Scalar Fields**. An end-to-end workflow of learning the HOMO scalar field of gas-phase water
monomers is provided in the `docs/example/field` directory. More specifically,
the notebook `ml_homo.ipynb` covers a complete end-to-end workflow, including
learning target generation. Note: generation of these learning targets requires
a specific version of the quantum chemistry code `FHI-aims`, which is not yet
publicly available. The notebook `ml_homo_no_aims.ipynb` features the same
workflow but without dependency on `FHI-aims`, using instead pre-calculated
target for the HOMOs of 10 water monomers.

**Tensors**. As for tensorial learning, an example notebook `ml_training.ipynb` is
provided in `docs/example/tensor`, learning pseudo-data in the form of random
tensors decomposed in the angular basis.


# References

* Symmetry-Adapted Machine Learning for Tensorial Properties of Atomistic
  Systems, Phys. Rev. Lett. 120, 036002. DOI:
  [10.1103/PhysRevLett.120.036002](https://doi.org/10.1103/PhysRevLett.120.036002)

* SALTED (Symmetry-Adapted Learning of Three-dimensional Electron Densities),
  GitHub:
  [github.com/andreagrisafi/SALTED](https://github.com/andreagrisafi/SALTED/),
  Andrea Grisafi, Alan M. Lewis.

* Transferable Machine-Learning Model of the Electron Density, ACS Cent. Sci.
  2019, 5, 57−64. DOI:
  [10.1021/acscentsci.8b00551](https://doi.org/10.1021/acscentsci.8b00551)

* Atom-density representations for machine learning, J. Chem. Phys. 150, 154110
  (2019). DOI: [10.1063/1.5090481](https://doi.org/10.1063/1.5090481)

* Learning the Exciton Properties of Azo-dyes, J. Phys. Chem. Lett. 2021, 12,
  25, 5957–5962. DOI:
  [10.1021/acs.jpclett.1c01425](https://doi.org/10.1021/acs.jpclett.1c01425)
  
* Impact of quantum-chemical metrics on the machine learning prediction of
  electron density, J. Chem. Phys. 155, 024107 (2021), DOI:
  [10.1063/5.0055393](https://doi.org/10.1063/5.0055393)