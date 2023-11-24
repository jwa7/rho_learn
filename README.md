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

## Installation

Pre-requisite: a working `conda` installation. With this, follow the
installation instructions below.

```
git clone https://github.com/jwa7/rho_learn
cd rho_learn
conda env create --file install/environment.yaml
conda activate rho
./install/extra-pip-packages.sh
pip install .
```

In the case of error `bash: ./install/extra-pip-packages.sh: Permission denied` you
might have to change the permission using 
`chmod +x ./install/extra-pip-packages.sh` before running `./install/extra-pip-packages.sh`.

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