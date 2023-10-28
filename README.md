# _rho\_learn_

Author: Joseph W. Abbott, PhD Student @ Lab COSMO, EPFL


## About

A proof-of-concept workflow for torch-based equivariant learning of scalar
fields and tensorial properties.

Equivariant structural representations, that is those that transform
equivariantly under rotation with the irreducible spherical components of thetarget property (i.e. whether they be invariant or covariant), can be
constructed using the xyz coordinates of a set of training molecules. As part of
a supervised machine learning scheme, these structural representations form
inputs to the machine learning model, while reference electron densities
calculated by quantum chemical methods are given as target outputs.

Model training is performed using PyTorch, but various other parts of the
workflow are performed with a number of open source packages from the software
stacks of labs COSMO and LCMD at EPFL. The main ones are outlined below.


<!-- **``Q-Stack``**: [lcmd-epfl/Q-stack](https://github.com/lcmd-epfl/Q-stack)

A software stack dedicated to pre- and post-processing tasks for quantum machine
learning. In **rholearn**, its **metatensor**-interfacing module ``equio`` is
used to pre-process electron density coefficients generated from quantum
chemical calculations into TensorMap format, ready for input into a
**metatensor**/**torch**-based ML model. Its ``fields`` module is also used to
post-process electron densities into a format suitable for visualization. -->


**``rascaline``**: [Luthaf/rascaline](https://github.com/Luthaf/rascaline)

This is used to generate equivariant structural representations for input into
ML models. In **rholearn**, **rascaline** is used to build an atom-centered
density descriptor of the molecules in the training data, which is then used to
generate an equivariant $\lambda$-SOAP representation.


**``metatensor``**: [lab-cosmo/metatensor](https://github.com/lab-cosmo/metatensor)

This is a storage format for atomistic machine learning, allowing an efficient
way to track data and associated metadata for a wide range of atomistic systems
and objects. In **rholearn**, it is used to store $\lambda$-SOAP structural
representations, ground and excited state electron densities, and Coulomb
overlap matrices.


**``equisolve``**: [lab-cosmo/equisolve](https://github.com/lab-cosmo/equisolve)

Concerned with higher-level functions and classes built on top of **metatensor**,
this package is used to prepare data and build models for machine learning.
It can be used for sample and feature selection prior to model training.


**``chemiscope``**: [lab-cosmo/chemiscope](https://github.com/lab-cosmo/chemiscope)

This package is used a an interactive visualizer and property explorer for the
molecular data from which the structural representations are built.


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

1. Clone the **``rholearn``** repo and create a virtual environment.

```
git clone https://github.com/jwa7/rho_learn.git
cd rho_learn
conda env create -f environment.yml
conda activate rho
```

Then, some atomistic ML packages from the lab COSMO and LCMD software stacks can
be installed in the ``rho`` environment. Ensure you install these **in the order
shown below** (this is very important) and with the exact commands, as some
development branches are required for this setup.
  
  2. **metatensor**: ``pip install git+https://github.com/lab-cosmo/metatensor@a816c6e2cff723d869afa90ee8064fff54c2a531``
  
  2. **rascaline**: ``pip install git+https://github.com/luthaf/rascaline.git@clebsch_gordan``

  <!-- 3. **equisolve**: ``pip install git+https://github.com/m-stack-org/equisolve.git@rholearn`` -->

  4. **qstack**: ``pip install git+https://github.com/jwa7/Q-stack.git``

  5. **chemiscope**: ``pip install chemiscope``

  6. **rholearn**: ensure you're in the ``rho_learn/`` directory then ``pip install .``


## Jupyter Notebooks

In order to run the example notebooks (see Examples section below), you'll need
to work within the ``rho`` conda environment. If the env doesn't show up in your
jupyter IDE, you can run the terminal command ``ipython kernel install --user
--name=rho`` and restart the jupyter session.

Then, working within the ``rho`` environment, **``rholearn``** and other modules
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

An end-to-end workflow of learning the HOMO scalar field of gas-phase water
monomers is provided in the `docs/example/scalar_field` directory. Note: generate of the
learning targets requires a specific version of the quantum chemistry code
`FHI-aims`, which is not yet publicly available.

An example of tensorial learning is provided in 

## Water

A demonstrative notebook is provided, implementing the key parts of the
model-training workflow. This begins with the generation of a $\lambda-SOAP
structural representation for a 1000-water monomer database. The relationship to
the electron density is then learned using a linear model, optimizing weights
based on gradient descent of an $L^2$ loss function. The aim of this notebook is
to be a concise introduction to the key components of torch-based learning of
the electron density.

<!-- The open-source packge ["Symmetry-Adapted Learning of Three-dimensional
Electron Densities" (SALTED)](https://github.com/andreagrisafi/SALTED) is the
source for this dataset and was used to generate reference electron densities,
with outputs converted to `metatensor` format. -->

<!-- ## Azoswitch

Also included in the examples are a set of more pedagogical notebooks on a more
complicated dataset. In order to be lightweight enough to run on a laptop, and
with the emphasis being on the workflow as opposed to accurate results, a
10-molecule database of azoswitch dyes are provided.

This dataset is a 10-molecule subset of a database of molecular azoswitches used
in the electron density learning paper entitled __"Learning the Exciton
Properties of Azo-dyes"__ DOI:
[10.1021/acs.jpclett.1c01425](https://doi.org/10.1021/acs.jpclett.1c01425).

There are 3 example notebooks for the azoswitch workflow.

The first notebook
[1_data.ipynb](https://github.com/jwa7/rho_learn/blob/main/docs/example/azoswitch/1_data.ipynb)
outlines how to visualize the molecular data, construct a $\lambda$-SOAP
representation, perform a train-test-validation split and partition data into
training subsets of various sizes such that a learning exercise can be
performed.

The second
[2_models.ipynb](https://github.com/jwa7/rho_learn/blob/main/docs/example/azoswitch/2_models.ipynb)
goes through the construction of both a linear and nonlinear model, custom loss
functions (based on the Coulomb metric), and how to use these in model training.

The third and final notebook
[3_analysis.ipynb](https://github.com/jwa7/rho_learn/blob/main/docs/example/azoswitch/3_analysis.ipynb)
outlines plotting figures analysing model training, as well as making a
prediction on a validation structure, visualizing this prediction and the
associated error. -->

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