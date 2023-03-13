# _rho\_learn_

Author: Joseph W. Abbott, PhD Student @ Lab COSMO, EPFL


## About

A proof-of-concept workflow for torch-based electron density learning.
Equivariant structural representations, that is those that transform
equivariantly under rotation with the irreducible spherical components of the
target property (i.e. whether they be invariant or covariant), can be
constructed using the xyz coordinates of a set of training molecules. As part of
a supervised machine learning scheme, these structural representations form
inputs to the machine learning model, while reference electron densities
calculated by quantum chemical methods are given as target outputs.

Model training is performed using PyTorch, but various other parts of the
workflow are performed with a number of open source packages from the software
stacks of labs COSMO and LCMD at EPFL. The main ones are outlined below.


**``Q-Stack``**: [lcmd-epfl/Q-stack](https://github.com/lcmd-epfl/Q-stack)

A software stack dedicated to pre- and post-processing tasks for quantum machine
learning. In **rholearn**, its **equistore**-interfacing module ``equio`` is
used to pre-process electron density coefficients generated from quantum
chemical calculations into TensorMap format, ready for input into a
**equistore**/**torch**-based ML model. Its ``fields`` module is also used to
post-process electron densities into a format suitable for visualization.


**``rascaline``**: [Luthaf/rascaline](https://github.com/Luthaf/rascaline)

This is used to generate equivariant structural representations for input into
ML models. In **rholearn**, **rascaline** is used to build an atom-centered
density descriptor of the molecules in the training data, which is then used to
generate an equivariant $\lambda$-SOAP representation.


**``equistore``**: [lab-cosmo/equistore](https://github.com/lab-cosmo/equistore)

This is a storage format for atomistic machine learning, allowing an efficient
way to track data and associated metadata for a wide range of atomistic systems
and objects. In **rholearn**, it is used to store $\lambda$-SOAP structural
representations, ground and excited state electron densities, and Coulomb
overlap matrices.


**``equisolve``**: [lab-cosmo/equisolve](https://github.com/lab-cosmo/equisolve)

Concerned with higher-level functions and classes built on top of **equistore**,
this package is used to prepare data and build models for machine learning.
However, it is still in a very early alpha development stage. In **rholearn**,
it is used to split TensorMap objects storing structural representations and
electron densities into train, test, and validation data.


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

Is used to compile code in ``rascaline`` and ``equistore``. To install
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

  2. **rascaline**: ``pip install git+https://github.com/luthaf/rascaline.git@rholearn``

  3. **equisolve**: ``pip install git+https://github.com/lab-cosmo/equisolve.git@rholearn``

  4. **equistore**: ``pip install git+https://github.com/lab-cosmo/equistore.git@rholearn``

  5. **qstack**: ``pip install git+https://github.com/jwa7/Q-stack.git``

  6. **chemiscope**: ``pip install chemiscope``

  7. **rholearn**: ensure you're in the ``rho_learn/`` directory then ``pip install .``


## Jupyter Notebooks

In order to run the example notebooks (see Examples section below), you'll need
to work within the ``rho`` conda environment. If the env doesn't show up in your
jupyter IDE, you can run the terminal command ``ipython kernel install --user
--name=rho`` and restart the jupyter session.

Then, working within the ``rho`` environment, **``rholearn``** and other modules
can then be imported in a Python script as follows:

```py
from rholearn.features import lambda_soap_vector
from rholearn.loss import MSELoss, CoulombLoss
from rholearn.models import EquiModelGlobal
from rholearn.training import train

from equistore import io, Labels, TensorMap
from equisolve.utils import split_data
```

## Updates and Troubleshooting

If you ever need to update ``rholearn``, or any of the other packages for that
matter, make sure you uninstall before ``git pull``-ing and reinstalling:

```
pip uninstall rholearn
cd rho_learn
git pull
pip install .
```

If you ever receive the equistore error ``ValueError: Trying to set the
EQUISTORE library path twice error`` it is probably due to a clash between
rascaline and equistore. This will soon be fixed in equistore, but unitl then
make sure that any rascaline modules are loaded before equistore ones. The main
rholearn module you need to be careful of in this regard is ``features.py``. If,
for example, you are importing from this module in a notebook, i.e. ``from
rholearn.features import lambda_soap_vector``, make sure you place this import
at the very top of the notebook. This is because ``features.py`` imports both
rascaline and equistore (in that order), but thus needs to be placed first.

# Examples

The functionality of this package is demonstrated in the example notebooks in
the ``rho_learn/docs/example/azoswitch`` directory. This section will briefly the
various parts of the workflow.

## Data

In order to keep the proof-of-concept workflow lightweight, the example dataset
is a 10-molecule subset of a 128-molecule database of molecular azoswitches.
This itself is a subset of a larger database used in the electron density
learning paper entitled __"Learning the Exciton Properties of Azo-dyes"__
DOI: [10.1021/acs.jpclett.1c01425](https://doi.org/10.1021/acs.jpclett.1c01425).


The 128-molecule database, including xyz files, ground- and 1st- and 2nd-
excited particle and hole electron densities, and coulomb matrices can be found
on [switchdrive](https://drive.switch.ch/index.php/s/qbOa1JKnsv0ebVv), but the
even smaller 10-molecule xyz files, ground state electron density, and coulomb
matrices can be found in the ``rholearn``
[``example/azoswitch/data/``](https://github.com/jwa7/rho_learn/tree/main/docs/example/azoswitch/data)
directory.

The first notebook
[1_data.ipynb](https://github.com/jwa7/rho_learn/blob/main/docs/example/azoswitch/1_data.ipynb)
outlines how to visualize the molecular data, construct a $\lambda$-SOAP
representation, perform a train-test-validation split and partition data into
training subsets of various sizes such that a learning exercise can be
performed.

## Model Training

The second notebook
[2_models.ipynb](https://github.com/jwa7/rho_learn/blob/main/docs/example/azoswitch/2_models.ipynb)
details how to build both linear and nonlinear global torch models, check for
equivariance, and perform model trianing.

## Analysis

The third notebook
[3_analysis.ipynb](https://github.com/jwa7/rho_learn/blob/main/docs/example/azoswitch/3_analysis.ipynb)
shows how some standard analysis figures, such as loss vs epoch, learning curves,
and parity plots can be plotted, as well as how to visualize the electron density.



# References

* Symmetry-Adapted Machine Learning for Tensorial Properties of Atomistic
  Systems, Phys. Rev. Lett. 120, 036002. DOI:
  [10.1103/PhysRevLett.120.036002](https://doi.org/10.1103/PhysRevLett.120.036002)

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