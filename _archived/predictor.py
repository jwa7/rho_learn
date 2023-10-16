"""
Module to make an electron density prediction on an xyz file using a pretrained
model.
"""
from typing import Optional

import ase.io
import numpy as np
import pyscf
import torch

import metatensor
from metatensor import TensorMap

import qstack
from qstack import equio

from rholearn import io, features, utils


def predict_density_from_xyz(
    xyz_path: str,
    rascal_hypers: dict,
    model_path: str,
    basis: str,
    inv_means_path: Optional[str] = None,
) -> np.ndarray:
    """
    Loads a xyz file of structure(s) at `xyz_path` and uses the `rascal_hypers`
    to generate a lambda-SOAP structural representation. Loads the the
    pretrained torch model from `model_path` and uses it to make a prediction on
    the electron density. Returns the prediction both as a TensorMap and as a
    vector of coefficients.

    :param xyz_path: Path to xyz file containing structure to predict density
        for.
    :param rascal_hypers: dict of rascaline hyperparameters to use when
        computing the lambda-SOAP representation of the input structure.
    :param model_path: path to the trained rholearn/torch model to use for
        prediction.
    :param basis: the basis set, i.e. "ccpvqz jkfit", to use when constructing
        the vectorised density coefficients. Must be the same as the basis used
        to calculate the training data.
    :param inv_means_path: if the invariant blocks have been standardized by
        subtraction of the mean of their features, the mean needs to be added
        back to the prediction. If so, `inv_means_path` should be the path to
        the TensorMap containing these means. Otherwise, pass as None (default).
    """

    # Load xyz file to ASE
    frame = ase.io.read(xyz_path)

    # Create a molecule object with Q-Stack
    mol = qstack.compound.xyz_to_mol(xyz_path, basis=basis)

    # Generate lambda-SOAP representation
    input = features.lambda_soap_vector([frame], rascal_hypers, even_parity_only=True)

    # Load model from file
    model = io.load_torch_object(
        model_path, device=torch.device("cpu"), torch_obj_str="model"
    )

    # Drop blocks from input that aren't present in the model
    input = metatensor.drop_blocks(input, keys=np.setdiff1d(input.keys, model.keys))

    # Convert the input TensorMap to torch
    input = metatensor.to(
        input,
        backend="torch",
        requires_grad=False,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    return predict_density_from_mol(input, mol, model_path, inv_means_path)


def predict_density_from_mol(
    input: TensorMap,
    mol: pyscf.gto.Mole,
    model_path: str,
    inv_means_path: Optional[str] = None,
) -> np.ndarray:
    """
    Loads the the pretrained torch model from `model_path` and uses it to make a
    prediction on the electron density. Returns the prediction both as a
    TensorMap and as a vector of coefficients.

    :param input: a TensorMap containing the lambda-SOAP representation of the
        structure to predict the density for.
    :param mol: a PySCF :py:class:`Mole` object, initialized with the correct
        basis, for the specific xyz structure the structural representation in
        `input` is constructed for.
    :param model_path: path to the trained rholearn/torch model to use for
        prediction.
    :param inv_means_path: if the invariant blocks have been standardized by
        subtraction of the mean of their features, the mean needs to be added
        back to the prediction. If so, `inv_means_path` should be the path to
        the TensorMap containing these means. Otherwise, pass as None (default).
    """
    # Load model from file
    model = io.load_torch_object(
        model_path, device=torch.device("cpu"), torch_obj_str="model"
    )

    # Make a prediction using the model
    with torch.no_grad():
        out_pred = model(input)

    # Add back the feature means to the invariant (l=0) blocks if the model was trained
    # against electron densities with standardized invariants
    if inv_means_path is not None:
        inv_means = metatensor.load(inv_means_path)
        out_pred = features.standardize_invariants(
            tensor=metatensor.to(out_pred, backend="numpy"),
            invariant_means=inv_means,
            reverse=True,
        )

    # Drop the structure label from the TensorMap
    tmp_out_pred = utils.drop_metadata_name(out_pred, axis="samples", name="structure")

    # Convert TensorMap to Q-Stack coeffs. Need to rename the TensorMap keys
    # here to fit LCMD naming convention
    vect_coeffs = qstack.equio.tensormap_to_vector(
        mol,
        utils.rename_tensor(
            tmp_out_pred, keys_names=["spherical_harmonics_l", "element"]
        ),
    )

    return out_pred, vect_coeffs
