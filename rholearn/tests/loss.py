"""
Tests functionality of the rholearn.loss module.
"""
import ase.io
import numpy as np
import pytest
import torch

import metatensor

from rholearn import utils, loss
from rhocalc import convert


def test_l2loss_identity_overlap():
    """
    In the case where the overlap matrix is the identity matrix (i.e. orthogonal
    basis functions, no coupling between them), the loss evaluated by the
    DenstiyLoss class should be exactly equivalent to the squared error on the
    coefficients evaluated by the L2Loss class.

    This function tests that this holds.
    """
    # Load the xyz file into an ASE frame
    frame = ase.io.read("data/frame.xyz")

    # Load dummy coefficients vector and overlap matrix
    output = np.load("data/coeff_vector.npy")
    ovlp = np.load("data/overlap_matrix.npy")

    # Create a new overlap matrix that is the identity matrix
    ovlp = np.eye(*ovlp.shape)

    # Load dummy calculation info. The only important bits here are the basis
    # set info: lmax and nmax
    calc = utils.unpickle_dict("data/calc_info.pickle")

    # Convert output to TensorMap
    out_tm = metatensor.to(
        convert.coeff_vector_ndarray_to_tensormap(
            frame, output, calc["lmax"], calc["nmax"], structure_idx=0
        ),
        backend="torch",
        dtype=torch.float64,
        device="cpu",
        requires_grad=True,
    )

    # Create a dummy coefficient prediction
    out_tm_pred = metatensor.random_uniform_like(out_tm)

    # Convert ovlp to TensorMap and sparsify
    ovlp_tm = convert.overlap_matrix_ndarray_to_tensormap(
        frame, ovlp, calc["lmax"], calc["nmax"], structure_idx=0
    )
    assert convert.overlap_is_symmetric(ovlp_tm)
    ovlp_tm = convert.overlap_drop_redundant_off_diagonal_blocks(ovlp_tm)

    # Transform overlap ready for RhoLoss
    ovlp_tm = metatensor.to(
        ovlp_tm, backend="torch", dtype=torch.float64, device="cpu", requires_grad=True
    )

    # Check for equivalence
    assert loss.L2Loss()(input=out_tm_pred, target=out_tm) == loss.L2Loss()(
        input=out_tm_pred, target=out_tm, overlap=ovlp_tm
    )
