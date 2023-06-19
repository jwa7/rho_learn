"""
Tests functionality of the rholearn.loss module.
"""
import ase.io
import numpy as np
import pytest
import torch

import equistore

from rholearn import io, loss
from rhoparse import convert


def test_densityloss_identity_overlap():
    """
    In the case where the overlap matrix is the identity matrix (i.e. orthogonal
    basis functions, no coupling between them), the loss evaluated by the
    DenstiyLoss class should be exactly equivalent to the squared error on the
    coefficients evaluated by the CoeffLoss class.

    This function tests that this holds.
    """
    # Load the xyz file into an ASE frame
    frame = ase.io.read("data/frame.xyz")

    # Load dummy coefficients vector and overlap matrix
    c = np.load("data/coeff_vector.npy")
    s = np.load("data/overlap_matrix.npy")

    # Create a new overlap matrix that is the identity matrix
    s = np.eye(*s.shape)

    # Load dummy calculation info. The only important bits here are the basis
    # set info: lmax and nmax
    calc = io.unpickle_dict("data/calc_info.pickle")

    # Convert c to TensorMap
    c_tm = equistore.to(
        convert.coeff_vector_ndarray_to_tensormap(
            frame, c, calc["lmax"], calc["nmax"], structure_idx=0
        ),
        backend="torch",
        dtype=torch.float64,
        device="cpu",
        requires_grad=True,
    )

    # Create a dummy coefficient prediction
    c_tm_pred = equistore.random_uniform_like(c_tm)

    # Convert s to TensorMap and sparsify
    s_tm = convert.overlap_matrix_ndarray_to_tensormap(
        frame, s, calc["lmax"], calc["nmax"], structure_idx=0
    )
    assert convert.overlap_is_symmetric(s_tm)
    s_tm = convert.overlap_drop_redundant_off_diagonal_blocks(s_tm)

    # Transform overlap ready for DensityLoss
    s_tm = equistore.to(
        s_tm, backend="torch", dtype=torch.float64, device="cpu", requires_grad=True
    )
    s_tm = loss.transform_overlap_for_densityloss(s_tm)

    # Check for equivalence
    assert loss.CoeffLoss()(c_tm_pred, c_tm) == loss.DensityLoss()(
        c_tm_pred, c_tm, s_tm
    )
