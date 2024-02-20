"""
Module for evaluating the loss between and input and target mts.TensorMap.
Currently implemented is the "L2Loss" function that finds the summed squared
error.

For tensorial properties of arbitrary rank, or real-space scalar fields expanded
on an orthogonal basis set the L2 loss is given by the standard sum of square
residuals between elements in the input and target tensors.

For real-space scalar fields expanded on a non-orthogonal basis set, the L2 loss
must be evaluated using an overlap-type matrix which accounts for spatial
correlations between basis functions.
"""
from typing import List, Optional, Union

import torch
import metatensor.torch as mts

from rhocalc import convention


class L2Loss(torch.nn.Module):
    """
    Evaluates the L2 loss between input and target TensorMaps. These can
    correspond to a batch of one or more structures. The loss is not averaged in
    any way.

    For tensorial properties of arbitrary rank, or real-space scalar fields
    expanded on an orthogonal basis set the L2 loss is given by the standard sum
    of square residuals between elements in the input and target tensors.

    .. math::

        L = \sum_i (∆c_i)^2

    where :math:`∆c_i` is the difference between the input and target TensorMaps
    for element :math:`i`.

    For real-space scalar fields expanded on a non-orthogonal basis set, the L2
    loss must be evaluated using an overlap-type matrix which accounts for
    spatial correlations between basis functions.

    .. math::

            L = \sum_ij ∆c_i . \hat{O}_ij . ∆c_j

    where :math:`\hat{O}_ij` is the overlap metric between two basis functions
    :math:`i` and :math:`j`. In principle any overlap-type matrix can be passed,
    such as the Coulomb matrix.

    As overlap matrices are in general symmetric, symmetrised overlap matrices
    in the mts.TensorMap format must be passed to this module, in accordnace with
    the symnmetrization procedures contained within the module
    :py:mod:`rhocalc.convert`. The data structure assumed can be found in the
    dummy mts.TensorMap in :py:mod:`rhocalc.convention`, but this is automatically
    produced when uses the parses provided with `rho_learn`.

    Note that no reduction method (i.e. averaging) is currently implemented for
    this evaluation, such that the returned loss is given by a sum over all
    separable loss components.
    """
    def __init__(self) -> None:
        super(L2Loss, self).__init__()

    def forward(
        self,
        input: Union[mts.TensorMap, List[mts.TensorMap]],
        target: Union[mts.TensorMap, List[mts.TensorMap]],
        overlap: Optional[Union[mts.TensorMap, List[mts.TensorMap]]] = None,
        structure_idxs: Optional[List[int]] = None,
        check_metadata: bool = True,
    ) -> torch.Tensor:
        """
        Calculate the L2Loss between input and target TensorMaps. If `overlap`
        is passed as None, a standard L2 loss is evaluated. Quantities can be
        tensors of arbitrary rank or real-space scalar fields expanded on an
        orthogonal basis set.

        If `overlap` is passed, the quantities are assumed to be scalar fields
        expanded on a non-orthogonal basis set, and the loss is evaluated using
        the spatial correlations between basis functions contained with the
        overlap matrix.

        In the latter case, the structure indices for which the loss should be
        evaluated can be passed.
        """
        # ===== Orthogonal basis set

        # Evaluate the L2 loss on the input and target as passed.
        # L = \sum_i (∆c_i)^2
        if overlap is None:
            return evaluate_l2_loss_orthogonal_basis(input=input, target=target)


        # ===== Non-orthogonal basis set

        # By passing overlap-type matrices, we assume the quantities passed are
        # a scalar field expanded on a non-orthogonal basis set.
        # L = \sum_ij ∆c_i . O_ij . ∆c_j
        return evaluate_l2_loss_nonorthogonal_basis(
            input=input,
            target=target,
            overlap=overlap,
            structure_idxs=structure_idxs,
        )        

# =================================
# ====== Non-orthogonal loss ======
# =================================

def evaluate_l2_loss_nonorthogonal_basis(
    input: List[mts.TensorMap],
    target: List[mts.TensorMap],
    overlap: List[mts.TensorMap],
    structure_idxs: List[int],
    check_metadata: bool = True,
) -> torch.Tensor:
    """
    Calculates the squared error loss between the input (ML) and target (QM)
    scalar fields. As these are expanded on the same non-orthogonal basis
    set, the loss is evaluated using the overlap-type matrix:

    .. math::

            L = \sum_ij ∆c_i . \hat{O}_ij . ∆c_j
    """
    # First slice the data so that we have one mts.TensorMap per structure, if
    # necessary
    if isinstance(input, mts.TensorMap):
        if structure_idxs is None:
            raise ValueError(
                "If `input` is a single mts.TensorMap, `structure_idxs` must be passed."
                " If passing a mts.TensorMap for a single structure, pass it in a list."
            )
        input = [
            mts.slice(
                input, 
                "samples", 
                mts.Labels(names=["structure"], values=np.array([A]).reshape(-1, 1))
            ) 
            for A in structure_idxs
        ]
    if isinstance(target, mts.TensorMap):
        if structure_idxs is None:
            raise ValueError(
                "If `input` is a single mts.TensorMap, `structure_idxs` must be passed."
                " If passing a mts.TensorMap for a single structure, pass it in a list."
            )
        target = [
            mts.slice(
                target, 
                "samples", 
                mts.Labels(names=["structure"], values=np.array([A]).reshape(-1, 1))
            ) 
            for A in structure_idxs
        ]
    if isinstance(overlap, mts.TensorMap):
        if structure_idxs is None:
            raise ValueError(
                "If `input` is a single mts.TensorMap, `structure_idxs` must be passed."
                " If passing a mts.TensorMap for a single structure, pass it in a list."
            )
        overlap = [
            mts.slice(
                overlap, 
                "samples", 
                mts.Labels(names=["structure"], values=np.array([A]).reshape(-1, 1))
            ) 
            for A in structure_idxs
        ]   

    loss = 0
    for inp, tar, ovl in zip(input, target, overlap):

        # Check metadata
        if check_metadata:
            _check_forward_args(inp, tar, ovl)

        # Evaluate loss
        loss += evaluate_l2_loss_nonorthogonal_basis_one_structure(
            input=inp, target=tar, overlap=ovl
        )

    return loss


def evaluate_l2_loss_nonorthogonal_basis_one_structure(
    input: mts.TensorMap,
    target: mts.TensorMap,
    overlap: mts.TensorMap,
) -> torch.Tensor:
    """
    Evaluates the L2 loss between the non-orthorthogonal basis set coefficients
    of the `input` and `target` scalar fields.

    Assumes that the passed TensorMaps correspond to only a single structure.
    """
    # Calculate the delta coefficient tensor
    delta_coeffs = mts.subtract(input, target)

    # Calculate the loss for each overlap matrix block in turn
    loss = 0
    for key, ovlp_block in overlap.items():

        # Unpack key values and retrieve the coeff blocks
        l1, l2, a1, a2 = key.values
        c1 = delta_coeffs.block(
            spherical_harmonics_l=l1,
            species_center=a1,
        ).values
        c2 = delta_coeffs.block(
            spherical_harmonics_l=l2,
            species_center=a2,
        ).values

        # Reshape the overlap block
        i1, m1, n1 = c1.shape
        i2, m2, n2 = c2.shape
        o_vals = ovlp_block.values.reshape(i1, i2, m1, m2, n1, n2)
        o_vals = o_vals.permute(0, 2, 4, 1, 3, 5)

        # Calculate the block loss by dot product
        block_loss = torch.tensordot(
            torch.tensordot(c1, o_vals, dims=3), c2, dims=3
        )

        # Count the off-diagonal blocks twice as we only work with the
        # upper-triangle of the overlap matrix
        if l1 == l2 and a1 == a2:
            loss += block_loss
        else:
            loss += 2 * block_loss

    return loss

# =============================
# ====== Orthogonal loss ======
# =============================


def evaluate_l2_loss_orthogonal_basis(
    input: Union[torch.ScriptObject, List[torch.ScriptObject]],
    target: Union[torch.ScriptObject, List[torch.ScriptObject]],
    check_metadata: bool = True,
) -> torch.Tensor:
    """
    Calculates the squared error loss between the input (ML) and target (QM)
    quantities. These could be any tensorial quantity of arbitrary rank, or a
    scalar field expanded on an orthogonal basis set.

    In this case, the loss is evaluated independently for each element in the
    input/target tensors, and summed to give the total loss.

    .. math::

        L = \sum_i (∆c_i)^2

    """
    # Check metadata
    if check_metadata:
        if isinstance(input, list) or isinstance(input, tuple):
            for inp, tar in zip(input, target):
                _check_forward_args(inp, tar)
        else:
            _check_forward_args(input, target)

    # Use the "sum" reduction method to calculate the loss for each block
    torch_mse = torch.nn.MSELoss(reduction="sum")
    loss = 0

    if isinstance(input, torch.ScriptObject):
        assert isinstance(target, torch.ScriptObject)
        for in_block, tar_block in zip(input, target):
            loss += torch_mse(input=in_block.values, target=tar_block.values)
    else:
        assert (isinstance(input, list) or isinstance(input, tuple))
        assert (isinstance(target, list) or isinstance(target, tuple))
        for inp, tar in zip(input, target):
            for in_block, tar_block in zip(inp, tar):
                loss += torch_mse(input=in_block.values, target=tar_block.values)

    return loss


def _check_forward_args(
    input: torch.ScriptObject,
    target: torch.ScriptObject,
    overlap: Optional[torch.ScriptObject] = None,
) -> None:
    """Checks metadata of arguments to forward are valid."""
    # Check metadata of input and target
    if not mts.equal_metadata(input, target):
        raise ValueError(
            "`input` and `target` TensorMaps must have equal metadata."
        )

    if overlap is not None:
        # Check metadata structure of the overlap matrix
        target_metadata = convention.OVERLAP_MATRIX
        if overlap.keys.names != target_metadata.keys.names:
            raise ValueError(
                "`overlap` mts.TensorMap must have key names"
                f"{target_metadata.keys.names}, got {overlap.keys.names}."
            )
        if overlap.sample_names != target_metadata.sample_names:
            raise ValueError(
                "`overlap` mts.TensorMap must have sample names"
                f"{target_metadata.sample_names}, got {overlap.sample_names}."
            )
        if overlap.component_names != target_metadata.component_names:
            raise ValueError(
                "`overlap` mts.TensorMap must have component names"
                f"{target_metadata.component_names}, got {overlap.component_names}."
            )
        if overlap.property_names != target_metadata.property_names:
            raise ValueError(
                "`overlap` mts.TensorMap must have property names"
                f"{target_metadata.property_names}, got {overlap.property_names}."
            )
