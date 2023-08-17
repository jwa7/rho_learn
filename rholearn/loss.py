from typing import List, Union, Sequence

import numpy as np
import torch

import equistore
from equistore import Labels, TensorBlock, TensorMap

from rholearn import utils

VALID_LOSS_FNS = ["RhoLoss", "CoeffLoss"]


class RhoLoss(torch.nn.Module):
    """
    Computes the mean squared loss on the electron density for a batch of one or
    more structures.

    As the basis functions the electron density is expanded on are
    non-orthogonal, evaluation of this loss requires use of overlap-type
    matrices that capture the coupling between the basis functions.

    For a given structure, A, the loss on the electron density is given by:

    .. math::

        L = \Delta c \hat{O} \Delta c

    where :math:`\Delta c` is the difference between predicted (i.e. ML) and
    reference (i.e. density fitted RI) electron density coefficients, and :math:
    `\hat{O}` is a matrix corresponding to an overlap-type metric, i.e.
    corresponding to either the overlap matrices, :math:`\hat{S}`, or the
    Coulomb matrices, :math:`\hat{J}`, between pairs of basis functions.

    If evaluating the loss for multiple structures, the total loss is given by
    the sum of individual losses for each structure.

    This class assumes that the passed overlap-type matrices are of equistore
    TensorMap format and have the following data structure:

        - key names: ('spherical_harmonics_l1', 'spherical_harmonics_l2',
          'species_center_1', 'species_center_2') - i.e. same as the input.
        - sample names: ('center_1',)
        - component names: [('spherical_harmonics_m1',), ('n1',), ('center_2',),
          ('spherical_harmonics_m2',),]
        - property names: ('n2',)

    This data structure is designed specifically for fast tensordot evaluation
    of the loss. The method :py:func:`transform_overlap_for_densityloss` should
    be used to pre-process the overlap-type matrices for use in this class.

    It is also assumed that, due to the symmetric nature of overlap-type
    matrices, they have been symmetrized such that for blocks with key indices
    (l1, l2, a1, a2), only blocks with keys where l1 <= l2, and where a1 <= a2
    in the case where l1 == l2 are stored.

    This symmetrization can be performed with the functions in the the
    :py:mod:`rhoparse.convert` module. Due to such symmetrization, contributions
    to the loss of diagonal blocks (i.e. where l1 == l2 and a1 == a2) is counted
    once, but those from off-diagonal blocks (i.e. where l1 != l2 or a1 != a2)
    are counted twice (2x multiplier).

    Due to the size of the overlap matrices, loss evaluation can be very
    memory-intensive. This class is built to allow for loss to be evaluated in
    batches of one structure or more. In the case of evaluating the loss for a
    batch of more than one structure, `input`, `target` and `overlap` should be
    passed as a list of TensorMap, where each element corresponds to a
    respective structure in the batch.

    :param input: a :py:class:`TensorMap` or list of :py:class:`TensorMap`
        corresponding to the batch of ML-predicted electron density
        coefficients.
    :param target: a :py:class:`TensorMap` or list of :py:class:`TensorMap`
        corresponding to the batch of reference electron density coefficients.
    :param overlap: a :py:class:`TensorMap` or list of :py:class:`TensorMap`
        corresponding to the batch of overlap-type matrices.
    """

    def __init__(self) -> None:
        super(RhoLoss, self).__init__()

    @staticmethod
    def _check_forward_args(
        input: TensorMap, target: TensorMap, overlap: TensorMap
    ) -> None:
        """
        Checks metadata of arguments to forward are valid.
        """
        # Check metadata of input and target
        if not equistore.equal_metadata(input, target):
            raise ValueError(
                "`input` and `target` TensorMaps must have equal metadata."
            )

        # Check metadata of overlap. Sample names may have the "tensor" index
        # present as a by-product of using the join function. This can be
        # removed once the join function is updated in equistore (TODO).
        if overlap.sample_names != ["structure", "center_1", "center_2"]:
            raise ValueError(
                "each `overlap` TensorMap must have sample names ('structure', "
                f"'center_1', 'center_2), got {overlap.sample_names}."
            )
        c_names = [
            ["spherical_harmonics_m1"],
            ["spherical_harmonics_m2"],
        ]
        if not np.all(overlap.components_names == c_names):
            raise ValueError(
                "each `overlap` TensorMap must have 4 components, corresponding to the axes "
                f"{c_names}, got {overlap.components_names}."
            )
        if overlap.property_names != ["n1", "n2"]:
            raise ValueError(
                "each `overlap` TensorMap must have property names ('n1', 'n2',), "
                f"got {overlap.property_names}."
            )

    def forward(
        self,
        input: TensorMap,
        target: TensorMap,
        overlap: TensorMap,
        check_args: bool = True,
        structure_idxs=None,
    ) -> torch.Tensor:
        """
        Calculates the squared error loss between the input (ML) and target (QM)
        electron densities.
        """
        if check_args:
            self._check_forward_args(input, target, overlap)

        # Get the struture indices present if not passed
        if structure_idxs is None:
            structure_idxs = equistore.unique_metadata(input, "samples", "structure")

        # Calculate the delta coefficient tensor
        delta_coeffs = equistore.subtract(input, target)

        # Calculate the loss for each overlap matrix block in turn
        total_loss = 0
        for key, ovlp_block in overlap.items():
            # Unpack key values and retrieve the coeff blocks
            l1, l2, a1, a2 = key.values
            c1 = delta_coeffs.block(
                spherical_harmonics_l=l1,
                species_center=a1,
            )
            c2 = delta_coeffs.block(
                spherical_harmonics_l=l2,
                species_center=a2,
            )
            block_loss = 0
            for A in structure_idxs:
                # Slice the block values to the current structure
                c1_vals = c1.values[c1.samples.column("structure") == A]
                c2_vals = c2.values[c2.samples.column("structure") == A]
                o_vals = ovlp_block.values[ovlp_block.samples.column("structure") == A]
                # Reshape the overlap block
                i1, m1, n1 = c1_vals.shape
                i2, m2, n2 = c2_vals.shape
                o_vals = o_vals.reshape(i1, i2, m1, m2, n1, n2)
                o_vals = o_vals.permute(0, 2, 4, 1, 3, 5)
                # Calculate the block loss by dot product
                block_loss += torch.tensordot(
                    torch.tensordot(c1_vals, o_vals, dims=3), c2_vals, dims=3
                )
            # Count the off-diagonal blocks twice as we only work with the
            # upper-triangle of the overlap matrix
            if l1 == l2 and a1 == a2:
                total_loss += block_loss
            else:
                total_loss += 2 * block_loss

        return total_loss


class CoeffLoss(torch.nn.Module):
    """
    Computes the squared loss on the electron density *coefficients* for a batch
    of one or more structures.

    For a given structure, A, the loss on the coefficients is given by:

    .. math::

        L = (\Delta c)^2

    where :math:`\Delta c` is the difference between input/predicted (i.e. ML)
    and target/reference (i.e. density fitted RI) electron density coefficients.

    If evaluating the loss for multiple structures, the total loss is given by
    the sum of individual losses for each structure.

    :param input: a :py:class:`TensorMap` or list of :py:class:`TensorMap`
        corresponding to the batch of ML-predicted electron density
        coefficients.
    :param target: a :py:class:`TensorMap` or list of :py:class:`TensorMap`
        corresponding to the batch of reference electron density coefficients.
    """

    def __init__(self) -> None:
        super(CoeffLoss, self).__init__()

    @staticmethod
    def _check_forward_args(input: TensorMap, target: TensorMap) -> None:
        """
        Checks input and target types and asserts they have equal metadata
        """
        if not equistore.equal_metadata(input, target):
            raise ValueError(
                "``input`` and ``target`` must have equal metadata in the same order."
            )

    def forward(
        self,
        input: Union[TensorMap, List[TensorMap]],
        target: Union[TensorMap, List[TensorMap]],
        check_args: bool = True,
    ) -> torch.Tensor:
        """
        Calculates the squared loss between 2 TensorMaps.

        Assumes both `input` and `target` TensorMaps are have a torch-backend,
        and have equal metadata in the same order.

        :param input: a :py:class:`TensorMap` corresponding to the ML-predicted
            electron density coefficients.
        :param target: a :py:class:`TensorMap` corresponding to the RI-fitted
            electron density coefficients.
        :param check_args: bool, if False, skips input checks for speed. Default
            true.

        :return loss: a :py:class:`torch.Tensor` containing a single float
            values, corresponding to the total loss metric.
        """
        # Collate TensorMaps if passed as a list or tuple
        if isinstance(input, Sequence):
            input = equistore.join(input, "samples")
        if isinstance(target, Sequence):
            target = equistore.join(target, "samples")

        # Input checks
        if check_args:
            self._check_forward_args(input, target)

        # Use the "sum" reduction method to calculate the loss for each block
        torch_mse = torch.nn.MSELoss(reduction="sum")
        loss = 0
        for key in input.keys:
            loss += torch_mse(input=input[key].values, target=target[key].values)

        return loss
        