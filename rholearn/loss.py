from typing import List

import numpy as np
import torch

import equistore
from equistore import Labels, TensorBlock, TensorMap
from rholearn import utils

VALID_LOSS_FNS = ["DensityLoss", "CoeffLoss"]


def transform_overlap_for_densityloss(overlap: TensorMap) -> TensorMap:
    """
    Reshapes and permutes the axes of the blocks of the overlap matrix such that
    it is ready for the tensor dot operation involved in the evaluation of the
    loss using the :py:class:`DensityLoss` class.

    This method assumes that the passed overlap-type matrices are of equistore
    TensorMap format and have the following data structure:

        - key names: ('spherical_harmonics_l1', 'spherical_harmonics_l2',
          'species_center_1', 'species_center_2')
        - sample names: ('structure', 'center_1', 'center_2')
        - component names: [('spherical_harmonics_m1',),
          ('spherical_harmonics_m2',)]
        - property names: ('n1', 'n2')

    and only correspond to one structure. This should be the case for TensorMaps
    processed with the functions in the :py:mod:`rhoparse.convert` module. The
    output TensorMap has a special data structure that conceptually goes against
    the equistore philosophy (see
    https://lab-cosmo.github.io/equistore/latest/get-started/concepts.html), but
    has been designed to allow for faster evaluation of the loss when using the
    :py:class:`DensityLoss` class in this module. The data strutcure of the
    output TensorMap is:

        - key names: ('spherical_harmonics_l1', 'spherical_harmonics_l2',
          'species_center_1', 'species_center_2') - i.e. same as the input.
        - sample names: ('structure', 'center_1',)
        - component names: [('spherical_harmonics_m1',), ('n1',), ('center_2',),
          ('spherical_harmonics_m2',),]
        - property names: ('n2',)

    i.e. the notable changes are that the center and radial channel indices now
    exist along orthogonal axes and the axes have been permuted.
    """
    # Iterate over blocks and manipulate each in turn
    keys, new_blocks = overlap.keys, []
    for key in keys:
        # Retrieve the block
        block = overlap[key]

        # Get the structure index
        A = np.unique(block.samples["structure"])[0]

        # Get the axes dimensions
        i1 = np.unique(block.samples["center_1"])
        i2 = np.unique(block.samples["center_2"])
        m1 = np.unique(block.components[0]["spherical_harmonics_m1"])
        m2 = np.unique(block.components[1]["spherical_harmonics_m2"])
        n1 = np.unique(block.properties["n1"])
        n2 = np.unique(block.properties["n2"])

        # Reshape the block and permute the axes such that the data for the
        # first of the two centers is on the 'left' and the second on the 'right'.
        s_block = block.values.reshape(
            len(i1), len(i2), len(m1), len(m2), len(n1), len(n2)
        )
        # Shape before: (n_i1, n_i2, n_m1, n_m2, n_n1, n_n2)
        # Shape after : (n_i1, n_m1, n_n1, n_i2, n_m2, n_n2)
        s_block = torch.permute(s_block, (0, 2, 4, 1, 3, 5)).contiguous()

        # Build a new TensorBlock with updated metadata
        new_block = TensorBlock(
            samples=Labels(
                names=["structure", "center_1"], values=np.array([[A, i] for i in i1])
            ),
            components=[
                Labels(names=["spherical_harmonics_m1"], values=m1.reshape(-1, 1)),
                Labels(names=["n1"], values=n1.reshape(-1, 1)),
                Labels(names=["center_2"], values=i2.reshape(-1, 1)),
                Labels(names=["spherical_harmonics_m2"], values=m2.reshape(-1, 1)),
            ],
            properties=Labels(names=["n2"], values=n2.reshape(-1, 1)),
            values=s_block,
        )
        new_blocks.append(new_block)

    return TensorMap(keys, new_blocks)


class CoeffLoss(torch.nn.Module):
    """
    Computes the mean squared loss on the electron density *coefficients* for a
    batch of one or more structures.

    For a given structure, A, the loss on the electron density is given by:

    .. math::

        L = (\Delta c)^2

    where :math:`\Delta c` is the difference between predicted (i.e. ML) and
    reference (i.e. density fitted RI) electron density coefficients.

    If evaluating the loss for multiple structures, the total loss is given by
    the sum of individual losses for each structure.
    """

    def __init__(self):
        super(CoeffLoss, self).__init__()

    @staticmethod
    def _check_forward_args(input: TensorMap, target: TensorMap):
        """
        Checks input and target types and asserts they have equal metadata
        """
        if not equistore.equal_metadata(input, target):
            raise ValueError(
                "``input`` and ``target`` must have equal metadata in the same order."
            )

    def forward(
        self, input: TensorMap, target: TensorMap, unsafe: bool = False
    ) -> float:
        """
        Calculates the mean squared loss between 2 TensorMaps.

        Assumes both `input` and `target` TensorMaps are have a torch-backend,
        and have equal metadata in the same order.

        :param input: a :py:class:`TensorMap` corresponding to the ML-predicted
            electron density coefficients.
        :param target: a :py:class:`TensorMap` corresponding to the RI-fitted
            electron density coefficients.
        :param unsafe: bool, if True, skips input checks for speed.

        :return loss: a :py:class:`torch.Tensor` containing a single float
            values, corresponding to the total loss metric.
        """
        # Input checks
        if not unsafe:
            self._check_forward_args(input, target)

        # Use the "sum" reduction method to calculate the loss for each block
        torch_mse = torch.nn.MSELoss(reduction="sum")
        loss = 0
        for key in input.keys:
            loss += torch_mse(input=input[key].values, target=target[key].values)

        return loss


class DensityLoss(torch.nn.Module):
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
    batches of one structure or more.
    """

    def __init__(self):
        super(DensityLoss, self).__init__()

    @staticmethod
    def _check_forward_args(input: TensorMap, target: TensorMap, overlap: TensorMap):
        """
        Checks inputs to the forward method, i.e. checks metadata is valid.
        """
        # Check metadata of input and target
        if not equistore.equal_metadata(input, target):
            raise ValueError(
                "``input`` and ``target`` must have equal metadata in the same order."
            )
        # Check metadata of overlap
        if overlap.sample_names != ("center_1",):
            raise ValueError("``overlap`` must have sample names ('center_1',).")
        if len(overlap.components) != 4:
            raise ValueError(
                "``overlap`` must have 4 components, corresponding to the axes "
                "('spherical_harmonics_m1',), ('n1',), ('center_2',), "
                "('spherical_harmonics_m2',)"
            )
        for i, names in enumerate(
            [
                ("spherical_harmonics_m1",),
                ("n1",),
                ("center_2",),
                ("spherical_harmonics_m2",),
            ]
        ):
            if overlap.components[i].names != names:
                raise ValueError(
                    "``overlap`` component {} must have names {}.".format(i, names)
                )
        if overlap.property_names != ("n2",):
            raise ValueError("``overlap`` must have property names ('n2',).")

    def forward(
        self,
        input: TensorMap,
        target: TensorMap,
        overlap: TensorMap,
        unsafe: bool = False,
    ) -> float:
        """
        Calculates the squared error loss between the input (ML) and target (QM)
        electron densities.
        """
        if not unsafe:
            self._check_forward_args(input, target, overlap)

        # Calculate the delta coefficients, i.e. the difference between the
        # input and target coefficients. `equistore.subtract` checks the
        # metadata of the input and target TensorMaps for us.
        delta_c = equistore.subtract(input, target)

        loss = 0
        for key in overlap.keys:
            l1, l2, a1, a2 = key

            # Retrieve the pairs of delta blocks we're evaluating the loss for
            delta_c_block_1 = delta_c.block(
                spherical_harmonics_l=l1, species_center=a1
            ).values
            delta_c_block_2 = delta_c.block(
                spherical_harmonics_l=l2, species_center=a2
            ).values

            # Retrieve the corresponding overlap block that measures the
            # correlations between pairs of basis functions in the corresponding
            # delta blocks
            s_block = overlap[key]

            # Calculate the loss for this pair of blocks by dot product. As the
            # overlap matrix has been symmetrized we only evaluate off-diagonal
            # blocks once. As such, multiply the result for off-diagonals by 2
            # to account for this.
            loss_block = torch.tensordot(
                torch.tensordot(delta_c_block_1, s_block, dims=3),
                delta_c_block_2,
                dims=3,
            )
            if l1 == l2 and a1 == a2:  # diagonal block
                loss += loss_block
            else:  # off-diagonal block
                loss += 2 * loss_block

        return loss
