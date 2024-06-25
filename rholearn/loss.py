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

    def __init__(self, overlap_type: str) -> None:
        super(L2Loss, self).__init__()
        assert overlap_type in [None, "diagonal", "on-site", "off-site", "full"]
        self._overlap_type = overlap_type

    def forward(
        self,
        input: Union[mts.TensorMap, List[mts.TensorMap]],
        target: Union[mts.TensorMap, List[mts.TensorMap]],
        overlap: Optional[Union[mts.TensorMap, List[mts.TensorMap]]] = None,
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

        loss = 0

        # ==== Orthonormal basis (overlap = identity)

        if overlap is None:  # orthogonal basis
            if self._overlap_type is not None:
                raise ValueError(
                    "If evaluating loss just on coefficients (orthogonal basis), L2Loss"
                    " should be initialized with `overlap_type=None`."
                )

            if isinstance(input, list) or isinstance(input, tuple):
                for inp, targ in zip(input, target):
                    if check_metadata:
                        _check_metadata(inp, targ)
                    loss += _orthogonal_basis(inp, targ)
            else:
                if check_metadata:
                    _check_metadata(input, target)
                loss += _orthogonal_basis(input, target)

            return loss

        # ==== Orthogonal basis (overlap = diagonal)

        if _passed_overlap_is_diagonal(overlap):  # nonorthogonal, diagonal overlap
            if self._overlap_type != "diagonal":
                raise ValueError(
                    "If passing a diagonal overlap matrix, L2Loss should be"
                    " initialized with `overlap_type=None`."
                )

            if isinstance(input, list) or isinstance(input, tuple):
                for inp, targ, ovlp in zip(input, target, overlap):
                    if check_metadata:
                        _check_metadata(inp, targ, overlap_diagonal=ovlp)
                    loss += _nonorthogonal_basis_diagonal_overlap(
                        inp, targ, overlap_diagonal=ovlp
                    )
            else:
                if check_metadata:
                    _check_metadata(input, target, overlap_diagonal=overlap)
                loss += _nonorthogonal_basis_diagonal_overlap(
                    input, target, overlap_diagonal=overlap
                )

            return loss

        # ==== Orthogonal basis (overlap = full matrix).
        # ==== Can evaluate using either the "on-site", "off-site", or "full" overlap

        if isinstance(input, list) or isinstance(input, tuple):
            for inp, targ, ovlp in zip(input, target, overlap):
                if check_metadata:
                    _check_metadata(inp, targ, overlap_full=ovlp)
                loss += _nonorthogonal_basis_full_overlap(
                    inp, targ, overlap_full=ovlp, overlap_type=self._overlap_type
                )
        else:
            if check_metadata:
                _check_metadata(input, target, overlap_full=ovlp)
            loss += _nonorthogonal_basis_full_overlap(
                input, target, overlap_full=overlap, overlap_type=self._overlap_type
            )

        return loss


# ===== Loss computations for different overlap matrices =====


def _orthogonal_basis(
    input: torch.ScriptObject, target: torch.ScriptObject
) -> torch.Tensor:
    """
    Evaluates the L2 loss between `input` and `target` coefficients as if they form an
    orthonormal basis set, i.e. as if the overlap matrix was the identity.

    .. math::

        L = \sum_ij ∆c_i \hat{O}_{ij} ∆c_j
          = \sum_ij ∆c_i \hat{I}_{ij} ∆c_j
          = \sum_i (∆c_i)^2

    The passed TensorMaps can correspond to multiple structures, but must have equal
    metadata.
    """
    square_error_fn = torch.nn.MSELoss(reduction="sum")
    loss = 0
    for key in input.keys:
        loss += square_error_fn(
            input=input.block(key).values, target=target.block(key).values
        )
    return loss


def _nonorthogonal_basis_diagonal_overlap(
    input: torch.ScriptObject,
    target: torch.ScriptObject,
    overlap_diagonal: torch.ScriptObject,
) -> torch.Tensor:
    """
    Evaluates the L2 loss between `input` and `target` coefficients as if they form an
    orthogonal, but not normalized basis set, i.e. as if overlap is a diagonal matrix
    with diagonal elements not necessarily equal to one.

    .. math::

        L = \sum_ij ∆c_i \hat{O}_{ij} ∆c_j
          = \sum_i ∆c_i \hat{O}_{ii} ∆c_i
          = \sum_i (∆c_i)^2 \hat{O}_{ii}

    It is assumed that the `overlap_diagonal` matrix is passed as a TensorMap, but with
    the same metadata structure as `input` and `target`. The passed TensorMaps can
    correspond to multiple structures, but must have equal metadata.
    """
    loss = 0
    for key in input.keys:
        loss += torch.sum(
            ((input.block(key).values - target.block(key).values) ** 2)
            * overlap_diagonal.block(key).values
        )
    return loss


def _nonorthogonal_basis_full_overlap(
    input: torch.ScriptObject,
    target: torch.ScriptObject,
    overlap_full: torch.ScriptObject,
    overlap_type: str,
) -> torch.Tensor:
    """
    Evaluates the L2 loss between `input` and `target` coefficients as if they form an
    orthogonal, but not normalized basis set, i.e. as if overlap is a diagonal matrix
    with diagonal elements not necessarily equal to one.

    .. math::

        L = \sum_ij ∆c_i \hat{O}_{ij} ∆c_j


    If `off_diagonal_only` is true, the overlap matrix elements for on-site self
    interaction terms are set to zero before evaluating the loss. This ensures only a
    loss between off-diagonal elements (i.e. off-site cross interactions) is calculated.

    It is assumed that the `overlap_full` matrix is passed as a TensorMap, but with a **
    different ** metadata structure to `input` and `target`. This is because it must
    track matrix elements between all pairs of basis functions. In practice,
    `overlap_full` is passed with redundant off-diagonal blocks dropped (due to overlap
    symmetry), i.e. by the function
    `rhocalc.convert.overlap_drop_redundant_off_diagonal_blocks()`.

    Note: the passed TensorMaps must correspond to a single structure as the overlap
    matrix as the overlap blocks need to be reshaped. `input` and `target` must have the
    same metadata.
    """
    # Calculate the delta coefficient tensor
    delta_coeffs = mts.subtract(input, target)

    l_1: int
    l_2: int
    a1: int
    a2: int

    # Calculate the loss for each overlap matrix block in turn
    loss = 0
    for key, ovlp_block in overlap_full.items():

        # Unpack key values and retrieve the coeff blocks
        l_1, l_2, a1, a2 = key.values

        # if overlap_type == "diagonal":
        #     if not (l_1 == l_2 and a1 == a2):  # self-terms have the all same indices
        #         continue

        if overlap_type == "on-site":
            if a1 != a2:  # on-site basis functions must be of the same atomic type
                continue

        c1_block = delta_coeffs.block(
            {"o3_lambda": int(l_1), "center_type": int(a1)},
        )
        c2_block = delta_coeffs.block(
            {"o3_lambda": int(l_2), "center_type": int(a2)},
        )
        c1 = c1_block.values
        c2 = c2_block.values
        o_vals = ovlp_block.values

        if overlap_type is not None:

            # Copy overlap block values as will be modifying in-place
            o_vals = ovlp_block.copy().values

            # Explicitly set overlap matrix elements to zero depending on the overlap
            # type. Do this by finding the indices for each axis that need to be set to
            # zero.
            # if overlap_type == "diagonal":  # set off-diagonal elements to zero
            #     # Find the indices
            #     sample_idxs = torch.arange(len(ovlp_block.samples))[
            #         ovlp_block.samples.values[:, 1] != ovlp_block.samples.values[:, 2]
            #     ]
            #     # component_idxs = range(len(ovlp_block.components[0]))
            #     property_idxs = torch.arange(len(ovlp_block.properties))[
            #         ovlp_block.properties.values[:, 0]
            #         != ovlp_block.properties.values[:, 1]
            #     ]
            #     # Set to zero
            #     for i_sample in sample_idxs:
            #         for i_component_1, m1 in enumerate(ovlp_block.components[0].values):
            #             for i_component_2, m2 in enumerate(ovlp_block.components[1].values):
            #                 if m1 != m2:
            #                     for i_property in property_idxs:
            #                         o_vals[
            #                             i_sample,
            #                             i_component_1,
            #                             i_component_2,
            #                             i_property,
            #                         ] = 0.0

            # elif overlap_type == "off-diagonal":  # set diagonal elements to zero
            #     # Find the indices
            #     sample_idxs = torch.arange(len(ovlp_block.samples))[
            #         ovlp_block.samples.values[:, 1] == ovlp_block.samples.values[:, 2]
            #     ]
            #     # component_idxs = range(len(ovlp_block.components[0]))
            #     property_idxs = torch.arange(len(ovlp_block.properties))[
            #         ovlp_block.properties.values[:, 0]
            #         == ovlp_block.properties.values[:, 1]
            #     ]
            #     # Set to zero
            #     for i_sample in sample_idxs:
            #         for i_component_1, m1 in enumerate(ovlp_block.components[0].values):
            #             for i_component_2, m2 in enumerate(ovlp_block.components[1].values):
            #                 if m1 == m2:
            #                     for i_property in property_idxs:
            #                         o_vals[i_sample, i_component_1, i_component_2, i_property] = 0.0

            if overlap_type == "on-site":  # set off-site elements to zero
                # Find the indices
                sample_idxs = torch.arange(len(ovlp_block.samples))[
                    ovlp_block.samples.values[:, 1] != ovlp_block.samples.values[:, 2]
                ]
                # Set to zero
                for i_sample in sample_idxs:
                    o_vals[i_sample] = o_vals[i_sample] * 0.0

            elif overlap_type == "off-site":  # set on-site elements to zero
                # Find the indices
                sample_idxs = torch.arange(len(ovlp_block.samples))[
                    ovlp_block.samples.values[:, 1] == ovlp_block.samples.values[:, 2]
                ]
                # Set to zero
                for i_sample in sample_idxs:
                    o_vals[i_sample] = o_vals[i_sample] * 0.0

            else:
                if overlap_type != "full":
                    raise ValueError(f"invalid overlap type: {overlap_type}")

        # Reshape the overlap block
        i1, m1, n_1 = c1.shape
        i2, m2, n_2 = c2.shape
        o_vals = o_vals.reshape(i1, i2, m1, m2, n_1, n_2)
        o_vals = o_vals.permute(0, 2, 4, 1, 3, 5)

        # Calculate the block loss by dot product
        block_loss = torch.tensordot(torch.tensordot(c1, o_vals, dims=3), c2, dims=3)

        # Count the off-diagonal blocks twice as we only work with off-diagonal blocks
        # from one triangle of the overlap matrix
        if l_1 == l_2 and a1 == a2:
            loss += block_loss
        else:
            loss += 2 * block_loss

    return loss


# ===== Metadata and input checks =====


def _passed_overlap_is_diagonal(
    overlap: Union[torch.ScriptObject, List[torch.ScriptObject]]
) -> bool:
    """
    Determines whether the overlap matrix passed is the diagonal overlap matrix.
    """
    if isinstance(overlap, torch.ScriptObject):
        key_names = overlap.keys.names
    else:
        key_names = overlap[0].keys.names

    if key_names == ["o3_lambda", "center_type"]:
        return True
    elif key_names == ["o3_lambda_1", "o3_lambda_2", "center_1_type", "center_2_type"]:
        return False

    raise ValueError(f"unexpected key names for the `overlap`: {key_names}")


def _check_metadata(
    input: torch.ScriptObject,
    target: torch.ScriptObject,
    overlap_diagonal: Optional[torch.ScriptObject] = None,
    overlap_full: Optional[torch.ScriptObject] = None,
) -> None:
    """Checks metadata of arguments to forward are valid."""
    # Check metadata of input and target
    if not mts.equal_metadata(input, target):
        raise ValueError(
            "`input` and `target` TensorMaps must have equal metadata."
            f" input={input}, input_block0={input[0]}"
            f" target={target}, target_block0={target[0]}"
        )

    if overlap_diagonal is not None:
        if not mts.equal_metadata(input, overlap_diagonal):
            raise ValueError(
                "`input`, `target`, and `overlap_diagonal` TensorMaps must have equal metadata."
            )

    if overlap_full is not None:
        # Check metadata structure of the overlap matrix
        target_metadata = convention.OVERLAP_MATRIX
        if overlap_full.keys.names != target_metadata.keys.names:
            raise ValueError(
                "`overlap` mts.TensorMap must have key names"
                f"{target_metadata.keys.names}, got {overlap_full.keys.names}."
            )
        if overlap_full.sample_names != target_metadata.sample_names:
            raise ValueError(
                "`overlap` mts.TensorMap must have sample names"
                f"{target_metadata.sample_names}, got {overlap_full.sample_names}."
            )
        if overlap_full.component_names != target_metadata.component_names:
            raise ValueError(
                "`overlap` mts.TensorMap must have component names"
                f"{target_metadata.component_names}, got {overlap_full.component_names}."
            )
        if overlap_full.property_names != target_metadata.property_names:
            raise ValueError(
                "`overlap` mts.TensorMap must have property names"
                f"{target_metadata.property_names}, got {overlap_full.property_names}."
            )

        assert (
            len(mts.unique_metadata(input, axis="samples", names="system")) == 1
        )
        assert (
            len(mts.unique_metadata(target, axis="samples", names="system")) == 1
        )
        assert (
            len(mts.unique_metadata(overlap_full, axis="samples", names="system")) == 1
        )


# # =============================
# # ====== Orthogonal loss ======
# # =============================


# def loss_orthogonal_basis(
#     input: Union[torch.ScriptObject, List[torch.ScriptObject]],
#     target: Union[torch.ScriptObject, List[torch.ScriptObject]],
#     check_metadata: bool = True,
# ) -> torch.Tensor:
#     """
#     Calculates the squared error loss between the input (ML) and target (QM) quantities.
#     These could be any tensorial quantity of arbitrary rank, or a scalar field expanded
#     on an orthogonal basis set.

#     In this case, the loss is evaluated independently for each element in the
#     input/target tensors, and summed to give the total loss.

#     .. math::

#         L = \sum_i (∆c_i)^2

#     """
#     # Check metadata
#     if check_metadata:
#         if isinstance(input, list) or isinstance(input, tuple):
#             for inp, tar in zip(input, target):
#                 _check_forward_args(inp, tar)
#         else:
#             _check_forward_args(input, target)

#     # Use the "sum" reduction method to calculate the loss for each block
#     torch_mse = torch.nn.MSELoss(reduction="sum")
#     loss = 0

#     if isinstance(input, torch.ScriptObject):
#         assert isinstance(target, torch.ScriptObject)
#         for key in input.keys:
#             in_block = input.block(key)
#             tar_block = target.block(key)
#             loss += torch_mse(input=in_block.values, target=tar_block.values)
#     else:
#         assert isinstance(input, list) or isinstance(input, tuple)
#         assert isinstance(target, list) or isinstance(target, tuple)
#         for inp, tar in zip(input, target):
#             for key in inp.keys:
#                 in_block = inp.block(key)
#                 tar_block = tar.block(key)
#                 loss += torch_mse(input=in_block.values, target=tar_block.values)

#     return loss


# # =================================
# # ====== Non-orthogonal loss ======
# # =================================


# def loss_nonorthogonal_basis_diagonal_overlap(
#     input: Union[torch.ScriptObject, List[torch.ScriptObject]],
#     target: Union[torch.ScriptObject, List[torch.ScriptObject]],
#     overlap: Union[torch.ScriptObject, List[torch.ScriptObject]],
#     check_metadata: bool = True,
# ) -> torch.Tensor:
#     """
#     Calculates the squared error loss between the input (ML) and target (QM) quantities,
#     with each squared error scaled by its self-overlap. These could be any tensorial
#     quantity of arbitrary rank, or a scalar field expanded on an orthogonal basis set.

#     In this case, the loss is evaluated independently for each element in the
#     input/target tensors, and summed to give the total loss.

#     .. math::

#         L = \sum_ij ∆c_i . \hat{O}_{ij} . ∆c_j     for all i == j

#     """
#     # Check metadata
#     if check_metadata:
#         if isinstance(input, list) or isinstance(input, tuple):
#             for inp, tar, ovlp in zip(input, target, overlap):
#                 _check_forward_args(inp, tar)
#                 _check_forward_args(inp, ovlp)  # ovlp has the same metadata structure
#         else:
#             _check_forward_args(input, target)
#             _check_forward_args(input, overlap)

#     # Use the "sum" reduction method to calculate the loss for each block
#     torch_mse = torch.nn.MSELoss(reduction="sum")
#     loss = 0

#     if isinstance(input, torch.ScriptObject):
#         assert isinstance(target, torch.ScriptObject)
#         assert isinstance(overlap, torch.ScriptObject)
#         for key in input.keys:
#             loss += torch.sum(
#                 ((input.block(key).values - target.block(key).values) ** 2)
#                 * ovlp.block(keys).values
#             )
#     else:
#         assert isinstance(input, list) or isinstance(input, tuple)
#         assert isinstance(target, list) or isinstance(target, tuple)
#         for inp, tar, ovlp in zip(input, target, overlap):
#             for key in inp.keys:
#                 loss += torch.sum(
#                     ((inp.block(key).values - tar.block(key).values) ** 2)
#                     * ovlp.block(key).values
#                 )

#     return loss


# def loss_nonorthogonal_basis_off_diagonal_overlap(
#     input: List[torch.ScriptObject],
#     target: List[torch.ScriptObject],
#     overlap: List[torch.ScriptObject],
#     structure_idxs: List[int],
#     check_metadata: bool = True,
# ) -> torch.Tensor:
#     """
#     Calculates the squared error loss between the input (ML) and target (QM)
#     scalar fields. As these are expanded on the same non-orthogonal basis
#     set, the loss is evaluated using the overlap-type matrix:

#     .. math::

#             L = \sum_ij ∆c_i . \hat{O}_ij . ∆c_j   for all i != j
#     """
#     # First slice the data so that we have one mts.TensorMap per structure, if
#     # necessary
#     if isinstance(input, torch.ScriptObject):
#         if structure_idxs is None:
#             raise ValueError(
#                 "If `input` is a single mts.TensorMap, `structure_idxs` must be passed."
#                 " If passing a mts.TensorMap for a single structure, pass it in a list."
#             )
#         input = [
#             mts.slice(
#                 input,
#                 "samples",
#                 mts.Labels(names=["system"], values=torch.tensor([A]).reshape(-1, 1)),
#             )
#             for A in structure_idxs
#         ]
#     if isinstance(target, torch.ScriptObject):
#         if structure_idxs is None:
#             raise ValueError(
#                 "If `input` is a single mts.TensorMap, `structure_idxs` must be passed."
#                 " If passing a mts.TensorMap for a single structure, pass it in a list."
#             )
#         target = [
#             mts.slice(
#                 target,
#                 "samples",
#                 mts.Labels(names=["system"], values=torch.tensor([A]).reshape(-1, 1)),
#             )
#             for A in structure_idxs
#         ]
#     if isinstance(overlap, torch.ScriptObject):
#         if structure_idxs is None:
#             raise ValueError(
#                 "If `input` is a single mts.TensorMap, `structure_idxs` must be passed."
#                 " If passing a mts.TensorMap for a single structure, pass it in a list."
#             )
#         overlap = [
#             mts.slice(
#                 overlap,
#                 "samples",
#                 mts.Labels(names=["system"], values=torch.tensor([A]).reshape(-1, 1)),
#             )
#             for A in structure_idxs
#         ]

#     assert isinstance(input, list) or isinstance(input, tuple)
#     assert isinstance(target, list) or isinstance(target, tuple)
#     assert isinstance(overlap, list) or isinstance(overlap, tuple)

#     loss = 0
#     for inp, tar, ovl in zip(input, target, overlap):

#         # Check metadata
#         if check_metadata:
#             _check_forward_args(inp, tar, ovl)

#         # Evaluate loss
#         loss += loss_nonorthogonal_basis_full_overlap_one_structure(
#             input=inp, target=tar, overlap=ovl
#         )

#     return loss


# def loss_nonorthogonal_basis_full_overlap(
#     input: List[torch.ScriptObject],
#     target: List[torch.ScriptObject],
#     overlap: List[torch.ScriptObject],
#     structure_idxs: List[int],
#     check_metadata: bool = True,
#     off_diagonal_only: bool = False,
# ) -> torch.Tensor:
#     """
#     Calculates the squared error loss between the input (ML) and target (QM)
#     scalar fields. As these are expanded on the same non-orthogonal basis
#     set, the loss is evaluated using the overlap-type matrix:

#     .. math::

#             L = \sum_ij ∆c_i . \hat{O}_ij . ∆c_j   for all i,j
#     """
#     # First slice the data so that we have one mts.TensorMap per structure, if
#     # necessary
#     if isinstance(input, torch.ScriptObject):
#         if structure_idxs is None:
#             raise ValueError(
#                 "If `input` is a single mts.TensorMap, `structure_idxs` must be passed."
#                 " If passing a mts.TensorMap for a single structure, pass it in a list."
#             )
#         input = [
#             mts.slice(
#                 input,
#                 "samples",
#                 mts.Labels(names=["system"], values=torch.tensor([A]).reshape(-1, 1)),
#             )
#             for A in structure_idxs
#         ]
#     if isinstance(target, torch.ScriptObject):
#         if structure_idxs is None:
#             raise ValueError(
#                 "If `input` is a single mts.TensorMap, `structure_idxs` must be passed."
#                 " If passing a mts.TensorMap for a single structure, pass it in a list."
#             )
#         target = [
#             mts.slice(
#                 target,
#                 "samples",
#                 mts.Labels(names=["system"], values=torch.tensor([A]).reshape(-1, 1)),
#             )
#             for A in structure_idxs
#         ]
#     if isinstance(overlap, torch.ScriptObject):
#         if structure_idxs is None:
#             raise ValueError(
#                 "If `input` is a single mts.TensorMap, `structure_idxs` must be passed."
#                 " If passing a mts.TensorMap for a single structure, pass it in a list."
#             )
#         overlap = [
#             mts.slice(
#                 overlap,
#                 "samples",
#                 mts.Labels(names=["system"], values=torch.tensor([A]).reshape(-1, 1)),
#             )
#             for A in structure_idxs
#         ]

#     assert isinstance(input, list) or isinstance(input, tuple)
#     assert isinstance(target, list) or isinstance(target, tuple)
#     assert isinstance(overlap, list) or isinstance(overlap, tuple)

#     loss = 0
#     for inp, tar, ovl in zip(input, target, overlap):

#         # Check metadata
#         if check_metadata:
#             _check_forward_args(inp, tar, ovl)

#         # Evaluate loss
#         loss += loss_nonorthogonal_basis_full_overlap_one_structure(
#             input=inp, target=tar, overlap=ovl, off_diagonal_only=off_diagonal_only
#         )

#     return loss


# def loss_orthogonal_basis(
#     input: Union[torch.ScriptObject, List[torch.ScriptObject]],
#     target: Union[torch.ScriptObject, List[torch.ScriptObject]],
#     check_metadata: bool = True,
# ) -> torch.Tensor:
#     """
#     Calculates the squared error loss between the input (ML) and target (QM) quantities.
#     These could be any tensorial quantity of arbitrary rank, or a scalar field expanded
#     on an orthogonal basis set.

#     In this case, the loss is evaluated independently for each element in the
#     input/target tensors, and summed to give the total loss.

#     .. math::

#         L = \sum_i (∆c_i)^2

#     """
#     # Check metadata
#     if check_metadata:
#         if isinstance(input, list) or isinstance(input, tuple):
#             for inp, tar in zip(input, target):
#                 _check_forward_args(inp, tar)
#         else:
#             _check_forward_args(input, target)

#     # Use the "sum" reduction method to calculate the loss for each block
#     torch_mse = torch.nn.MSELoss(reduction="sum")
#     loss = 0

#     if isinstance(input, torch.ScriptObject):
#         assert isinstance(target, torch.ScriptObject)
#         for key in input.keys:
#             in_block = input.block(key)
#             tar_block = target.block(key)
#             loss += torch_mse(input=in_block.values, target=tar_block.values)
#     else:
#         assert isinstance(input, list) or isinstance(input, tuple)
#         assert isinstance(target, list) or isinstance(target, tuple)
#         for inp, tar in zip(input, target):
#             for key in inp.keys:
#                 in_block = inp.block(key)
#                 tar_block = tar.block(key)
#                 loss += torch_mse(input=in_block.values, target=tar_block.values)

#     return loss


# # ==================================
# # ====== Helper functions ==========
# # ==================================
