"""
Module containing utility functions for transforming data for use in a metatensor learn
Dataset.
"""

from itertools import product
from typing import List

import torch
import metatensor.torch as mts


# ===== Fxns to mask / unmask tensors =====


def mask_coeff_vector_tensormap(
    vector_tensor: torch.ScriptObject,
    idxs_to_keep: List[int],
    atomic_center_name: str = "center",
) -> torch.ScriptObject:
    """
    Takes a TensorMap corresponding to a vector of coefficients and slices it to only
    contain sample indices in `idxs_to_keep`, essentially masking those indices not
    present. Any blocks that have been sliced to zero samples are dropped. Returns the
    masked TensorMap.

    Assumes that the atomic centers are in the samples and named as
    "{atomic_center_name}".
    """
    # Mask the tensor
    vector_tensor_masked = mts.slice(
        vector_tensor,
        axis="samples",
        labels=mts.Labels(
            names=[atomic_center_name],
            values=torch.tensor(idxs_to_keep).reshape(-1, 1),
        ),
    )

    # Find empty blocks
    keys_to_drop = []
    for key, block in vector_tensor_masked.items():
        if block.values.shape[0] == 0:  # has been sliced to zero samples
            keys_to_drop.append(key)

    # Drop empty blocks
    vector_tensor_masked = mts.drop_blocks(
        vector_tensor_masked,
        keys=mts.Labels(
            names=keys_to_drop[0].names,
            values=torch.tensor([[i for i in k.values] for k in keys_to_drop]),
        ),
    )

    return vector_tensor_masked


def mask_ovlp_matrix_tensormap(
    matrix_tensor: torch.ScriptObject,
    idxs_to_keep: List[int],
    atomic_center_name_prefix: str = "center",
) -> torch.ScriptObject:
    """
    Takes a TensorMap corresponding to a matrix of overlaps and slices it to only
    contain sample indices in `idxs_to_keep`, essentially masking those indices not
    present. Any blocks that have been sliced to zero samples are dropped. Returns the
    masked TensorMap.

    Assumes the pairs of atomic centers are in the samples and named as
    "{atomic_center_name_prefix}_1" and "{atomic_center_name_prefix}_2".
    """
    # Mask the tensor
    matrix_tensor_masked = mts.slice(
        matrix_tensor,
        axis="samples",
        labels=mts.Labels(
            names=[f"{atomic_center_name_prefix}_1", f"{atomic_center_name_prefix}_2"],
            values=torch.tensor(list(product(idxs_to_keep, idxs_to_keep))),
        ),
    )

    # Find empty blocks
    keys_to_drop = []
    for key, block in matrix_tensor_masked.items():
        if block.values.shape[0] == 0:  # has been sliced to zero samples
            keys_to_drop.append(key)

    # Drop empty blocks
    matrix_tensor_masked = mts.drop_blocks(
        matrix_tensor_masked,
        keys=mts.Labels(
            names=keys_to_drop[0].names,
            values=torch.tensor([[i for i in k.values] for k in keys_to_drop]),
        ),
    )

    return matrix_tensor_masked


def unmask_coeff_vector_tensormap(
    tensor_unmasked: torch.ScriptObject, tensor_masked: torch.ScriptObject
) -> torch.ScriptObject:
    """
    Builds a zeros tensor with `mts.zeros_like(tensor_unmasked)` and fills in its block
    values with the values for the samples present in `tensor_masked`.

    `tensor_masked` must be the masked (i.e. with sliced samples) version of
    `tensor_unmasked`, such that every sample in `tensor_masked` is present and can be
    filled in in `tensor_unmasked`.
    """

    # Generate a TensorMap of zeros of the unmasked shape
    tensor_unmasked = mts.zeros_like(tensor_unmasked)

    # Fill in the values from masked TensorMap
    for key, block_masked in tensor_masked.items():

        block_unmasked = tensor_unmasked[key]
        for i, masked_sample in enumerate(block_masked.samples):
            block_unmasked.values[block_unmasked.samples.position(masked_sample)] = (
                block_masked.values[i]
            )

    return tensor_unmasked
