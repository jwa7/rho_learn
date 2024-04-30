"""
Module containing utility functions for transforming data for use in a metatensor learn
Dataset.
"""

from itertools import product
from typing import List, Optional, Union

import numpy as np
import torch

import metatensor.torch as mts


# ===== Fxns to mask / unmask tensors =====


def mask_coeff_vector_tensormap(
    vector_tensor: torch.ScriptObject,
    idxs_to_keep: List[int],
    atomic_center_name: str = "atom",
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
    atomic_center_name_prefix: str = "atom",
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


# ===== Fxns for creating groups of indices for train/test/validation splits


def group_idxs(
    idxs: List[int],
    n_groups: int,
    group_sizes: Optional[Union[List[float], List[int]]] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Returns the indices in `idxs` in `n_groups` groups of indices, according
    to the relative or absolute sizes in `group_sizes`.

    For instance, if `n_groups` is 2 (i.e. for a train/test split), 2 arrays are
    returned. If `n_groups` is 3 (i.e. for a train/test/validation split), 3
    arrays are returned.

    If `group_sizes` is None, the group sizes returned are (to the nearest
    integer) equally sized for each group. If `group_sizes` is specified as a
    List of floats (i.e. relative sizes, whose sum is <= 1), the group sizes
    returned are converted to absolute sizes, i.e. multiplied by `n_indices`. If
    `group_sizes` is specified as a List of int, the group sizes returned
    are the absolute sizes specified.

    If `shuffle` is False, no shuffling of `idxs` is performed. If true, and
    `seed` is not None, `idxs` is shuffled using `seed` as the seed for the
    random number generator. If `seed` is None, the random number generator is
    not manually seeded.
    """
    # Check that group sizes are valid
    if group_sizes is not None:
        if len(group_sizes) != n_groups:
            raise ValueError(
                f"Length of group_sizes ({len(group_sizes)}) must match n_groups ({n_groups})."
            )

    # Create a copy of the indices so that shuffling doesn't affect the original
    idxs = np.array(idxs).copy()

    # Shuffle indices if seed is specified
    if shuffle:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(idxs)

    # Get absolute group sizes for train/test/validation split
    abs_group_sizes = get_group_sizes(n_groups, len(idxs), group_sizes)

    if np.sum(abs_group_sizes) != len(idxs):
        raise ValueError("sum of group sizes not equal to len of `idxs` passed")

    # Grouped the indices
    grouped_idxs = []
    prev_size = 0
    for size in abs_group_sizes:
        grouped_idxs.append(idxs[prev_size : size + prev_size])
        prev_size += size

    # Check that there are no intersections between the groups
    ref_group = grouped_idxs[0]
    for group in grouped_idxs[1:]:
        assert len(np.intersect1d(ref_group, group)) == 0

    return grouped_idxs


def get_group_sizes(
    n_groups: int,
    n_indices: int,
    group_sizes: Optional[Union[List[float], List[int]]] = None,
) -> np.ndarray:
    """
    Parses the `group_sizes` arg and returns an array of group sizes in absolute
    terms. If `group_sizes` is None, the group sizes returned are (to the
    nearest integer) evenly distributed across the number of unique indices;
    i.e. if there are 12 unique indices (`n_indices=10`), and `n_groups` is 3,
    the group sizes returned will be np.array([4, 4, 4]).

    If `group_sizes` is specified as a List of floats (i.e. relative sizes,
    whose sum is <= 1), the group sizes returned are converted to absolute
    sizes, i.e. multiplied by `n_indices`. If `group_sizes` is specified as a
    List of int, no conversion is performed. A cascade round is used to make
    sure that the group sizes are integers, with the sum of the List
    preserved and the rounding error minimized.

    :param n_groups: an int, the number of groups to split the data into :param
        n_indices: an int, the number of unique indices present in the data by
        which the data should be grouped.
    :param n_indices: a :py:class:`int` for the number of unique indices present
        in the input data for the specified `axis` and `names`.
    :param group_sizes: a sequence of :py:class:`float` or :py:class:`int`
        indicating the absolute or relative group sizes, respectively.

    :return: a :py:class:`numpy.ndarray` of :py:class:`int` indicating the
        absolute group sizes.
    """
    if group_sizes is None:  # equally sized groups
        group_sizes = np.array([1 / n_groups] * n_groups) * n_indices
    elif np.all([isinstance(size, int) for size in group_sizes]):  # absolute
        group_sizes = np.array(group_sizes)
    else:  # relative; List of float
        group_sizes = np.array(group_sizes) * n_indices

    # The group sizes may not be integers. Use cascade rounding to round them
    # all to integers whilst attempting to minimize rounding error.
    group_sizes = _cascade_round(group_sizes)

    return group_sizes


def _cascade_round(array: np.ndarray) -> np.ndarray:
    """
    Given an array of floats that sum to an integer, this rounds the floats
    and returns an array of integers with the same sum.
    Adapted from https://jsfiddle.net/cd8xqy6e/.
    """
    # Check type
    if not isinstance(array, np.ndarray):
        raise TypeError("must pass `array` as a numpy array.")
    # Check sum
    mod = np.sum(array) % 1
    if not np.isclose(round(mod) - mod, 0):
        raise ValueError("elements of `array` must sum to an integer.")

    float_tot, integer_tot = 0, 0
    rounded_array = []
    for element in array:
        new_int = round(element + float_tot) - integer_tot
        float_tot += element
        integer_tot += new_int
        rounded_array.append(new_int)

    # Check that the sum is preserved
    assert round(np.sum(array)) == round(np.sum(rounded_array))

    return np.array(rounded_array)


def get_log_subset_sizes(
    n_max: int,
    n_subsets: int,
    base: Optional[float] = 10.0,
) -> np.array:
    """
    Returns an ``n_subsets`` length array of subset sizes equally spaced along a
    log of specified ``base`` (default base 10) scale from 0 up to ``n_max``.
    Elements of the returned array are rounded to integer values. The final
    element of the returned array may be less than ``n_max``.
    """
    # Generate subset sizes evenly spaced on a log scale, custom base
    subset_sizes = np.logspace(
        np.log(n_max / n_subsets) / np.log(base),
        np.log(n_max) / np.log(base),
        num=n_subsets,
        base=base,
        endpoint=True,
        dtype=int,
    )
    return subset_sizes