"""
Utility functions for cross-validation and computing
means and standard deviations of `metatensor-learn` `Dataset` and
`IndexedDataset` objects.
"""
from typing import List, Optional, Union

import numpy as np
import torch

import metatensor
import metatensor.learn.data
from metatensor import Labels, TensorMap

from rholearn import loss


def get_dataset_invariant_means(
    dataset: metatensor.learn.data.dataset._BaseDataset,
    field: str, 
    torch_kwargs: dict
) -> TensorMap:
    """
    Joins all samples from `dataset` for the specified data `field`, extracts
    the invariant blocks (i.e. where key "spherical_harmonics_l" = 0) and
    returns a TensorMap containing the means these invariant features.

    As typically these invariant means are used for non-learnable biases in an
    model, the torch tensors are set with requires_grad to false.
    """
    # Join all the data into a single TensorMap
    tensor = metatensor.join(
        [getattr(dataset[idx], field) for idx in dataset._indices],
        axis="samples",
        different_keys="union",
        remove_tensor_name=True,
    )
    # Convert to numpy
    tensor = metatensor.detach(tensor).to(arrays="numpy")

    # Get invariant means
    inv_means = get_invariant_means(tensor)

    # Convert back to torch and return
    inv_means = inv_means.to(arrays="torch")
    inv_means = inv_means.to(**torch_kwargs)
    inv_means = metatensor.requires_grad(inv_means, False)
    return inv_means


def get_standard_deviation(
    dataset: metatensor.learn.data.dataset._BaseDataset,
    field: str,
    torch_kwargs: dict,
    invariant_baseline: Optional[TensorMap] = None,
    overlap_field: Optional[str] = None,
) -> float:
    """
    Returns the standard deviation of the data indexed by `idxs` relative to
    some baseline. If ``invariant_baseline`` is passed, this is used.
    Otherwise, the mean invariant features are calculated.

    If "which_data" is "input", the mean invariant features of the input
    data are calculated. If "which_data" is "output", the mean invariant
    features of the output data are calculated.

    By default, the standard deviation is calculated as the square root of
    the sum of square residuals between the data and the baseline:

    .. math:

        \sqrt{
            \frac{1}{N - 1} \sum_A^N ( \Delta \bar{c}^2 )
        }

    where A is 1 of N training structures, and \Delta \bar{c_A} = c_A - c^0
    is the data for structure minus the baseline (itself perhaps an average
    over all structures).

    If `which_data` is "output", then `use_overlaps` can be set to true. In
    this case, the data is assumed to be non-orthogonal basis fxn
    coefficients, where the overlaps as used to evaluate the loss on the
    real space quantity they expand. In this case, the standard deviation is
    calculated as:

    .. math:

        \sqrt{
            \frac{1}{N - 1} \sum_A^N ( \Delta \bar{c_A} \hat{S_A} \Delta
            \bar{c_A}
        }

    where \hat{S_A} is the overlap matrix for structure A.
    """
    # Calculate the invariant baseline if not passed
    if invariant_baseline is None:
        invariant_baseline = get_dataset_invariant_means(
            dataset, field, torch_kwargs
        )

    # We can use the RhoLoss function to evaluate the error in the
    # output data relative to the mean baseline of the training data
    loss_fn = loss.L2Loss()

    # As this function is only ever called once for a given set of training
    # indices, iterate over each training structure in turn to reduce memory
    # overhead
    stddev = 0
    for idx in dataset._indices:
        # Load/retrieve the output training structure and overlap matrix
        tensor = getattr(dataset[idx], field)
        overlap = getattr(dataset[idx], overlap_field) if overlap_field else None

        # If the data field hasn't been standardized, do it
        output_std = standardize_invariants(
            tensor, invariant_baseline, add_baseline=False  # i.e. subtract
        )

        # Build a dummy TensorMap of zeros - this will be used for input
        # into the RhoLoss loss function as the 'target'
        out_zeros = metatensor.zeros_like(output_std)

        # Accumulate the 'loss' on the training ouput relative to the
        # spherical baseline
        stddev += loss_fn(
            input=[output_std], 
            target=[out_zeros], 
            overlap=[overlap], 
            structure_idxs=[idx],
        )

    # Return the standard deviation
    return torch.sqrt(stddev / (len(dataset._indices) - 1))


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


def get_invariant_means(tensor: TensorMap) -> TensorMap:
    """
    Calculates the mean of the invariant (l=0) blocks on the input `tensor`
    using the `metatensor.mean_over_samples` function. Returns the result in a
    new TensorMap, whose number of blocks is equal to the number of invariant
    blocks in `tensor`.

    Assumes `tensor` is a numpy-based TensorMap.
    """
    # Define the keys of the covariant blocks
    keys_to_drop = Labels(
        names=tensor.keys.names,
        values=tensor.keys.values[tensor.keys.column("spherical_harmonics_l") != 0],
    )

    # Drop these blocks
    inv_tensor = metatensor.drop_blocks(tensor, keys=keys_to_drop)

    # Find the mean over sample for the invariant blocks
    return metatensor.mean_over_samples(
        inv_tensor, sample_names=inv_tensor.sample_names
    )


def standardize_invariants(
    tensor: TensorMap, invariant_means: TensorMap, add_baseline: bool = False
) -> TensorMap:
    """
    Standardizes the invariant (l=0) blocks on the input `tensor` by subtracting
    from each coefficient the mean of the coefficients belonging to that
    feature. Returns a new TensorMap.

    Must pass the TensorMap containing the means of the features,
    `invariant_means`. If `reverse` is true, the mean is instead added back to
    the coefficients of each feature.
    """
    new_keys = tensor.keys
    new_blocks = []
    # Iterate over the invariant keys
    for key in new_keys:
        if key in invariant_means.keys:  # standardize
            # Copy the block
            new_block = tensor[key].copy()
            # Manipulate values of copied block in place
            for p in range(len(new_block.properties)):
                if add_baseline:  # add the mean to the values
                    new_block.values[..., p] += invariant_means[key].values[..., p]
                else:  # subtract
                    new_block.values[..., p] -= invariant_means[key].values[..., p]
            new_blocks.append(new_block)
        else:  # Don't standardize
            new_blocks.append(tensor[key].copy())

    return TensorMap(new_keys, new_blocks)
