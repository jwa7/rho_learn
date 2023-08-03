"""
Module for creating a custom torch datasets specifically for density data.
"""
import os
from typing import Sequence, Optional, Tuple, Union, List

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

import equistore
from equistore import TensorMap

# Consider dataloading
# - num_workers * prefetch_factor * batch_size * (size per structure) < RAM


class RhoData(torch.utils.data.Dataset):
    """
    Custom torch Dataset class for loading input/descriptors (i.e. lambda-SOAP)
    and output/features (i.e. electron density coefficients) data from disk. If
    evaluating the loss of non-orthogonal basis functions using overlap-type
    matrices, the directory containing these matrices must also be specified.

    In each of the directories `input_dir`, `output_dir`, and `overlap_dir`
    (though they could be the same directory), there must be subdirectories
    named according to the structure indices in `structure_idxs`. The
    descriptors, coefficients, and overlap matrices (if applicable) must be
    stored in equistore TensorMap format under respective filenames "x.npz",
    "c.npz", and "s.npz".

    :param structure_idxs: Sequence[int], Sequence of structure indices that define the
        complete dataset.
    :param input_dir: str, absolute path to directory containing
        input/descriptor (i.e. lambda-SOAP) data. In this directory, descriptors
        must be stored at relative path A/x.npz, where A is the structure index.
    :param output_dir: str, absolute path to directory containing
        output/features (i.e. electron density coefficients) data. In this
        directory, coefficients must be stored at relative path A/c.npz, where A
        is the structure index.
    :param overlap_dir: str, absolute path to directory containing overlap-type
        matrices. Optional, only required if evaluating the loss of
        non-orthogonal basis functions. In this directory, coefficients must be
        stored at relative path A/s.npz, where A is the structure index.
    :param out_invariant_means_path: str, optional. Absolute path to file containing the
        invariant means of the output/features (i.e. electron density).
    **torch_kwargs: dict of kwargs for loading TensorMaps to torch backend.
        `dtype`, `device`, and `requires_grad` are required.
    """

    def __init__(
        self,
        structure_idxs: Sequence[int],
        input_dir: str,
        output_dir: str,
        overlap_dir: Optional[str] = None,
        out_invariant_means_path: Optional[str] = None,
        **torch_kwargs,
    ):
        super(RhoData, self).__init__()

        # Check input args
        RhoData._check_input_args(input_dir, output_dir, overlap_dir)

        # Assign attributes
        self.structure_idxs = structure_idxs
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.overlap_dir = overlap_dir
        self.out_invariant_means_path = out_invariant_means_path
        self.dtype = torch_kwargs.get("dtype")
        self.device = torch_kwargs.get("device")
        self.requires_grad = torch_kwargs.get("requires_grad")

        # Set the data directories and check files exist
        self.in_paths = [
            os.path.join(self.input_dir, f"{structure_idx}/x.npz")
            for structure_idx in self.structure_idxs
        ]
        for path in self.in_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Descriptor coeffs at {path} does not exist.")
        self.out_paths = [
            os.path.join(self.output_dir, f"{structure_idx}/c.npz")
            for structure_idx in self.structure_idxs
        ]
        for path in self.out_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Features / electron density coeffs at {path} does not exist."
                )
        if overlap_dir is not None:
            self.overlap_paths = [
                os.path.join(self.overlap_dir, f"{structure_idx}/s.npz")
                for structure_idx in self.structure_idxs
            ]
            for path in self.overlap_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Overlap matrix at {path} does not exist.")

    @staticmethod
    def _check_input_args(
        input_dir: str, output_dir: str, overlap_dir: Optional[str] = None
    ):
        """Checks args to the constructor."""
        if not os.path.exists(input_dir):
            raise NotADirectoryError(
                f"Input/descriptor data directory {input_dir} does not exist."
            )
        if not os.path.exists(output_dir):
            raise NotADirectoryError(
                f"Input/descriptor data directory {output_dir} does not exist."
            )
        if overlap_dir is not None:
            if not os.path.exists(overlap_dir):
                raise NotADirectoryError(
                    f"Input/descriptor data directory {overlap_dir} does not exist."
                )

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.structure_idxs)

    def __getitem__(self, structure_idx: int) -> TensorMap:
        """
        Loads and returns the input/output data pair for the corresponding
        `structure_idx`. Input, output, and overlap matrices (if applicable) are
        loaded with a torch backend.
        """
        input = equistore.core.io.load_custom_array(
            self.in_paths[structure_idx],
            create_array=equistore.core.io.create_torch_array,
        )
        output = equistore.core.io.load_custom_array(
            self.out_paths[structure_idx],
            create_array=equistore.core.io.create_torch_array,
        )

        # Return input/output pair if no overlap matrix required
        if self.overlap_dir is None:
            return structure_idx, input, output

        # Return overlap matrix if applicable
        overlap = equistore.core.io.load_custom_array(
            self.overlap_paths[structure_idx],
            create_array=equistore.core.io.create_torch_array,
        )
        return structure_idx, input, output, overlap

    def get_out_invariant_means(self) -> TensorMap:
        """
        Returns a TensorMap of the output (i.e. elctron density) invariant means.
        
        Only applicable if the RhoData object was initialized with the path
        containing this file.
        """
        if self.out_invariant_means_path is None:
            raise ValueError(
                "Output invariant means path not provided to RhoData constructor."
            )
        return equistore.core.io.load_custom_array(
            self.out_invariant_means_path,
            create_array=equistore.core.io.create_torch_array,
        )


class RhoLoader:
    """
    Class for loading data from the custom RhoData dataset class.
    """

    def __init__(
        self,
        dataset,
        subset_idxs: np.ndarray,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        self.dataset = dataset
        batch_size = batch_size if batch_size is not None else len(subset_idxs)

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_rho_data_batch,
            sampler=SubsetRandomSampler(subset_idxs),
            **kwargs,
        )

    def collate_rho_data_batch(self, batch):
        """
        Takes a list of lists where each element corresponding to the RhoData for a
        single structure index, of the form:

            List[
                # structure_idx, input, output
                List[int, TensorMap, TensorMap], # structure i
                List[int, TensorMap, TensorMap], # structure j
                ...
            ]

        or, if overlap matrices are included, of the form:

            List[
                # structure_idx, input, output, overlap
                List[int, TensorMap, TensorMap, TensorMap], # structure i
                List[int, TensorMap, TensorMap, TensorMap], # structure j
                ...
            ]

        and returns a list of lists of the form:

            List[
                List[int, int, ...] # structure_idx_i, structure_idx_j, ...
                List[TensorMap, TensorMap, ...] # input_i, input_j, ...
                List[TensorMap, TensorMap, ...] # output_i, output_j, ...
            ]

        or, if overlap matrices are included:

            List[
                List[int, int, ...] # structure_idx_i, structure_idx_j, ...
                List[TensorMap, TensorMap, ...] # input_i, input_j, ...
                List[TensorMap, TensorMap, ...] # outputs
                List[TensorMap, TensorMap, ...] # output_i, output_j, ...
            ]
        """
        # Zip the batch such that data is grouped by the data they represent
        # rather than by structure idx
        batch = list(zip(*batch))

        return batch

    def __iter__(self):
        """Returns an iterable for the dataloader."""
        return iter(self.dataloader)


# ===== Fxns for creating groups of indices for train/test/validation splits


def group_idxs(
    all_idxs: Sequence[int],
    n_groups: int,
    group_sizes: Optional[Union[Sequence[float], Sequence[int]]] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Sequence[np.ndarray]:
    """
    Returns the indices in `all_idxs` in `n_groups` groups of indices, according
    to the relative or absolute sizes in `group_sizes`.

    For instance, if `n_groups` is 2 (i.e. for a train/test split), 2 arrays are
    returned. If `n_groups` is 3 (i.e. for a train/test/validation split), 3
    arrays are returned.

    If `group_sizes` is None, the group sizes returned are (to the nearest
    integer) equally sized for each group. If `group_sizes` is specified as a
    Sequence of floats (i.e. relative sizes, whose sum is <= 1), the group sizes
    returned are converted to absolute sizes, i.e. multiplied by `n_indices`. If
    `group_sizes` is specified as a Sequence of int, the group sizes returned
    are the absolute sizes specified.

    If `shuffle` is False, no shuffling of `all_idxs` is performed. If true, and
    `seed` is not None, `all_idxs` is shuffled using `seed` as the seed for the
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
    idxs = np.array(all_idxs).copy()

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
    group_sizes: Optional[Union[Sequence[float], Sequence[int]]] = None,
) -> np.ndarray:
    """
    Parses the `group_sizes` arg and returns an array of group sizes in absolute
    terms. If `group_sizes` is None, the group sizes returned are (to the
    nearest integer) evenly distributed across the number of unique indices;
    i.e. if there are 12 unique indices (`n_indices=10`), and `n_groups` is 3,
    the group sizes returned will be np.array([4, 4, 4]). If `group_sizes` is
    specified as a Sequence of floats (i.e. relative sizes, whose sum is <= 1),
    the group sizes returned are converted to absolute sizes, i.e. multiplied by
    `n_indices`. If `group_sizes` is specified as a Sequence of int, no
    conversion is performed. A cascade round is used to make sure that the group
    sizes are integers, with the sum of the Sequence preserved and the rounding
    error minimized.

    :param n_groups: an int, the number of groups to split the data into :param
        n_indices: an int, the number of unique indices present in the data by
        which the data should be grouped.
    :param n_indices: a :py:class:`int` for the number of unique indices present
        in the input data for the specified `axis` and `names`.
    :param group_sizes: a sequence of :py:class:`float` or
        :py:class:`int` indicating the absolute or relative group sizes,
        respectively.

    :return: a :py:class:`numpy.ndarray` of :py:class:`int` indicating the
        absolute group sizes.
    """
    if group_sizes is None:  # equally sized groups
        group_sizes = np.array([1 / n_groups] * n_groups) * n_indices
    elif np.all([isinstance(size, int) for size in group_sizes]):  # absolute
        group_sizes = np.array(group_sizes)
    else:  # relative; Sequence of float
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


# Convert structure idxs to torch tensor, convert TensorMaps to torch
# backend
# return [torch.tensor(batch[0])] + [
#     [
#         equistore.to(
#             tensormap,
#             backend="torch",
#             dtype=self.dataset.dtype,
#             device=self.dataset.device,
#             requires_grad=self.dataset.requires_grad,
#         )
#         for tensormap in tensormap_list
#     ]
#     for tensormap_list in batch[1:]
# ]
