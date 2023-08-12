"""
Module for creating a custom torch datasets specifically for density data.
"""
import gc
import os
from typing import Sequence, Optional, Tuple, Union, List

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

import equistore
from equistore import Labels, TensorBlock, TensorMap

# Consider dataloading
# - num_workers * prefetch_factor * batch_size * (size per structure) < RAM


# Define a filename convention for the input/output/overlap TensorMaps. In
# general, for each structure indexed by A, these files would live at relative
# paths A/x.npz, A/c.npz, and A/s.npz, respectively.
FILENAMES = ["x", "c", "s"]


class RhoData(torch.utils.data.Dataset):
    """
    Custom torch Dataset class for loading input/descriptors (i.e. lambda-SOAP)
    and output/features (i.e. electron density coefficients) data from disk. If
    evaluating the loss of non-orthogonal basis functions using overlap-type
    matrices, the directory containing these matrices must also be specified.

    In each of the directories `input_dir`, `output_dir`, and `overlap_dir`
    (though they could be the same directory), there must be subdirectories
    named according to the indices in `idxs`. The descriptors, coefficients, and
    overlap matrices (if applicable) must be stored under the respective
    filenames passed in `filenames`, in equistore TensorMap (.npz) format.

    :param idxs: Sequence[int], Sequence of indices that define the complete
        dataset.
    :param input_dir: str, absolute path to directory containing
        input/descriptor (i.e. lambda-SOAP) data. In this directory, descriptors
        must be stored at relative path A/{filenames[0]}.npz, where A is the
        structure index.
    :param output_dir: str, absolute path to directory containing
        output/features (i.e. electron density coefficients) data. In this
        directory, coefficients must be stored at relative path
        A/{filenames[1]}.npz, where A is the structure index.
    :param overlap_dir: str, absolute path to directory containing overlap-type
        matrices. Optional, only required if evaluating the loss of
        non-orthogonal basis functions. In this directory, coefficients must be
        stored at relative path A/{filenames[2]}.npz, where A is the structure
        index.
    :param keep_in_mem: bool, whether to keep the data in memory. If true, all
        data is lazily loaded upon initialization and stored in a dict indexed
        by `idxs`. When __getitem__ is called, the dict is just accessed from
        memory. If false, data is loaded from disk upon each call to
        __getitem__.
    :param standardize_invariants: Sequence[str], Sequence of strings indicating
        which data to standardize. If "input" is in the list, the invariant
        blocks of the input data are standardized using the invariant means of
        the training data. If "output" is in the list, the invariant blocks of
        the output data are standardized. In either or both of these cases,
        `train_idxs` should be specified. If None, no standardization is
        performed.
    :param train_idxs: Sequence[int], Sequence of indices corresponding to the
        training data. Only required if `standardize_invariants` is not None.
    :param filenames: Sequence[str], Sequence of strings indicating the filename
        convention for the input/output/overlap TensorMaps respectively. By
        default the filenames are set to ["x", "c", "s"], meaning for each
        structure indexed by A, these files would live at relative paths
        A/x.npz, A/c.npz, and A/s.npz, respectively.
    **torch_kwargs: dict of kwargs for loading TensorMaps to torch backend.
        `dtype`, `device`, and `requires_grad` are required.
    """

    def __init__(
        self,
        idxs: Sequence[int],
        input_dir: str,
        output_dir: str,
        overlap_dir: Optional[str] = None,
        keep_in_mem: bool = True,
        standardize_invariants: Optional[Sequence[List]] = None,
        train_idxs: Optional[Sequence[int]] = None,
        filenames: Sequence[str] = FILENAMES,
        **torch_kwargs,
    ):
        super(RhoData, self).__init__()

        # Check input args
        RhoData._check_input_args(input_dir, output_dir, overlap_dir)

        # Assign attributes
        self._idxs = idxs
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._overlap_dir = overlap_dir
        self._filenames = filenames
        self._torch_kwargs = torch_kwargs

        # Set the data directories and check files exist
        self.in_paths = {
            idx: os.path.join(self._input_dir, f"{idx}/{filenames[0]}.npz")
            for idx in self._idxs
        }
        for path in self.in_paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Descriptor coeffs at {path} does not exist.")
        self.out_paths = {
            idx: os.path.join(self._output_dir, f"{idx}/{filenames[1]}.npz")
            for idx in self._idxs
        }
        for path in self.out_paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Features / electron density coeffs at {path} does not exist."
                )
        if overlap_dir is not None:
            self.overlap_paths = {
                idx: os.path.join(self._overlap_dir, f"{idx}/{filenames[2]}.npz")
                for idx in self._idxs
            }
            for path in self.overlap_paths.values():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Overlap matrix at {path} does not exist.")

        # Lazily load data upon initialization if requested
        if keep_in_mem:
            self._load_data()
            self._data_in_mem = True
        else:
            self._data_in_mem = False

        # Calculate the means of the invariant blocks of training data if
        # standardization is to be performed
        if standardize_invariants is not None:
            if train_idxs is None:
                raise ValueError(
                    "`train_idxs` must be specified if standardizing invariants."
                )
            self._calculate_invariant_means(
                standardize_what=standardize_invariants, train_idxs=train_idxs
            )
            self._standardize_invariants(standardize_what=standardize_invariants)

        gc.collect()

    @staticmethod
    def _check_input_args(
        input_dir: str,
        output_dir: str,
        overlap_dir: Optional[str] = None,
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

    def __loaditem__(self, idx: int) -> Tuple:
        """
        Loads a data item for the corresponding `idx` from disk and return it in
        a tuple. Returns the idx, and input and output TensorMaps, as well as
        the overlap TensorMap if applicable.
        """
        input = equistore.core.io.load_custom_array(
            self.in_paths[idx],
            create_array=equistore.core.io.create_torch_array,
        )
        output = equistore.core.io.load_custom_array(
            self.out_paths[idx],
            create_array=equistore.core.io.create_torch_array,
        )
        input = equistore.to(input, "torch", **self._torch_kwargs)
        output = equistore.to(output, "torch", **self._torch_kwargs)

        # Return input/output pair if no overlap matrix required
        if self._overlap_dir is None:
            return idx, input, output

        # Return overlap matrix if applicable
        overlap = equistore.core.io.load_custom_array(
            self.overlap_paths[idx],
            create_array=equistore.core.io.create_torch_array,
        )
        overlap = equistore.to(overlap, "torch", **self._torch_kwargs)
        return idx, input, output, overlap

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self._idxs)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Loads and returns the input/output data pair for the corresponding
        `idx`. Input, output, and overlap matrices (if applicable) are
        loaded with a torch backend.
        """
        if self._data_in_mem:
            return self._loaded_data[idx]
        return self.__loaditem__(idx)

    def _load_data(self) -> None:
        """
        Lazily loads the data for all items corresponding to `idxs` upon class
        initialization. Stores it in a dict indexed by `self._idxs`.

        Each item correpsonds to a tuple of the idx, and TensorMaps for the
        input, output, and (if applicable) overlap matrices.
        """
        self._loaded_data = {}
        for idx in self._idxs:
            self._loaded_data[idx] = self.__loaditem__(idx)

    def _calculate_invariant_means(
        self, standardize_what: Sequence[str], train_idxs: Sequence[int]
    ) -> None:
        """
        Retrieves the `data="input"` or `data="output"` TensorMaps for the
        `train_idxs`, joins them, then only for the invariant blocks performs a
        reduction by taking the mean over samples. Stores the resulting
        TensorMap in `self._in_invariant_means` or `self._out_invariant_means`,
        """
        if "input" in standardize_what:
            # Retrieve the input training data and join
            in_train_list = [self[idx][1] for idx in train_idxs]
            in_train = equistore.join(
                in_train_list,
                axis="samples",
                remove_tensor_name=True,
            )
            in_train = equistore.to(in_train, "numpy")
            # Calc and store invariant means
            in_inv_means = get_invariant_means(in_train)
            in_inv_means = equistore.to(in_inv_means, "torch", **self._torch_kwargs)
            self._in_invariant_means = in_inv_means
            # Clear memory
            del in_train_list, in_train, in_inv_means
            gc.collect()

        if "output" in standardize_what:
            # Retrieve the output training data and join
            out_train_list = [self[idx][2] for idx in train_idxs]
            out_train = equistore.join(
                out_train_list,
                axis="samples",
                remove_tensor_name=True,
            )
            out_train = equistore.to(out_train, "numpy")
            # Calc and store invariant means
            out_inv_means = get_invariant_means(out_train)
            out_inv_means = equistore.to(out_inv_means, "torch", **self._torch_kwargs)
            self._out_invariant_means = out_inv_means
            # Clear memory
            del out_train_list, out_train, out_inv_means
            gc.collect()

    def _standardize_invariants(self, standardize_what: Sequence[str]) -> None:
        """
        Standardizes the invariant blocks of the input and/or putout data.

        If `self.keep_in_mem=True`, the attribute appropraite data stored in
        `self._loaded_data` is overwritten with the standardized data.

        If `self.keep_in_mem=False`, the standardized data is written to file
        alongside the unstandardized data. In this case, standardized inputs are
        stored in `self._input_dir` under filenames "{filenames[0]}_std.npz", and
        standardized outputs are stored in `self._output_dir` under filenames
        "{filenames[0]}_std.npz".
        """
        if len(standardize_what) == 0:
            return
        # Standardize and overwrite the unstd data already in memory
        if self._data_in_mem:
            for idx in self._idxs:
                item = self._loaded_data[idx]
                tmp_input, tmp_output = item[1], item[2]
                if "input" in standardize_what:
                    tmp_input = standardize_invariants(
                        item[1],
                        invariant_means=self._in_invariant_means,
                        reverse=False,
                    )
                if "output" in standardize_what:
                    tmp_output = standardize_invariants(
                        item[2],
                        invariant_means=self._out_invariant_means,
                        reverse=False,
                    )
                if len(item) == 4:
                    self._loaded_data[idx] = (item[0], tmp_input, tmp_output, item[3])
                else:
                    assert len(item) == 3
                    self._loaded_data[idx] = (item[0], tmp_input, tmp_output)
                del item, tmp_input, tmp_output
                gc.collect()

        # Calculate standardized data and write newly to file
        else:
            for idx in self._idxs:
                if "input" in standardize_what:
                    in_std = standardize_invariants(
                        self[idx][1],
                        invariant_means=self._in_invariant_means,
                        reverse=False,
                    )
                    equistore.save(
                        os.path.join(
                            self._input_dir, f"{idx}/{self._filenames[0]}_std.npz"
                        )
                    )
                    del in_std
                    gc.collect()

                if "output" in standardize_what:
                    out_std = standardize_invariants(
                        self[idx][2],
                        invariant_means=self._out_invariant_means,
                        reverse=False,
                    )
                    equistore.save(
                        os.path.join(
                            self._output_dir, f"{idx}/{self._filenames[1]}_std.npz"
                        )
                    )
                    del in_std
                    gc.collect()

    @property
    def in_invariant_means(self) -> TensorMap:
        """
        Returns a TensorMap of the output (i.e. elctron density) invariant means.

        Only applicable if the RhoData object was initialized with the path
        containing this file.
        """
        return self._in_invariant_means

    @property
    def out_invariant_means(self) -> TensorMap:
        """
        Returns a TensorMap of the output (i.e. elctron density) invariant means.

        Only applicable if the RhoData object was initialized with the path
        containing this file.
        """
        return self._out_invariant_means


class RhoLoader:
    """
    Class for loading data from the RhoData dataset class.
    """

    def __init__(
        self,
        dataset,
        idxs: np.ndarray,
        batch_size: Optional[int] = None,
        get_overlaps: bool = False,
        **kwargs,
    ):
        self.dataset = dataset
        self._idxs = idxs
        self.batch_size = batch_size if batch_size is not None else len(idxs)
        self.get_overlaps = get_overlaps

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_rho_data_batch,
            sampler=SubsetRandomSampler(idxs),
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
        # rather than by idx
        batch = list(zip(*batch))

        # Return the idxs, input, output, and overlaps
        if self.get_overlaps:
            return (
                batch[0],
                equistore.join(batch[1], axis="samples", remove_tensor_name=True),
                equistore.join(batch[2], axis="samples", remove_tensor_name=True),
                equistore.join(batch[3], axis="samples", remove_tensor_name=True),
            )
        # Otherwise, just return the idxs and the input/output pair
        return (
            batch[0],
            equistore.join(batch[1], axis="samples", remove_tensor_name=True),
            equistore.join(batch[2], axis="samples", remove_tensor_name=True),
        )

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
    using the `equistore.mean_over_samples` function. Returns the result in a
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
    inv_tensor = equistore.drop_blocks(tensor, keys=keys_to_drop)

    # Find the mean over sample for the invariant blocks
    return equistore.mean_over_samples(inv_tensor, sample_names=inv_tensor.sample_names)


def standardize_invariants(
    tensor: TensorMap, invariant_means: TensorMap, reverse: bool = False
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
                if reverse:  # add the mean to the values
                    new_block.values[..., p] += invariant_means[key].values[..., p]
                else:  # subtract
                    new_block.values[..., p] -= invariant_means[key].values[..., p]
            new_blocks.append(new_block)
        else:  # Don't standardize
            new_blocks.append(tensor[key].copy())

    return TensorMap(new_keys, new_blocks)
