"""
Module for creating metatensor/torch datasets and dataloaders.
"""
import gc
import os
from typing import Sequence, Optional, Tuple, Union, List

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

import metatensor
from metatensor import Labels, TensorBlock, TensorMap

from rholearn import loss


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
    filenames passed in `filenames`, in metatensor TensorMap (.npz) format.

    `calc_out_train_inv_means` passed as True will calculate the mean baseline
    of the invariant blocks of the output training data.

    `calc_out_train_std_dev` passed as True will calculate the standard
    deviation of the training data, i.e. the L2 loss of the output training
    *density* relative to the baseline training density.

    :param idxs: Sequence[int], Sequence of indices that define the complete
        dataset.
    :param train_idxs: Sequence[int], Sequence of indices corresponding to the
        training data.
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
    :param calc_out_train_inv_means: bool indicating whether or not to calculate
        the means of invariant blocks of the output training data.
    :param calc_out_train_std_dev: bool indicating whether to calculate the
        standard deviation of the output training data. If true, the standard
        deviation is calculated and stored in the attribute `out_train_std_dev`.
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
        all_idxs: Sequence[int],
        train_idxs: Sequence[int],
        input_dir: str,
        output_dir: str,
        overlap_dir: Optional[str] = None,
        keep_in_mem: bool = True,
        calc_out_train_inv_means: bool = True,
        calc_out_train_std_dev: bool = False,
        filenames: Optional[Sequence[str]] = ["lsoap", "ri_coeffs", "ri_ovlp"],
        **torch_kwargs,
    ):
        super(RhoData, self).__init__()

        # Check input args
        RhoData._check_input_args(
            all_idxs,
            train_idxs,
            input_dir,
            output_dir,
            overlap_dir,
            calc_out_train_std_dev,
        )

        # Assign attributes
        self._all_idxs = all_idxs
        self._train_idxs = train_idxs
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._overlap_dir = overlap_dir
        self._filenames = filenames
        self._torch_kwargs = torch_kwargs

        # Set the data directories and check files exist
        self.in_paths = {
            idx: os.path.join(self._input_dir, f"{idx}/{filenames[0]}.npz")
            for idx in self._all_idxs
        }
        for path in self.in_paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Descriptor coeffs at {path} does not exist.")
        self.out_paths = {
            idx: os.path.join(self._output_dir, f"{idx}/{filenames[1]}.npz")
            for idx in self._all_idxs
        }
        for path in self.out_paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Features / electron density coeffs at {path} does not exist."
                )
        if overlap_dir is not None:
            self.overlap_paths = {
                idx: os.path.join(self._overlap_dir, f"{idx}/{filenames[2]}.npz")
                for idx in self._all_idxs
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

        # If standardizing the outputs or calculating the standard deviation of
        # the output training data, the invariant means of the output training
        # data will be required. Calculate and store these.
        if calc_out_train_inv_means or calc_out_train_std_dev:
            self._calculate_out_train_inv_means()

        # Calculate the standard deviation of the output training data
        if calc_out_train_std_dev:
            self._calculate_out_train_std_dev()

        gc.collect()

    @staticmethod
    def _check_input_args(
        all_idxs: Sequence[int],
        train_idxs: Sequence[int],
        input_dir: str,
        output_dir: str,
        overlap_dir: Optional[str],
        calc_out_train_std_dev: bool,
    ):
        """Checks args to the constructor."""
        if not np.all([i in all_idxs for i in train_idxs]):
            raise ValueError(
                "all the idxs passed in `train_idxs` must be in `all_idxs`"
            )
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
        else:
            if calc_out_train_std_dev:
                raise ValueError(
                    "cannot calculate standard deviation of density without"
                    " specification of the `overlap_dir`"
                )

    def __loaditem__(self, idx: int) -> Tuple:
        """
        Loads a data item for the corresponding `idx` from disk and return it in
        a tuple. Returns the idx, and input and output TensorMaps, as well as
        the overlap TensorMap if applicable.
        """
        input = metatensor.io.load_custom_array(
            self.in_paths[idx],
            create_array=metatensor.io.create_torch_array,
        )
        output = metatensor.io.load_custom_array(
            self.out_paths[idx],
            create_array=metatensor.io.create_torch_array,
        )
        input = metatensor.to(input, "torch", **self._torch_kwargs)
        output = metatensor.to(output, "torch", **self._torch_kwargs)

        # Return input/output pair if no overlap matrix required
        if self._overlap_dir is None:
            return idx, input, output

        # Return overlap matrix if applicable
        overlap = metatensor.io.load_custom_array(
            self.overlap_paths[idx],
            create_array=metatensor.io.create_torch_array,
        )
        overlap = metatensor.to(overlap, "torch", **self._torch_kwargs)
        return idx, input, output, overlap

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self._all_idxs)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Loads and returns the input/output data pair for the corresponding
        `idx`. Input, output, and overlap matrices (if applicable) are
        loaded with a torch backend.

        Each item can be accessed with `self[idx]`.
        """
        if self._data_in_mem:
            return self._loaded_data[idx]
        return self.__loaditem__(idx)

    def _load_data(self) -> None:
        """
        Lazily loads the data for all items corresponding to `all_idxs` upon class
        initialization. Stores it in a dict indexed by these indices

        Each item correpsonds to a tuple of the idx, and TensorMaps for the
        input, output, and (if applicable) overlap matrices.
        """
        self._loaded_data = {}
        for idx in self._all_idxs:
            self._loaded_data[idx] = self.__loaditem__(idx)

    @property
    def out_train_inv_means(self) -> TensorMap:
        """
        Returns a TensorMap containing the invariant block means of the output
        training data.
        """
        return self._out_train_inv_means

    def _calculate_out_train_inv_means(self) -> None:
        """
        Retrieves the `data="input"` or `data="output"` TensorMaps for the
        `train_idxs`, joins them, then only for the invariant blocks performs a
        reduction by taking the mean over samples. Stores the resulting
        TensorMap in `self._out_train_inv_means`,
        """
        # Retrieve the output training data and join
        out_train_list = [self[idx][2] for idx in self._train_idxs]
        out_train = metatensor.join(
            out_train_list,
            axis="samples",
            remove_tensor_name=True,
        )
        out_train = metatensor.to(out_train, "numpy")

        # Calc and store invariant means
        out_train_inv_means = get_invariant_means(out_train)
        out_train_inv_means = metatensor.to(
            out_train_inv_means, 
            "torch", 
            dtype=self._torch_kwargs["dtype"], 
            device=self._torch_kwargs["device"], 
            requires_grad=False,
        )
        self._out_train_inv_means = out_train_inv_means

        # Clear memory
        del out_train_list, out_train, out_train_inv_means
        gc.collect()

    @property
    def out_train_std_dev(self) -> float:
        """
        Returns the standard deviation of the output training data, given by the
        expression:

        .. math:

            \sqrt{
                \frac{1}{N - 1} \sum_A^N ... \Delta \bar{c} \hat{S} \Delta
                \bar{c}
            }

        where A is 1 of N training structures, and \Delta \bar{c} =
        c^{\text{RI}} - \bar{c} is the vector of reference RI coefficients minus
        the mean baseline, and \hat{S} is the overlap matrix.
        """
        return self._out_train_std_dev

    def _calculate_out_train_std_dev(self) -> None:
        """
        Calculates the standard deviaton of the output training data.

        See property `RhoLoss.out_train_std_dev` for the mathematical expression.
        """
        # We can use the RhoLoss function to evaluate the error in the
        # output data relative to the mean baseline of the training data
        loss_fn = loss.RhoLoss()

        # As this function is only ever called once for a given set of training
        # indices, iterate over each training structure in turn to reduce memory
        # overhead
        stddev = 0
        for idx in self._train_idxs:
            # Load/retrieve the output training structure and overlap matrix
            _, _, out_train, overlap = self[idx]

            # If the output data hasn't been standardized, do it
            out_train = standardize_invariants(out_train, self._out_train_inv_means)

            # Build a dummy TensorMap of zeros - this will be used for input
            # into the RhoLoss loss function as the 'target'
            out_zeros = metatensor.zeros_like(out_train)

            # Accumulate the 'loss' on the training ouput relative to the
            # spherical baseline
            stddev += loss_fn(input=out_train, target=out_zeros, overlap=overlap)

            # Clear memory
            del out_train, overlap, out_zeros
            gc.collect()

        # Calculate the final stddev and store
        stddev = torch.sqrt(stddev / (len(self._train_idxs) - 1))
        self._out_train_std_dev = stddev


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
        self._idxs = idxs
        self._batch_size = batch_size if batch_size is not None else len(idxs)
        self._get_overlaps = get_overlaps

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._batch_size,
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
                TensorMap # joined input_i, input_j, ...
                TensorMap # joined output_i, output_j, ...
            ]

        or, if overlap matrices are included:

            List[
                List[int, int, ...] # structure_idx_i, structure_idx_j, ...
                TensorMap # joined input_i, input_j, ...
                TensorMap # joined outputs
                TensorMap # joined output_i, output_j, ...
            ]
        """
        # Zip the batch such that data is grouped by the data they represent
        # rather than by idx
        batch = list(zip(*batch))

        # Return the idxs, input, output, and overlaps
        if self._get_overlaps:
            return (
                batch[0],
                metatensor.join(batch[1], axis="samples", remove_tensor_name=True),
                metatensor.join(batch[2], axis="samples", remove_tensor_name=True),
                metatensor.join(batch[3], axis="samples", remove_tensor_name=True),
            )
        # Otherwise, just return the idxs and the input/output pair
        return (
            batch[0],
            metatensor.join(batch[1], axis="samples", remove_tensor_name=True),
            metatensor.join(batch[2], axis="samples", remove_tensor_name=True),
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
    the group sizes returned will be np.array([4, 4, 4]).

    If `group_sizes` is specified as a Sequence of floats (i.e. relative sizes,
    whose sum is <= 1), the group sizes returned are converted to absolute
    sizes, i.e. multiplied by `n_indices`. If `group_sizes` is specified as a
    Sequence of int, no conversion is performed. A cascade round is used to make
    sure that the group sizes are integers, with the sum of the Sequence
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
    return metatensor.mean_over_samples(inv_tensor, sample_names=inv_tensor.sample_names)


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


# def _standardize_output_data(self) -> None:
#         """
#         Standardizes all the output data by subtracting baseline mean of the
#         output training data. As the means of all covariant blocks (lambda > 0)
#         are assumed to be zero, this in practice involves just subtracting the
#         invariant means of the output training data from the invariant blocks of
#         all output data.

#         If `self.keep_in_mem=True`, the attribute appropraite data stored in
#         `self._loaded_data` is overwritten with the standardized data.

#         If `self.keep_in_mem=False`, the standardized data is written to file
#         alongside the unstandardized data. In this case, standardized outputs
#         are stored in `self._output_dir` under filenames
#         "{filenames[1]}_std.npz".
#         """
#         # Standardize and overwrite the unstd data already in memory
#         if self._data_in_mem:
#             for idx in self._all_idxs:
#                 item = self._loaded_data[idx]
#                 tmp_output = standardize_invariants(
#                     item[2],
#                     invariant_means=self._out_train_inv_means,
#                     reverse=False,
#                 )
#                 if len(item) == 4:
#                     self._loaded_data[idx] = (item[0], item[1], tmp_output, item[3])
#                 else:
#                     assert len(item) == 3
#                     self._loaded_data[idx] = (item[0], item[1], tmp_output)
#                 del item, tmp_output
#                 gc.collect()

#         # Calculate standardized data and write to file
#         else:
#             for idx in self._all_idxs:
#                 out_std = standardize_invariants(
#                     self[idx][2],
#                     invariant_means=self._out_train_inv_means,
#                     reverse=False,
#                 )
#                 metatensor.save(
#                     os.path.join(
#                         self._output_dir, f"{idx}/{self._filenames[1]}_std.npz"
#                     )
#                 )
#                 del out_std
#                 gc.collect()
