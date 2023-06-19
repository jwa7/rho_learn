"""
Module for creating torch datasets and dataloaders.
"""
import os
from typing import List, Optional, Tuple

import torch

import equistore
from equistore import TensorMap


class DensityData(torch.utils.data.Dataset):
    """
    Custom torch Dataset class for loading input/descriptors (i.e. lambda-SOAP)
    and output/features (i.e. electron density coefficients) data from disk. If
    evaluating the loss of non-orthogonal basis functions using overlap-type
    matrices, the directory containing these matrices must also be specified.

    For every structure indexed in `structure_idxs`, there must be files named
    f"{structure_idx}".npz in the directories `input_dir`, `output_dir`, and
    `overlap_dir`, if applicable.

    :param structure_idxs: List[int], list of structure indices that define the
        complete dataset.
    :param input_dir: str, absolute path to directory containing
        input/descriptor (i.e. lambda-SOAP) data.
    :param output_dir: str, absolute path to directory containing
        output/features (i.e. electron density coefficients) data.
    :param overlap_dir: str, absolute path to directory containing overlap-type
        matrices. Optional, only required if evaluating the loss of
        non-orthogonal basis functions.
    **torch_kwargs: dict of kwargs for loading TensorMaps to torch backend.
        `dtype`, `device`, and `requires_grad` are required.
    """

    def __init__(
        self,
        structure_idxs: List[int],
        input_dir: str,
        output_dir: str,
        overlap_dir: Optional[str] = None,
        **torch_kwargs,
    ):
        super(DensityData, self).__init__()

        # Check input args
        DensityData._check_input_args(input_dir, output_dir, overlap_dir)

        # Assign attributes
        self.structure_idxs = structure_idxs
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.overlap_dir = overlap_dir
        self.dtype = torch_kwargs.get("dtype")
        self.device = torch_kwargs.get("device")
        self.requires_grad = torch_kwargs.get("requires_grad")

        # Set the data directories and check files exist
        self.in_paths = [
            os.path.join(self.input_dir, f"{structure_idx}.npz")
            for structure_idx in self.structure_idxs
        ]
        for path in self.in_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Descriptor coeffs at {path} does not exist.")
        self.out_paths = [
            os.path.join(self.output_dir, f"{structure_idx}.npz")
            for structure_idx in self.structure_idxs
        ]
        for path in self.out_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Features / electron density coeffs at {path} does not exist."
                )
        if overlap_dir is not None:
            self.overlap_paths = [
                os.path.join(self.overlap_dir, f"{structure_idx}.npz")
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
        `structure_idx`.

        TODO: load directly to torch. Requires changes to
        equistore-torch/src/register.cpp.
        """
        input = equistore.to(
            equistore.load(self.in_paths[structure_idx]),
            backend="torch",
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        output = equistore.to(
            equistore.load(self.out_paths[structure_idx]),
            backend="torch",
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )

        # Return input/output pair if no overlap matrix required
        if self.overlap_dir is None:
            return input, output

        # Return overlap matrix if applicable
        overlap = equistore.to(
            equistore.load(self.overlap_paths[structure_idx]),
            backend="torch",
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        return input, output, overlap


def collate_density_data_batch(batch: List[List[TensorMap]]) -> List[List[TensorMap]]:
    """
    Takes a list of list of TensorMaps corresponding to the input/descriptor
    (i.e. lambda-SOAP), output/features (i.e. electron density coefficients),
    and if applicable overlap matrices. Zips them together such that the
    returned list of lists groups by data type (i.e. input, output, overlap,
    respectively) instead of by structure.

    `batch` is a list of list of TensorMaps, with each element corresponding to
    a different structure in the batch and whose length is the batch size. Each
    element in `batch` is a list of either length 2 or 3, depending on whether
    the overlap matrices are included.

    If `batch` doesn't include overlap matrices, it takes the form [[input_0,
    output_0], [input_1, output_1], ...]. Returned is a list of 2 lists of
    TensorMaps, of the form [[input_0, input_1, ...], [output_0, output_1,
    ...]].

    If `batch` does include overlap matrices, it takes the form [[input_0,
    output_0, overlap_0], [input_1, output_1, overlap_1], ...]. Returned is a
    list of 3 lists of TensorMaps, of the form [[input_0, input_1, ...],
    [output_0, output_1, ...], [overlap_0, overlap_1, ...]].

    :param batch: list of list of TensorMaps corresponding to the
        input/descriptor (i.e. lambda-SOAP), output/features (i.e. electron
        density coefficients), and if applicable, overlap matrices.

    :return: list of list of TensorMaps corresponding to the input/descriptor
        (i.e. lambda-SOAP), output/features (i.e. electron density
        coefficients), and if applicable, overlap matrices.
    """
    return list(zip(*batch))


# NOTE: this function is no longer used, but kept for reference.
# def collate_density_data_batch(batch: List[List[TensorMap]]) -> List[TensorMap]:
#     """
#     Takes a list of list of TensorMaps corresponding to the input/descriptor
#     (i.e. lambda-SOAP), output/features (i.e. electron density coefficients),
#     and if applicable overlap matrices.

#     Uses the :py:func:`equistore.join` function to collate seperate TensorMaps
#     corresponding to different structures into a single TensorMap, for each of
#     the data; input/descriptor (i.e. lambda-SOAP), output/features (i.e.
#     electron density coefficients), and if applicable overlap matrices.

#     `batch` is a list of list of TensorMaps, with each element corresponding to
#     a different structure in the batch and whose length is the batch size. Each
#     element in `batch` is a list of either length 2 or 3.

#     In the former case, each element a list containing 2 TensorMaps,
#     corresponding to the input/descriptor (i.e. lambda-SOAP) and output/features
#     (i.e. electron density coefficients), respectively. `batch` thus takes the
#     form [[input_0, output_0], [input_1, output_1], ...]. Returned is a list of
#     2 TensorMaps, corresponding to the collated input and output data,
#     respectively.

#     In the latter case, each element is a list containing 3 TensorMaps,
#     corresponding to the input, output, and overlap matrices, respectively.
#     `batch` thus takes the form [[input_0, output_0, overlap_0], [input_1,
#     output_1, overlap_1], ...]. Returned is a list of 3 TensorMaps,
#     corresponding to the collated input, output, and overlap data, respectively.

#     :param batch: list of list of TensorMaps corresponding to the
#         input/descriptor (i.e. lambda-SOAP), output/features (i.e. electron
#         density coefficients), and if applicable, overlap matrices.

#     :return: tuple of TensorMaps corresponding to the collated input,
#         output/features, and if applicable overlap matrices.
#     """
#     if len(batch[0]) not in [2, 3]:
#         raise ValueError(
#             f"Expected batch to contain either 2 or 3 forms "
#             "of data for each structure, got {len(batch[0])}."
#         )

#     return [equistore.join(tensor, "samples") for tensor in zip(*batch)]
