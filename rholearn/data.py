"""
Module for creating torch datasets and dataloaders.
"""
import os
from typing import List

import torch

import equistore
from equistore import TensorMap


class OverlapData(torch.utils.data.Dataset):
    """
    Custom torch Dataset class for loading overlap-type matrices from disk.

    All overlap matrices must be stored in the specified `overlap_dir` as
    TensorMaps with filenames of the form `{structure_idx}.npz`, where
    `structure_idx` is the index of the structure the overlap-type matrix
    corresponds to.

    :param overlap_dir: str, absolute path to directory containing overlap-type
        matrices.
    :param structure_idxs: List[int], list of structure indices that define the
        complete dataset.
    """

    def __init__(self, overlap_dir: str, structure_idxs: List[int]):
        """
        Constructs the OverlapData object.
        """
        super(OverlapData, self).__init__()

        # Check input args
        _check_input_args(overlap_dir, structure_idxs)

        # Assign attributes
        self.overlap_dir = overlap_dir
        self.structure_idxs = structure_idxs
        self.paths = [
            os.path.join(self.overlap_dir, f"{structure_idx}.npz")
            for structure_idx in self.structure_idxs
        ]
        # Check files exist
        for path in self.paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Overlap matrix {path} does not exist.")

    @staticmethod
    def _check_input_args(overlap_dir: str):
        """Checks args to the constructor."""
        if not os.path.exists(overlap_dir):
            raise NotADirectoryError(f"Overlap directory {overlap_dir} does not exist.")

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.structure_idxs)

    def __getitem__(self, structure_idx: int) -> TensorMap:
        """
        Loads and returns the overlap matrix for the corresponding
        `structure_idx`.

        TODO: load directly to torch. Requires changes to
        equistore-torch/src/register.cpp.
        """
        overlap = equistore.load(self.paths[structure_idx])
        return overlap


class DensityData(torch.utils.data.Dataset):
    """
    Custom torch Dataset class for loading input/descriptors (i.e. lambda-SOAP)
    and output/features (i.e. electron density coefficients) data from disk.

    :param input_dir: str, absolute path to directory containing
        input/descriptor (i.e. lambda-SOAP) data.
    :param output_dir: str, absolute path to directory containing
        output/features (i.e. electron density coefficients) data.
    :param structure_idxs: List[int], list of structure indices that define the
        complete dataset.
    """

    def __init__(self, input_dir: str, output_dir: str, structure_idxs: List[int]):
        """
        Constructs the DensityData object.
        """
        super(DensityData, self).__init__()

        # Check input args
        _check_input_args(input_dir, output_dir, structure_idxs)

        # Assign attributes
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.structure_idxs = structure_idxs
        self.in_paths = [
            os.path.join(self.input_dir, f"{structure_idx}.npz")
            for structure_idx in self.structure_idxs
        ]
        self.out_paths = [
            os.path.join(self.output_dir, f"{structure_idx}.npz")
            for structure_idx in self.structure_idxs
        ]
        # Check files exist
        for path in self.in_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Descriptor coeffs at {path} does not exist.")
        for path in self.out_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Features / electron density coeffs at {path} does not exist."
                )

    @staticmethod
    def _check_input_args(input_dir: str, output_dir: str):
        """Checks args to the constructor."""
        if not os.path.exists(input_dir):
            raise NotADirectoryError(
                f"Input/descriptor data directory {input_dir} does not exist."
            )
        if not os.path.exists(output_dir):
            raise NotADirectoryError(
                f"Input/descriptor data directory {output_dir} does not exist."
            )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.structure_idxs)

    def __getitem__(self, structure_idx: int) -> TensorMap:
        """
        Loads and returns the input/output data pair for the corresponding
        `structure_idx`.

        TODO: load directly to torch. Requires changes to
        equistore-torch/src/register.cpp.
        """
        input = equistore.load(self.in_paths[structure_idx])
        output = equistore.load(self.out_paths[structure_idx])
        return input, output
