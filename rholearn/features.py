"""
Generates features vectors for equivariant structural representations.
Currently implemented:
    - lambda-SOAP
"""
import os
import pickle
from typing import Sequence, Optional

import ase
import numpy as np

import equistore
from equistore import Labels, TensorMap

import rascaline
from rascaline.utils import clebsch_gordan

from rholearn import io, spherical, utils


def lambda_feature_kernel(lsoap_vector: TensorMap) -> TensorMap:
    """
    Takes a lambda-feature vector (i.e. lambda-SOAP or lambda-LODE) as a
    TensorMap and takes the relevant inner products to form a lambda-feature
    kernel, returned as a TensorMap.
    """
    raise NotImplementedError


def get_invariant_means(tensor: TensorMap) -> TensorMap:
    """
    Calculates the mean of the invariant (l=0) blocks on the input `tensor`
    using the `equistore.mean_over_samples` function. Returns the result in a
    new TensorMap, whose number of blocks is equal to the number of invariant
    blocks in `tensor`. Assumes `tensor` is a numpy-based TensorMap.
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
    the coefficients of each feature. Assumes `tensor` and `invariant_means` are
    numpy-based TensorMaps.
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
