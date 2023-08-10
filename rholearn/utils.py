import ctypes
import gc
import os
from typing import List, Union, Optional

import numpy as np
import torch

import equistore
from equistore import Labels, TensorBlock, TensorMap


# ===== tensors to contiguous


def make_contiguous_numpy(tensor: TensorMap) -> TensorMap:
    """
    Takes a TensorMap of numpy backend and makes the ndarray block and
    gradient values contiguous in memory.
    """
    new_blocks = []
    for key, block in tensor.items():
        new_block = TensorBlock(
            values=np.ascontiguousarray(block.values),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        for parameter, gradient in block.gradients():
            new_gradient = TensorBlock(
                values=np.ascontiguousarray(gradient.values),
                samples=gradient.samples,
                components=gradient.components,
                properties=gradient.properties,
            )
            new_block.add_gradient(parameter, new_gradient)
        new_blocks.append(new_block)

    return TensorMap(
        keys=tensor.keys,
        blocks=new_blocks,
    )


# ===== fxns converting types of Labels object entries:
# i.e. np.void <--> tuple <--> str


def key_to_str(key: Union[np.void, tuple]) -> str:
    """
    Takes a single numpy void key (i.e. an element of a Labels object) or a
    tuple and converts it to a string. Return string has the form
    f"{key[0]}_{key[1]}_...", where element values are separated by undescores.
    """
    return "_".join([str(k) for k in key])


def key_npvoid_to_tuple(key: np.void) -> tuple:
    """
    Takes a single numpy void key (i.e. an element of a Labels object) and
    converts it to a tuple.
    """
    return tuple(k for k in key)


def key_tuple_to_npvoid(key: tuple, names: List[str]) -> np.void:
    """
    Takes a key as a tuple and the associated names of the values in that tuple
    and returns a numpy void object that can be used to access blocks in a
    TensorMap, as well as values in a dict that are indexed by these numpy void
    keys.
    """
    # We need to create a TensorMap object here, as this allows a hashable object that
    # can be used to access dict values to be returned.
    tensor = TensorMap(
        keys=Labels(
            names=names,
            values=np.array(
                [key],
                dtype=np.int32,
            ),
        ),
        blocks=[
            TensorBlock(
                values=np.full((1, 1), 0.0),
                samples=Labels(
                    ["s"],
                    np.array(
                        [
                            [0],
                        ],
                        dtype=np.int32,
                    ),
                ),
                components=[],
                properties=Labels(["p"], np.array([[0]], dtype=np.int32)),
            )
        ],
    )
    return tensor.keys[0]


# ===== TensorMap + TensorBlock functions


def pad_with_empty_blocks(
    input: TensorMap, target: TensorMap, slice_axis: str = "samples"
) -> TensorMap:
    """
    Takes an ``input`` TensorMap with fewer blocks than the ``target``
    TensorMap. For every key present in ``target`` but not ``input``, an empty
    block is created by slicing the ``target`` block to zero dimension along the
    ``slice_axis``, which is either "samples" or "properties". For every key
    present in both ``target`` and ``input``, the block is copied exactly from
    ``input``. A new TensorMap, with the same number of blocks at ``target``,
    but all the original data from ``input``, is returned.
    """
    blocks = []
    for key in target.keys:
        if key in input.keys:
            # Keep the block
            blocks.append(input[key].copy())
        else:
            samples = target[key].samples
            properties = target[key].properties
            # Create an empty sliced block
            if slice_axis == "samples":
                values = target[key].values[:0]
                samples = samples[:0]
            else:  # properties
                target[key].values[..., :0]
                properties = properties[:0]
            blocks.append(
                TensorBlock(
                    samples=samples,
                    components=target[key].components,
                    properties=properties,
                    values=values,
                )
            )
    return TensorMap(keys=target.keys, blocks=blocks)


def sort_tensormap(tensor: TensorMap) -> TensorMap:
    """
    Uses np.sort to rearrange block values along every axis. Note: doesn't
    currently handle gradient blocks. Assumes there is only one components axis.
    """
    keys = tensor.keys
    blocks = []
    for key in keys:
        block = tensor[key]

        # Define the samples resorting order
        unsorted_s = block.samples
        sorted_s = np.sort(unsorted_s)
        samples_filter = np.array([np.where(unsorted_s == s)[0][0] for s in sorted_s])

        # Define the components resorting order
        unsorted_c = block.components
        sorted_c = [np.sort(un_c) for un_c in unsorted_c]
        components_filter = [
            np.array([np.where(un_c == c)[0][0] for c in sor_c])
            for un_c, sor_c in zip(unsorted_c, sorted_c)
        ]

        # Define the properties resorting order
        unsorted_p = block.properties
        sorted_p = np.sort(unsorted_p)
        properties_filter = np.array(
            [np.where(unsorted_p == p)[0][0] for p in sorted_p]
        )

        # Sort the block
        blocks.append(
            TensorBlock(
                samples=sorted_s,
                components=sorted_c,
                properties=sorted_p,
                values=np.ascontiguousarray(
                    block.values[samples_filter][:, components_filter[0]][
                        ..., properties_filter
                    ]
                ),
            )
        )
    return TensorMap(keys=keys, blocks=blocks)


def feature_labels_tensormap(tensor: TensorMap) -> TensorMap:
    """
    Returns a TensorMap with minimal data, and dummy metadata, except the keys
    and properties correspond to the input tensor.
    """
    blocks = []
    for key in tensor.keys:
        props = tensor[key].properties
        blocks.append(
            TensorBlock(
                samples=Labels(names=["_"], values=np.array([0]).reshape(-1, 1)),
                components=[],
                properties=props,
                values=np.zeros((1, len(props))),
            )
        )
    return TensorMap(keys=tensor.keys, blocks=blocks)


# ===== other utility functions


def flatten_dict(d: dict) -> dict:
    """
    Takes a nested dict and flattens it into a single level dict.

    Adapted from https://stackoverflow.com/a/66789625
    """
    result = {}
    if isinstance(d, dict):
        for k in d:
            # Recursively flatten the nested dict
            flattened_dict = flatten_dict(d[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                result[tuple(key)] = val
    else:
        result[()] = d
    return result


def flatten_tensormap(tensor, backend: str) -> float:
    """
    Flattens all block values and returns as a 1D numpy array.
    """
    if backend == "numpy":
        flattened = np.array([])
        for block in tensor.blocks():
            flattened = np.concatenate((flattened, block.values.flatten()))
    elif backend == "torch":
        flattened = torch.tensor([])
        for block in tensor.blocks():
            flattened = torch.cat((flattened, block.values.flatten()))

    else:
        raise ValueError(f"Unknown backend {backend}, must be 'numpy' or 'torch'")

    return flattened


def num_elements_tensormap(tensor: TensorMap) -> int:
    """
    Returns the total number of elements in the input tensor.

    If the input tensor is a TensorMap the number of elements is given by the
    sum of the product of the dimensions for each block.

    If the input tensor is a TensorBlock or a torch.Tensor, the number of
    elements is just given by the product of the dimensions.
    """
    n_elems = 0
    if isinstance(tensor.block(0).values, np.ndarray):
        for block in tensor.blocks():
            n_elems += np.prod(block.values.shape)
    elif isinstance(tensor.block(0).values, torch.Tensor):
        for block in tensor.blocks():
            n_elems += torch.prod(torch.tensor(block.values.shape))

    return int(n_elems)


# ===== memory utils


def trim_memory() -> int:
    # Garbage collect
    gc.collect()
    # Release memory back to the OS
    try:
        libc = ctypes.CDLL("libc.so.6")
        return libc.malloc_trim(0)
    except OSError:
        # libc = ctypes.CDLL("libc++.dylib")  # for MacOS ?
        return
