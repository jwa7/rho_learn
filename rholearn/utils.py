import ctypes
import datetime
import gc
import os
from typing import List, Union, Optional

import numpy as np
import torch

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


def timestamp() -> str:
    """Return a timestamp string in format YYYYMMDDHHMMSS."""
    return datetime.datetime.today().strftime('%Y%m%d%H%M%S')

# ===== torch to core

def mts_tensormap_torch_to_core(tensor: torch.ScriptObject) -> metatensor.TensorMap:

    return metatensor.TensorMap(
        keys=mts_labels_torch_to_core(tensor.keys),
        blocks=[
            mts_tensorblock_torch_to_core(block)
            for block in tensor
        ]
    )

def mts_tensorblock_torch_to_core(block: torch.ScriptObject) -> metatensor.TensorBlock:
    return metatensor.TensorBlock(
        values=np.array(block.values),
        samples=mts_labels_torch_to_core(block.samples),
        components=[mts_labels_torch_to_core(c) for c in block.components],
        properties=mts_labels_torch_to_core(block.properties),
    )

def mts_labels_torch_to_core(labels: torch.ScriptObject) -> metatensor.Labels:
    return metatensor.Labels(labels.names, values=np.array(labels.values))

# ===== core to torch

def mts_tensormap_core_to_torch(tensor: metatensor.TensorMap) -> torch.ScriptObject:

    return metatensor.torch.TensorMap(
        keys=mts_labels_core_to_torch(tensor.keys),
        blocks=[
            mts_tensorblock_core_to_torch(block)
            for block in tensor
        ]
    )

def mts_tensorblock_core_to_torch(block: metatensor.TensorBlock) -> torch.ScriptObject:
    return metatensor.torch.TensorBlock(
        values=torch.tensor(block.values),
        samples=mts_labels_core_to_torch(block.samples),
        components=[mts_labels_core_to_torch(c) for c in block.components],
        properties=mts_labels_core_to_torch(block.properties),
    )

def mts_labels_core_to_torch(labels: metatensor.Labels) -> torch.ScriptObject:
    return metatensor.torch.Labels(labels.names, values=torch.tensor(labels.values))


def labels_where(labels: Labels, selection: Labels):
    """
    Returns the `labels` object sliced to only contain entries that match the
    `selection`.
    """
    # Extract the relevant columns from `selection` that the selection will
    # be performed on
    keys_out_vals = [[k[name] for name in selection.names] for k in labels]

    # First check that all of the selected keys exist in the output keys
    for slct in selection.values:
        if not np.any([np.all(slct == k) for k in keys_out_vals]):
            raise ValueError(
                f"selected key {selection.names} = {slct} not found"
                " in the output keys. Check the `selection` argument."
            )

    # Build a mask of the selected keys
    mask = [np.any([np.all(i == j) for j in selection.values]) for i in keys_out_vals]

    labels = Labels(names=labels.names, values=labels.values[mask])

    return labels


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


def flatten_tensormap(
    tensor: TensorMap, backend: str
) -> Union[np.ndarray, torch.Tensor]:
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
