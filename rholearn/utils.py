import ctypes
import datetime
import gc
import os
import pickle
from typing import List, Union, Optional

import numpy as np
import torch

import metatensor as mts
from metatensor import Labels, TensorBlock, TensorMap


def timestamp() -> str:
    """Return a timestamp string in format YYYY-MM-DD-HH:MM:SS."""
    return datetime.datetime.today().strftime("%Y-%m-%d-%H:%M:%S")


# ===== torch to core


def mts_tensormap_torch_to_core(tensor: torch.ScriptObject) -> TensorMap:

    return TensorMap(
        keys=mts_labels_torch_to_core(tensor.keys),
        blocks=[mts_tensorblock_torch_to_core(block) for block in tensor],
    )


def mts_tensorblock_torch_to_core(block: torch.ScriptObject) -> TensorBlock:
    return TensorBlock(
        values=np.array(block.values),
        samples=mts_labels_torch_to_core(block.samples),
        components=[mts_labels_torch_to_core(c) for c in block.components],
        properties=mts_labels_torch_to_core(block.properties),
    )


def mts_labels_torch_to_core(labels: torch.ScriptObject) -> Labels:
    return Labels(labels.names, values=np.array(labels.values))


# ===== core to torch


def mts_tensormap_core_to_torch(tensor: TensorMap) -> torch.ScriptObject:

    return mts.torch.TensorMap(
        keys=mts_labels_core_to_torch(tensor.keys),
        blocks=[mts_tensorblock_core_to_torch(block) for block in tensor],
    )


def mts_tensorblock_core_to_torch(block: TensorBlock) -> torch.ScriptObject:
    return mts.torch.TensorBlock(
        values=torch.tensor(block.values),
        samples=mts_labels_core_to_torch(block.samples),
        components=[mts_labels_core_to_torch(c) for c in block.components],
        properties=mts_labels_core_to_torch(block.properties),
    )


def mts_labels_core_to_torch(labels: Labels) -> torch.ScriptObject:
    return mts.torch.Labels(labels.names, values=torch.tensor(labels.values))


# Rename TensorMaps


def rename_coeff_tensor(tensor: TensorMap) -> TensorMap:
    """
    Renames a TensorMap corresponding to a coefficient vector according to the new
    naming convention.
    """
    # Keys
    tensor = mts.rename_dimension(tensor, "keys", "spherical_harmonics_l", "o3_lambda")
    tensor = mts.rename_dimension(tensor, "keys", "species_center", "center_type")
    # Samples
    tensor = mts.rename_dimension(tensor, "samples", "structure", "system")
    tensor = mts.rename_dimension(tensor, "samples", "center", "atom")
    # Components
    tensor = TensorMap(
        tensor.keys,
        blocks=[
            TensorBlock(
                values=block.values,
                samples=block.samples,
                components=[
                    Labels(names=["o3_mu"], values=block.components[0].values),
                ],
                properties=block.properties,
            )
            for block in tensor
        ],
    )

    return tensor


def rename_ovlp_matrix(tensor: TensorMap) -> TensorMap:
    """
    Renames a TensorMap corresponding to an overlap matrix according to the new naming
    convention.
    """
    # Keys
    tensor = mts.rename_dimension(
        tensor, "keys", "spherical_harmonics_l1", "o3_lambda_1"
    )
    tensor = mts.rename_dimension(
        tensor, "keys", "spherical_harmonics_l2", "o3_lambda_2"
    )
    tensor = mts.rename_dimension(tensor, "keys", "species_center_1", "center_1_type")
    tensor = mts.rename_dimension(tensor, "keys", "species_center_2", "center_2_type")
    # Samples
    tensor = mts.rename_dimension(tensor, "samples", "structure", "system")
    tensor = mts.rename_dimension(tensor, "samples", "center_1", "atom_1")
    tensor = mts.rename_dimension(tensor, "samples", "center_2", "atom_2")
    # Components
    tensor = TensorMap(
        tensor.keys,
        blocks=[
            TensorBlock(
                values=block.values,
                samples=block.samples,
                components=[
                    Labels(names=["o3_mu_1"], values=block.components[0].values),
                    Labels(names=["o3_mu_2"], values=block.components[1].values),
                ],
                properties=block.properties,
            )
            for block in tensor
        ],
    )
    # Properties
    tensor = mts.rename_dimension(tensor, "properties", "n1", "n_1")
    tensor = mts.rename_dimension(tensor, "properties", "n2", "n_2")

    return tensor


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
    Sorts the TensorMap and then flattens all block values and returns as a 1D numpy
    array.
    """
    try:
        tensor = mts.sort(tensor)
    except ValueError:
        tensor = mts.torch.sort(tensor)
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


# ===== File IO utilities =====


def check_or_create_dir(dir_path: str):
    """
    Takes as input an absolute directory path. Checks whether or not it exists.
    If not, creates it.
    """
    if not os.path.exists(dir_path):
        try:
            os.mkdir(dir_path)
        except FileNotFoundError:
            raise ValueError(
                f"Specified directory {dir_path} is not valid."
                + " Check that the parent directory of the one you are trying to create exists."
            )


def pickle_dict(path: str, dict: dict):
    """
    Pickles a dict at the specified absolute path. Add a .pickle suffix if
    not given in the path.
    """
    if not path.endswith(".pickle"):
        path += ".pickle"
    with open(path, "wb") as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_dict(path: str):
    """
    Unpickles a dict object from the specified absolute path and returns
    it.
    """
    with open(path, "rb") as handle:
        d = pickle.load(handle)
    return d


def log(log_path: str, line: str, comment: bool = True, use_timestamp: bool = True):
    """
    Writes the string in `line` to the file at `log_path`, inserting a newline
    character at the end. By default a '#' is prefixed to every line, followed
    optionally by a timestamp.
    """
    log_line = ""
    if comment:
        log_line = "#"
    if use_timestamp:
        log_line += " " + timestamp() + " "
    log_line += line
    if os.path.exists(log_path):
        with open(log_path, "a") as f:
            f.write(log_line + "\n")
    else:
        with open(log_path, "w") as f:
            f.write(log_line + "\n")
