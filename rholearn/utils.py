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
    Takes a TensorMap whose block values are ndarrays and ensures they are
    contiguous. Allows tensors produced by slicing/splitting to be saved to file
    using the equistore.save method.
    """

    new_blocks = []
    for key, block in tensor:
        new_block = TensorBlock(
            samples=block.samples,
            components=block.components,
            properties=block.properties,
            values=np.ascontiguousarray(block.values),
        )
        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                samples=gradient.samples,
                components=gradient.components,
                data=np.ascontiguousarray(gradient.data),
            )
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


# ===== fxns for equistore Labels objects comparisons


def labels_intersection(a: Labels, b: Labels):
    """
    For 2 :py:class:`Labels` objects ``a`` and ``b``, returns a new Labels
    object of the indices that they share, i.e. the intersection.
    """
    # Find the intersection
    intersection_idxs = [i for i in [a.position(b_i) for b_i in b] if i is not None]
    return a[intersection_idxs]


# ===== TensorMap + TensorBlock functions


def rename_tensor(
    tensor: TensorMap,
    keys_names: Optional[List[str]] = None,
    samples_names: Optional[List[str]] = None,
    components_names: Optional[List[str]] = None,
    properties_names: Optional[List[str]] = None,
) -> TensorMap:
    """
    Constructs and returns a new TensorMap where the names of the Labels
    metadata for the keys, samples, components, and/or properties have been
    modified according to the new names. Note: does not yet handle gradients.
    """
    new_keys = tensor.keys
    # Modify key names
    if keys_names is not None:
        if len(keys_names) != len(tensor.keys.names):
            raise ValueError(
                "must pass the same number of new keys names as there are old ones"
            )
        new_keys = Labels(
            names=keys_names,
            values=np.array([[i for i in k] for k in tensor.keys]),
        )
    if samples_names is None and components_names is None and properties_names is None:
        new_blocks = [tensor[key].copy() for key in tensor.keys]
    else:
        new_blocks = [
            rename_block(
                tensor[key],
                samples_names=samples_names,
                components_names=components_names,
                properties_names=properties_names,
            )
            for key in tensor.keys
        ]
    return TensorMap(keys=new_keys, blocks=new_blocks)


def rename_block(
    block: TensorBlock,
    samples_names: Optional[List[str]] = None,
    components_names: Optional[List[str]] = None,
    properties_names: Optional[List[str]] = None,
) -> TensorBlock:
    """
    Constructs and returns a new TensorBlock where the names of the Labels
    metadata for samples, components, and/or properties have been modified
    according to the new names. Note: does not yet handle gradients.
    """
    new_samples = block.samples
    new_components = block.components
    new_properties = block.properties
    # Modify samples names
    if samples_names is not None:
        if len(samples_names) != len(block.samples.names):
            raise ValueError(
                "must pass the same number of new samples names as there are old ones"
            )
        samp_values = np.array(
            [i for s in block.samples for i in s], dtype=np.int32
        ).reshape(-1, len(samples_names))
        new_samples = Labels(
            names=samples_names,
            values=samp_values,
        )
    # Modify components names
    if components_names is not None:
        if len(block.components) != len(components_names):
            raise ValueError("must pass same number of new components as old")
        new_components = []
        for c_i in range(len(block.components)):
            if len(components_names[c_i]) != len(block.components[c_i].names):
                raise ValueError(
                    "must pass the same number of new components names as there are old ones"
                )
            comp_values = np.array(
                [j for i in block.components[c_i] for j in i], dtype=np.int32
            ).reshape(-1, len(components_names[c_i]))
            new_components.append(
                Labels(
                    names=components_names[c_i],
                    values=comp_values,
                )
            )
    # Modify properties names
    if properties_names is not None:
        if len(properties_names) != len(block.properties.names):
            raise ValueError(
                "must pass the same number of new properties names as there are old ones"
            )
        prop_values = np.array(
            [i for p in block.properties for i in p], dtype=np.int32
        ).reshape(-1, len(properties_names))
        new_properties = Labels(
            names=properties_names,
            values=prop_values,
        )

    return TensorBlock(
        samples=new_samples,
        components=new_components,
        properties=new_properties,
        values=block.values,
    )


def drop_key_name(tensor: TensorMap, key_name: str) -> TensorMap:
    """
    Takes a TensorMap and drops the key_name from the keys. Every key must have
    the same value for the key_name, otherwise a ValueError is raised.
    """
    keys = tensor.keys
    # Check that the key_name is present and unique
    if not len(np.unique(keys[key_name])) == 1:
        raise ValueError(
            f"key_name {key_name} is not unique in the keys."
            " Can only drop a key_name where the value is the"
            " same for all keys."
        )

    # Define the idx of the key_name to drop
    drop_idx = keys.names.index(key_name)

    # Build the new TensorMap
    new_keys = Labels(
        names=keys.names[:drop_idx] + keys.names[drop_idx + 1 :],
        values=np.array(
            [k.tolist()[:drop_idx] + k.tolist()[drop_idx + 1 :] for k in keys]
        ),
    )
    new_tensor = TensorMap(
        keys=new_keys,
        blocks=[
            TensorBlock(
                samples=tensor[key].samples,
                components=tensor[key].components,
                properties=tensor[key].properties,
                values=tensor[key].values,
            )
            for key in keys
        ],
    )
    del tensor
    trim_memory()
    return new_tensor


def drop_metadata_name(
    tensor: TensorMap, axis: str, name: str, unsafe: bool = False
) -> TensorMap:
    """
    Takes a TensorMap and drops the `name` from either the "samples" or
    "properties" labels of every block. If `unsafe=False` (default), every block
    must have the same value for the `name`, otherwise a ValueError is raised.
    if `unsafe=True`, the `name` is dropped from every block, even if the values
    for this name are not equivalent. This is unsafe as it can create non-unique
    metadata, and should only be used if this is not the case.
    """
    if axis not in ["samples", "properties"]:
        raise ValueError(f"axis must be 'samples' or 'properties', not {axis}")
    # Check that the name is present and unique
    for block in tensor.blocks():
        if axis == "samples":
            uniq = np.unique(block.samples[name])
        else:
            uniq = np.unique(block.properties[name])
        if len(uniq) > 1:
            if unsafe is False:
                raise ValueError(
                    f"name {name} is not unique in the {axis}."
                    " Can only drop a `name` where the value is the"
                    f" same for all {axis}."
                )
            else:
                pass
    # Identify the idx of the name to drop
    if axis == "samples":
        drop_idx = tensor.blocks()[0].samples.names.index(name)
    elif axis == "properties":
        drop_idx = tensor.blocks()[0].properties.names.index(name)
    # Construct new blocks with the dropped name
    new_blocks = []
    for key in tensor.keys:
        new_samples = tensor[key].samples
        new_properties = tensor[key].properties
        if axis == "samples":
            new_samples = Labels(
                names=tensor[key].samples.names[:drop_idx]
                + tensor[key].samples.names[drop_idx + 1 :],
                values=np.array(
                    [
                        s.tolist()[:drop_idx] + s.tolist()[drop_idx + 1 :]
                        for s in tensor[key].samples
                    ]
                ),
            )
        else:
            new_properties = Labels(
                names=tensor[key].properties.names[:drop_idx]
                + tensor[key].properties.names[drop_idx + 1 :],
                values=np.array(
                    [
                        p.tolist()[:drop_idx] + p.tolist()[drop_idx + 1 :]
                        for p in tensor[key].properties
                    ]
                ),
            )
        new_blocks.append(
            TensorBlock(
                samples=new_samples,
                components=tensor[key].components,
                properties=new_properties,
                values=tensor[key].values,
            )
        )
    new_tensor = TensorMap(
        keys=tensor.keys,
        blocks=new_blocks,
    )
    del tensor
    trim_memory()
    return new_tensor


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
