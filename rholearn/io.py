import os
import pickle
from typing import List

import numpy as np
import torch

import metatensor
from metatensor import Labels, TensorBlock, TensorMap

from rholearn import utils

# Define the attributes of EquiModelGlobal and EquiModelLocal that are dict
# objects and whose keys need hashing/unhashing when saving/loading the models
MODEL_DICT_ATTRS = [
    "models",
    "in_feature_labels",
    "out_feature_labels",
    "in_invariant_features",
    "hidden_layer_widths",
]
LOSS_DICT_ATTRS = ["output_samples", "output_shapes", "processed_coulomb"]


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


def load_tensormap_to_torch(
    path: str, requires_grad: bool, dtype: torch.dtype, device: torch.device
):
    """
    Loads a TensorMap using metatensor.load, then converts its block values to
    torch tensors with the specified grad, dtype, and device options.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"file at path {path} does not exist")
    return metatensor.to(
        metatensor.load(path),
        backend="torch",
        requires_grad=requires_grad,
        dtype=dtype,
        device=device,
    )


def save_torch_object(torch_obj: torch.nn.Module, path: str, torch_obj_str: str):
    """
    Save a torch object ``torch_obj``, either a ``torch_obj_str``="model" or
    "loss_fn" to file at the specified absolute ``path``.

    For the appropriate dict attributes of the torch object, the keys are
    modified upon saving to change them from np.void type to tuple type. This
    allows them to be saved to file.
    """
    # Retrieve the list of dict attributes whose keys will need to be modified
    # upon saving
    if torch_obj_str not in ["model", "loss_fn"]:
        raise ValueError("can currently only save 'model' or 'loss_fn'")
    attr_names = MODEL_DICT_ATTRS if torch_obj_str == "model" else LOSS_DICT_ATTRS

    # Iterate over the dict attr names, get the attributes, modify the keys and
    # set the new attribute of the torch object
    old_attr_dicts = {}
    for attr in attr_names:
        # Get the attr if it exists
        try:
            old_attr_dict = getattr(torch_obj, attr)
        except AttributeError:
            continue

        # Store the old dict attributes before modifying
        old_attr_dicts[attr] = old_attr_dict

        # Modify the keys of the attr dict
        new_attr_dict = {
            utils.key_npvoid_to_tuple(key): val for key, val in old_attr_dict.items()
        }
        setattr(torch_obj, attr, new_attr_dict)

    # Save the torch object to file
    torch.save(
        torch_obj,
        path,
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )

    # Reset the dict attr to the original ones so that the object can continue
    # being used by reference in other places if needed
    for attr, old_attr_dict in old_attr_dicts.items():
        setattr(torch_obj, attr, old_attr_dict)


def load_torch_object(
    path: str, device: torch.device, torch_obj_str: str
) -> torch.nn.Module:
    """
    Loads a torch object, either a torch ``torch_obj_str``="model" or "loss_fn"
    from file at ``path`` onto the specified ``device`` (i.e. 'cpu', 'cuda').

    For the appropriate dict attributes of the torch object, the keys are
    modified upon loading to change them from tuple types to np.void type. This
    allows them to be accessed in the usual way blocks of a TensorMap are
    accessed, i.e. using individual entries in an metatensor Labels object.
    """
    # Load the torch object from file
    torch_obj = torch.load(path, map_location=device.type)

    # Retrieve the list of dict attributes whose keys will need to be modified
    # upon saving
    if torch_obj_str not in ["model", "loss_fn"]:
        raise ValueError("can currently only load 'model' or 'loss_fn'")
    attr_names = MODEL_DICT_ATTRS if torch_obj_str == "model" else LOSS_DICT_ATTRS

    # Convert the appropriate keys Labels object to be searchable
    # TODO: remove this once checked OK
    # if torch_obj_str == "model":  # only the 'keys' attribute
    #     torch_obj.keys = utils.searchable_labels(torch_obj.keys)
    # else:  # both the 'coulomb_keys' and 'output_keys' attributes
    #     torch_obj.coulomb_keys = utils.searchable_labels(torch_obj.coulomb_keys)
    #     torch_obj.output_keys = utils.searchable_labels(torch_obj.output_keys)

    # Iterate over the dict attributes, get them from the torch object, modify
    # the keys, and reset the attributes
    for attr in attr_names:
        try:  # Get the attribute if it exists
            old_attr_dict = getattr(torch_obj, attr)
        except AttributeError:
            continue

        # Define the names of the keys for the attribute. For a torch model, all
        # dict attributes are indexed by the same keys. For CoulombLoss, the
        # keys for the 'coulomb_loss' attribute take the form (l1, l2, a1, a2)
        # compared to the other attributess that have form (l, a)
        if torch_obj_str == "model":
            names = torch_obj.keys.names
        else:
            names = (
                torch_obj.coulomb_keys.names
                if attr == "processed_coulomb"
                else torch_obj.output_keys.names
            )

        # Modify the keys of the attr dict
        new_attr_dict = {
            utils.key_tuple_to_npvoid(key, names=names): val
            for key, val in old_attr_dict.items()
        }
        setattr(torch_obj, attr, new_attr_dict)

    return torch_obj


def log(log_path: str, line: str):
    """
    Writes the string in `line` to the file at `log_path`, inserting a newline
    character at the end.
    """
    if os.path.exists(log_path):
        with open(log_path, "a") as f:
            f.write(line + "\n")
    else:
        with open(log_path, "w") as f:
            f.write(line + "\n")
