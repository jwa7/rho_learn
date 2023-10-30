"""
Module for reading and writing data to disk.
"""
import os
import pickle

import torch
import metatensor


TO_TORCH_ATTRS = ["_in_metadata", "_out_metadata", "_out_train_inv_means"]


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


def load_rho_model(path: str) -> torch.nn.Module:
    """
    Loads a saved model and ensures all TensorMaps are converted to a torch
    backend.
    """
    model = torch.load(path)
    # Convert each attribute to torch
    if model._out_train_inv_means is None:
        attrs = TO_TORCH_ATTRS[:2]
    else:
        attrs = TO_TORCH_ATTRS
    for attr in attrs:
        setattr(
            model,
            attr,
            metatensor.to(
                getattr(model, attr),
                "torch",
                dtype=model._torch_settings["dtype"],
                device=model._torch_settings["device"],
            ),
        )

    return model


def save_checkpoint(path: str, model, optimizer, scheduler=None):
    """
    Saves a checkpoint at `path` for the model,
    optimizer, and scheduler if applicable.
    """
    chkpt_dict = {
        "model": model,
        "optimizer": optimizer,
    }
    if scheduler is not None:
        chkpt_dict.update({"scheduler": scheduler})
    torch.save(chkpt_dict, path)


def load_from_checkpoint(path: str) -> dict:
    """
    Loads a checkpoint from file. The model is returned in training mode.
    """
    chkpt = torch.load(path)
    model = chkpt["model"]
    # Convert each attribute to torch
    if model._out_train_inv_means is None:
        attrs = TO_TORCH_ATTRS[:2]
    else:
        attrs = TO_TORCH_ATTRS
    for attr in attrs:
        setattr(
            model,
            attr,
            metatensor.to(
                getattr(model, attr),
                "torch",
                dtype=model._torch_settings["dtype"],
                device=model._torch_settings["device"],
            ),
        )
    model.train()
    chkpt["model"] = model

    return chkpt
