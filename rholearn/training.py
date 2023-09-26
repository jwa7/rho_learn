"""
Module containing functions for training the model.
"""
import os
from typing import Optional
import numpy as np
import torch

import metatensor
from metatensor import TensorMap, TensorBlock, Labels

from rholearn import loss, models

TO_TORCH_ATTRS = ["_in_metadata", "_out_metadata", "_out_train_inv_means"]


def load_rho_model(path) -> torch.nn.Module:
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
            )
        )
    model.train()
    chkpt["model"] = model

    return chkpt
