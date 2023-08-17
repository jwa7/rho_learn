"""
Module containing functions for training the model.
"""
import os
from typing import Optional
import numpy as np
import torch

from equistore import TensorMap, TensorBlock, Labels

from rholearn import loss, models


def load_rho_model(path) -> torch.nn.Module:
    """
    Loads a saved model and ensures all TensorMaps are converted to a torch
    backend.
    """
    model = torch.load(path)
    # Convert each attribute to torch
    if model._outputs_standardized:
        attrs = ["_in_metadata", "_out_metadata", "_out_invariant_means"]
    else:
        attrs = ["_in_metadata", "_out_metadata"]
    for attr in attrs:
        setattr(
            model,
            attr,
            equistore.to(getattr(model, attr), "torch", **model._torch_settings),
        )

    return model


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Module,
    scheduler: Optional[torch.optim.Module] = None,
):
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
    if model._outputs_standardized:
        attrs = ["_in_metadata", "_out_metadata", "_out_invariant_means"]
    else:
        attrs = ["_in_metadata", "_out_metadata"]
    for attr in attrs:
        setattr(
            model,
            attr,
            equistore.to(getattr(model, attr), "torch", **model._torch_settings),
        )
    model.train()
    chkpt["model"] = model

    return chkpt
    