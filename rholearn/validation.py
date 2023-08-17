"""
Module containing functions for evaluating the model.
"""
import os
from typing import Optional
import numpy as np
import torch

from equistore import TensorMap, TensorBlock, Labels

from rholearn import loss, models

def load_model_for_eval(path: str) -> dict:
    """
    Loads a checkpoint from file. The model is returned in eval mode.
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
    model.eval()

    return model