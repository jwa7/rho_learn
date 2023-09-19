"""
Module containing functions for evaluating the model.
"""
import torch
from rholearn import training


def load_model_for_eval(path: str) -> dict:
    """
    Loads a checkpoint from file. The model is returned in eval mode.
    """
    chkpt = training.load_from_checkpoint(path)
    model = chkpt["model"]
    model.eval()

    return model