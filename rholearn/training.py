"""
Module containing functions for training the model.
"""
import os
from typing import Optional
import numpy as np
import torch

from equistore import TensorMap, TensorBlock, Labels

from rholearn import loss, models


def init_training_objects(
    ml_settings: dict,
    input: TensorMap,
    output: TensorMap,
    out_invariant_means: Optional[TensorMap] = None,
):
    """
    Initializes and returns the model, optimizer, and loss function for
    training.
    """

    # Define the keys as the intersection of the input and output keys
    keys = input.keys.intersection(output.keys)

    # Initialize model
    model = models.RhoModel(
        model_type=ml_settings["model"]["type"],
        keys=keys,
        in_features=[input[key].properties for key in keys],
        out_features=[output[key].properties for key in keys],
        components=[input[key].components for key in keys],
        out_invariant_means=out_invariant_means,
        **ml_settings["model"]["args"],
    )

    # Initialize optimizer and scheduler (if applicable)
    optimizer = ml_settings["optimizer"]["algorithm"](
        params=model.parameters(),
        **ml_settings["optimizer"]["args"],
    )
    scheduler = None
    if ml_settings.get("scheduler") is not None:
        scheduler = ml_settings["scheduler"]["algorithm"](
            optimizer=optimizer,
            **ml_settings["scheduler"]["args"],
        )

    # Initialize loss functions
    rho_loss_fn = loss.RhoLoss()
    coeff_loss_fn = loss.CoeffLoss()

    return model, optimizer, rho_loss_fn, coeff_loss_fn, scheduler


def save_checkpoint(
    save_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.nn.Module,
    scheduler: Optional[torch.nn.Module] = None,
):
    """
    Saves the state dicts for the model, optimizer, and scheduler (if
    applicable) to a checkpoint file, at path f"{save_dir}/checkpoint_{epoch}.pt".
    """
    if scheduler is None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(save_dir, f"checkpoint_{epoch}.pt"),
        )
    else:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(save_dir, f"checkpoint_{epoch}.pt"),
        )


def load_from_checkpoint(
    path: str,
    ml_settings: dict,
    input: TensorMap,
    output: TensorMap,
    train_mode: bool = True,
):
    """
    Initializes training objects then loads the state dict from a checkpoint
    file. Returns the epoch, model, optimizer, and RhoLoss and CoeffLoss
    functions.

    Warning: loads model in training mode by default.
    """
    # First initialize the various objects
    (
        model,
        optimizer,
        rho_loss_fn,
        coeff_loss_fn,
        scheduler,
    ) = init_training_objects(ml_settings, input, output)

    # Now load the checkpoint
    checkpoint = torch.load(path)

    # Update state of initialized objects
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # If scheduler is not None, load its state dict
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Set model to train/eval mode
    if train_mode:
        model.train()
    else:
        model.eval()

    return model, optimizer, rho_loss_fn, coeff_loss_fn, scheduler
