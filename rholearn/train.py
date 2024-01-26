"""
Module containing functions to perform model training and evaluation steps.
"""
import os
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

from rhocalc.aims import aims_parser


def training_step(
    model,
    loss_fn,
    optimizer,
    train_loader,
    val_loader=None,
    scheduler=None,
    check_args=False,
) -> Tuple[torch.Tensor]:
    """
    Perform a single training step using mini-batch gradient descent. Returns
    the train and validation losses for the epoch.
    """
    model.train()

    # ====== Training step ======
    train_loss_epoch = 0
    n_train_epoch = 0
    for train_batch in train_loader:  # minibatches

        idxs_train, frames_train, in_train, out_train, aux_train = train_batch
        optimizer.zero_grad()  # zero grads
        out_train_pred = model(  # forward pass
            in_train, check_args=check_args
        )
        train_loss_batch = loss_fn(  # train loss
            input=out_train_pred,
            target=out_train,
            overlap=aux_train,
            check_args=check_args,
        )
        train_loss_batch.backward()  # backward pass
        optimizer.step()  # update parameters
        train_loss_epoch += train_loss_batch # store loss
        n_train_epoch += len(idxs_train)

    train_loss_epoch /= n_train_epoch  # normalize loss

    # ====== Validation step ======
    with torch.no_grad():

        val_loss_epoch = 0
        n_val_epoch = 0
        for val_batch in val_loader:  # minibatches
            idxs_val, frames_val, in_val, out_val, aux_val = val_batch
            out_val_pred = model(in_val, check_args=check_args)  # prediction
            val_loss_batch = loss_fn(  # validation loss
                input=out_val_pred,
                target=out_val,
                overlap=aux_val,
                check_args=check_args,
            )
            val_loss_epoch += val_loss_batch  # store loss
            n_val_epoch += len(idxs_val)

        val_loss_epoch /= n_val_epoch  # normalize loss

    # ====== Learning rate update ======
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss_epoch)  # use validation loss
        else:
            scheduler.step()  # works on milestones

    return train_loss_epoch, val_loss_epoch


def evaluation_step(
    model,
    dataloader,
    save_dir: Callable,
    calculate_error: bool = False,
    target_type: Optional[str] = None,
    reference_dir: Optional[Callable] = None,
) -> Union[None, float]:
    """
    Evaluates the model by making a prediction (with no gradient tracking) and
    rebuilding the scalar field from these coefficients by calling AIMS. Rebuilt
    scalar fields are saved to `save_dir`, a callable called with each structure
    index as an argument.
    
    If `calculate_error` is set to true, the % MAE (normalized by the number of
    electrons) of the rebuilt scalar field relative to either the DFT scalar
    field (`target_type="ref"`) or the RI scalar field (`target_type="ri"`) is
    returned.

    In this case, the directories where the reference DFT calculation files are
    stored must be specified in `reference_dir`. This is again a callable,
    called with each structure index.
    """
    model.eval()

    assert target_type in ["ref", "ri"]

    # Compile relevant data from all minibatches
    structure_idxs, frames, descriptors = [], [], []
    for batch in dataloader:
        for idx in batch.sample_ids:
            structure_idxs.append(idx)
        for frame in batch.frames:
            frames.append(frame)
        for desc in batch.descriptors:
            descriptors.append(desc)

    # Make predictions
    predictions = model.predict(
        frames=frames,
        structure_idxs=structure_idxs,
        descriptors=descriptors,
        build_targets=True,
        return_targets=calculate_error,
        save_dir=save_dir,
    )

    if not calculate_error:
        return

    assert reference_dir is not None

    percent_maes = []
    for A, frame, prediction in zip(structure_idxs, frames, predictions):

        grid = np.loadtxt(  # integration weights
            os.path.join(reference_dir(A), "partition_tab.out")
            )
        target = np.loadtxt(  # target scalar field
            os.path.join(reference_dir(A), f"rho_{target_type}.out")
        )
        percent_mae = aims_parser.get_percent_mae_between_fields(  # calc MAE
            input=prediction,
            target=target,
            grid=grid,
        )
        percent_maes.append(percent_mae)

    return np.mean(percent_maes)
