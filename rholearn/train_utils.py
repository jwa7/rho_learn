"""
Module containing functions to perform model training and evaluation steps.
"""

import os
from os.path import exists, join
import time
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import ase
import numpy as np
import torch

import metatensor.torch as mts
from metatensor.torch.learn.data import IndexedDataset
from metatensor.torch.learn.nn import Sequential

import rascaline.torch

from rhocalc.aims import aims_fields
from rhocalc.ase import structure_builder
from rholearn import data
from rholearn.model import DescriptorCalculator


def load_dft_data(path: str, torch_kwargs: dict) -> torch.ScriptObject:
    """Loads a TensorMap from file and converts its backend to torch"""
    return mts.sort(mts.load(path)).to(**torch_kwargs)


def parse_use_overlap_setting(
    use_overlap: Union[bool, int], epochs: torch.Tensor
) -> List[bool]:
    """
    Returns a list of boolean values, indicating whether to use the auxiliary
    data (i.e. overlaps) in loss evaluation at each of the epochs in `epochs`.
    """
    if isinstance(use_overlap, bool):
        return [use_overlap] * len(epochs)

    assert isinstance(use_overlap, int)

    parsed_use_overlap = []
    for epoch in epochs:
        if epoch < use_overlap:
            parsed_use_overlap.append(False)
        else:
            parsed_use_overlap.append(True)

    return parsed_use_overlap


def training_step(
    train_loader,
    model,
    optimizer,
    loss_fn,
    check_metadata: bool = False,
    use_aux: bool = False,
) -> Tuple[torch.Tensor]:
    """
    Performs a single epoch of training by minibatching. Returns the training loss for
    the epoch, normalized by the total number of structures across all minibatches.
    """
    model.train()

    train_loss_epoch, n_train_epoch = 0, 0
    for train_batch in train_loader:

        optimizer.zero_grad()  # zero grads
        out_train_pred = model(  # forward pass
            descriptor=train_batch.descriptor, check_metadata=check_metadata
        )

        # Drop the blocks from the prediction that aren't part of the target
        if isinstance(out_train_pred, torch.ScriptObject):
            out_train_pred = mts.TensorMap(
                keys=train_batch.target.keys,
                blocks=[out_train_pred[key].copy() for key in train_batch.target.keys],
            )
        else:
            out_train_pred = [
                mts.TensorMap(
                    keys=train_batch.target[i].keys,
                    blocks=[
                        out_train_pred[i][key].copy()
                        for key in train_batch.target[i].keys
                    ],
                )
                for i in range(len(out_train_pred))
            ]

        train_loss_batch = loss_fn(  # train loss
            input=out_train_pred,
            target=train_batch.target,
            overlap=train_batch.aux if use_aux else None,
            check_metadata=check_metadata,
        )
        train_loss_batch.backward()  # backward pass
        optimizer.step()  # update parameters
        train_loss_epoch += train_loss_batch  # store loss
        n_train_epoch += len(
            train_batch.sample_id
        )  # accumulate num structures in epoch

    train_loss_epoch /= n_train_epoch  # normalize loss by num structures

    return train_loss_epoch


def validation_step(
    val_loader, model, loss_fn, check_metadata: bool = False, use_aux: bool = False
) -> torch.Tensor:
    """
    Performs a single validation step by minibatching. Returns the validation loss,
    normalized by the total number of structures across all minibatches.
    """
    with torch.no_grad():
        val_loss_epoch, out_val_pred = torch.nan, None
        val_loss_epoch = 0
        n_val_epoch = 0
        for val_batch in val_loader:  # minibatches

            out_val_pred = model(  # forward pass
                descriptor=val_batch.descriptor, check_metadata=check_metadata
            )
            val_loss_batch = loss_fn(  # validation loss
                input=out_val_pred,
                target=val_batch.target,
                overlap=val_batch.aux if use_aux else None,
                check_metadata=check_metadata,
            )
            val_loss_epoch += val_loss_batch  # store loss
            n_val_epoch += len(val_batch.sample_id)

        val_loss_epoch /= n_val_epoch  # normalize loss

        return val_loss_epoch


def step_scheduler(val_loss: torch.Tensor, scheduler) -> None:
    """Updates the scheduler parameters based on the validation loss"""
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_losses[0])
    else:
        scheduler.step()


def get_block_losses(model, dataloader):
    """
    For all structures in the dataloader, sums the squared errors on the
    predictions from the model and returns the total SE for each block in a
    dictionary.
    """
    with torch.no_grad():

        def block_loss(input, target):
            return torch.mean( ( (input - target) / target ) ** 2)

        # Get all descriptors and targets over all minibatches
        all_batches = [batch for batch in dataloader]
        if isinstance(all_batches[0].target, torch.ScriptObject):  # single TensorMap
            expect_list = False
        else:  # list of TensorMap
            expect_list = True

        if expect_list:
            descriptor = [desc for batch in all_batches for desc in batch.descriptor]
            target = [targ for batch in all_batches for targ in batch.target]

        else:
            descriptor = mts.join(
                [batch.descriptor for batch in all_batches],
                axis="samples",
                remove_tensor_name=True,
            )
            target = mts.join(
                [batch.target for batch in all_batches],
                axis="samples",
                remove_tensor_name=True,
            )

        prediction = model(descriptor=descriptor, check_metadata=False)
        if expect_list:
            keys = prediction[0].keys
        else:
            keys = prediction.keys

        block_losses = {tuple(key): 0 for key in keys}
        for key in keys:
            if expect_list:  # list of TensorMap
                for pred, targ in zip(prediction, target):
                    if key not in targ.keys:  # target does not have this key
                        continue
                    # Remove predicted blocks that aren't in the target
                    pred = mts.TensorMap(
                        keys=targ.keys, blocks=[pred[key].copy() for key in targ.keys]
                    )
                    assert mts.equal_metadata(pred, targ)
                    block_losses[tuple(key)] += block_loss(
                        pred[key].values, targ[key].values
                    )
            else:  # single TensorMap
                if key not in target.keys:  # target does not have this key
                    continue
                # Remove predicted blocks that aren't in the target
                prediction = mts.TensorMap(
                    keys=target.keys,
                    blocks=[prediction[key].copy() for key in target.keys],
                )
                assert mts.equal_metadata(prediction, target)
                block_losses[tuple(key)] = block_loss(
                    prediction[key].values, target[key].values
                )

    return block_losses


def save_checkpoint(model: torch.nn.Module, optimizer, scheduler, chkpt_dir: str):
    """
    Saves model object, model state dict, optimizer state dict, scheduler state dict,
    to file.
    """
    if not exists(chkpt_dir):  # create chkpoint dir
        os.makedirs(chkpt_dir)

    torch.save(model, join(chkpt_dir, f"model.pt"))  # model obj
    torch.save(  # model state dict
        model.state_dict(),
        join(chkpt_dir, f"model_state_dict.pt"),
    )
    # Optimizer and scheduler
    torch.save(optimizer.state_dict(), join(chkpt_dir, f"optimizer_state_dict.pt"))
    if scheduler is not None:
        torch.save(
            scheduler.state_dict(),
            join(chkpt_dir, f"scheduler_state_dict.pt"),
        )


def run_training_sbatch(run_dir: str, python_command: str, **kwargs) -> None:
    """
    Writes a bash script to `fname` that allows running of model training.
    `run_dir` must contain two files; "run_training.py" and "settings.py".
    """
    top_dir = os.getcwd()

    # Copy training script and settings
    shutil.copy(join(top_dir, "ml_settings.py"), join(run_dir, "ml_settings.py"))

    os.chdir(run_dir)
    fname = "run_training.sh"

    with open(join(run_dir, fname), "w+") as f:
        # Make a dir for the slurm outputs
        if not exists(join(run_dir, "slurm_out")):
            os.mkdir(join(run_dir, "slurm_out"))

        f.write("#!/bin/bash\n")  # Write the header
        for tag, val in kwargs.items():  # Write the sbatch parameters
            f.write(f"#SBATCH --{tag}={val}\n")
        f.write(f"#SBATCH --output={join(run_dir, 'slurm_out', 'slurm_train.out')}\n")
        f.write("#SBATCH --get-user-env\n\n")

        # Define the run directory, cd to it, run command
        f.write(f"RUNDIR={run_dir}\n")
        f.write("cd $RUNDIR\n\n")
        f.write(f"{python_command}\n")

    os.system(f"sbatch {fname}")
    os.chdir(top_dir)


def run_python_local(run_dir: str, python_command: str) -> None:
    """
    Runs a python command locally in `run_dir`.
    """
    top_dir = os.getcwd()
    os.chdir(run_dir)
    os.system(f"{python_command}")
    os.chdir(top_dir)
