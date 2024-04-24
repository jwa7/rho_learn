"""
Module containing functions to perform model training and evaluation steps.
"""

import os
import time
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

import metatensor

from rholearn import io
from rhocalc.aims import aims_parser


def training_step(
    model,
    loss_fn,
    optimizer,
    train_loader,
    scheduler=None,
    check_metadata=False,
    use_aux: bool = False,
) -> Tuple[torch.Tensor]:
    """
    Performs a single epoch of training by minibatching.
    """
    model.train()

    train_loss_epoch, n_train_epoch = 0, 0
    for train_batch in train_loader:

        optimizer.zero_grad()  # zero grads

        id_train, struct_train, in_train, out_train, aux_train = (
            train_batch  # unpack batch
        )
        if not use_aux:
            aux_train = None

        out_train_pred = model(  # forward pass
            descriptor=in_train, check_metadata=check_metadata
        )

        # Drop the blocks from the prediction that aren't part of the target
        if isinstance(out_train_pred, torch.ScriptObject):
            out_train_pred = mts.TensorMap(
                keys=out_train.keys,
                blocks=[out_train_pred[key].copy() for key in out_train.keys],
            )
        else:
            out_train_pred = [
                mts.TensorMap(
                    keys=out_train[i].keys,
                    blocks=[out_train_pred[i][key].copy() for key in out_train[i].keys],
                )
                for i in range(len(out_train_pred))
            ]

        train_loss_batch = loss_fn(  # train loss
            input=out_train_pred,
            target=out_train,
            overlap=aux_train,
            check_metadata=check_metadata,
        )
        train_loss_batch.backward()  # backward pass
        optimizer.step()  # update parameters
        train_loss_epoch += train_loss_batch  # store loss
        n_train_epoch += len(id_train)  # accumulate num structures in epoch

    train_loss_epoch /= n_train_epoch  # normalize loss by num structures

    return train_loss_epoch


def validation_step(
    model,
    loss_fn,
    val_loader,
) -> torch.Tensor:
    """
    Performs a single validation step
    """
    with torch.no_grad():
        val_loss_epoch, out_val_pred = torch.nan, None
        val_loss_epoch = 0
        n_val_epoch = 0
        for val_batch in val_loader:  # minibatches

            id_val, struct_val, in_val, out_val, aux_val = val_batch  # unpack batch
            if not use_aux:
                aux_val = None

            out_val_pred = model(in_val, check_metadata=check_metadata)
            val_loss_batch = loss_fn(  # validation loss
                input=out_val_pred,
                target=out_val,
                overlap=aux_val,
                check_metadata=check_metadata,
            )
            val_loss_epoch += val_loss_batch  # store loss
            n_val_epoch += len(id_val)

        val_loss_epoch /= n_val_epoch  # normalize loss

        return val_loss_epoch


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
    structure_id, structure, descriptor = [], [], []
    for batch in dataloader:
        for idx in batch.sample_id:
            structure_id.append(idx)
        for struct in batch.structure:
            structure.append(struct)
        for desc in batch.descriptor:
            descriptor.append(desc)

    # Make prediction with the model
    with torch.no_grad():
        prediction = model(  # return a list of TensorMap (or ScriptObject)
            descriptor=descriptor, check_metadata=True
        )

    if not calculate_error:
        return np.nan

    assert reference_dir is not None

    percent_maes = []
    for A, frame, prediction in zip(structure_id, STRUCTURES, predictions):

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


def get_block_losses(model, dataloader):
    """
    For all structures in the dataloader, sums the squared errors on the
    predictions from the model and returns the total SE for each block in a
    dictionary.
    """
    with torch.no_grad():
        # Get all the descriptors structures
        descriptor = [desc for batch in dataloader for desc in batch.descriptor]
        target = [targ for batch in dataloader for targ in batch.target]
        prediction = model(descriptor=descriptor, check_metadata=False)
        keys = prediction[0].keys

        block_losses = {tuple(key): 0 for key in keys}
        for key in keys:
            for pred, targ in zip(prediction, target):
                if key not in targ.keys:  # target does not have this key
                    continue
                # Remove predicted blocks that aren't in the target
                pred = mts.TensorMap(
                    keys=targ.keys, blocks=[pred[key].copy() for key in targ.keys]
                )
                assert mts.equal_metadata(pred, targ)
                block_losses[tuple(key)] += torch.nn.MSELoss(reduction="sum")(
                    pred[key].values, targ[key].values
                )

    return block_losses


def run_training_sbatch(run_dir: str, **kwargs) -> None:
    """
    Writes a bash script to `fname` that allows running of model training.
    `run_dir` must contain two files; "run_training.py" and "settings.py".
    """
    top_dir = os.getcwd()
    os.chdir(run_dir)

    fname = "run-training.sh"

    with open(os.path.join(run_dir, fname), "w+") as f:

        # Make a dir for the slurm outputs
        if not os.path.exists(os.path.join(run_dir, "slurm_out")):
            os.mkdir(os.path.join(run_dir, "slurm_out"))

        # Write the header
        f.write("#!/bin/bash\n")

        # Write the sbatch parameters
        for tag, val in kwargs.items():
            f.write(f"#SBATCH --{tag}={val}\n")

        f.write(
            f"#SBATCH --output={os.path.join(run_dir, 'slurm_out', 'slurm_train.out')}\n"
        )
        f.write("#SBATCH --get-user-env\n")
        f.write("\n\n")

        # Define the run directory and cd to it
        f.write("# Define the run directory and cd into it\n")
        f.write(f"RUNDIR={run_dir}\n")
        f.write("cd $RUNDIR\n\n")

        f.write("# Run the Python command\n")
        f.write("python run_training.py\n\n")

    os.system(f"sbatch {fname}")
    os.chdir(top_dir)

    return


def run_training_local(run_dir: str) -> None:
    """
    Runs the training loop in the local environment. `run_dir` must contain this
    file "run_training.py", and the settings file "settings.py".
    """
    top_dir = os.getcwd()
    os.chdir(run_dir)

    os.system("python run_training.py")

    os.chdir(top_dir)

    return