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
    val_loader=None,
    scheduler=None,
    check_metadata=False,
    use_aux: bool = True,
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

        if isinstance(optimizer, List):
            for optim in optimizer:
                optim.zero_grad()
        else:
            optimizer.zero_grad()  # zero grads
        out_train_pred = model(  # forward pass
            in_train, check_metadata=check_metadata
        )
        if not use_aux:  # don't use the overlap to train
            aux_train = None
        train_loss_batch = loss_fn(  # train loss
            input=out_train_pred,
            target=out_train,
            overlap=aux_train,
            check_metadata=check_metadata,
        )
        train_loss_batch.backward()  # backward pass

        # Update parameters
        if isinstance(optimizer, List):
            for optim in optimizer:
                optim.step()
        else:
            optimizer.step()
        train_loss_epoch += train_loss_batch # store loss
        n_train_epoch += len(idxs_train)

    train_loss_epoch /= n_train_epoch  # normalize loss

    # ====== Validation step ======
    val_loss_epoch = torch.nan
    out_val_pred = None
    if val_loader is not None:
        with torch.no_grad():
            val_loss_epoch = 0
            n_val_epoch = 0
            for val_batch in val_loader:  # minibatches
                idxs_val, frames_val, in_val, out_val, aux_val = val_batch
                out_val_pred = model(in_val, check_metadata=check_metadata)  # prediction
                val_loss_batch = loss_fn(  # validation loss
                    input=out_val_pred,
                    target=out_val,
                    overlap=aux_val,
                    check_metadata=check_metadata,
                )
                val_loss_epoch += val_loss_batch  # store loss
                n_val_epoch += len(idxs_val)

            val_loss_epoch /= n_val_epoch  # normalize loss

    # ====== Learning rate update ======
    if scheduler is not None:
        if isinstance(scheduler, List):
            for sched in scheduler:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(val_loss_epoch)  # use validation loss
                else:
                    sched.step()  # works on milestones
        else:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss_epoch)  # use validation loss
            else:
                scheduler.step()  # works on milestones

    return train_loss_epoch, val_loss_epoch, out_train_pred, out_val_pred


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
        for idx in batch.sample_id:
            structure_idxs.append(idx)
        for frame in batch.frame:
            frames.append(frame)
        for desc in batch.descriptor:
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
        return np.nan

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


def get_block_losses(model, dataloader):
    """
    For all structures in the dataloader, sums the squared errors on the
    predictions from the model and returns the total SE for each block in a
    dictionary.
    """
    model.eval()
    # Get all the validation structures
    descriptors = [desc for batch in dataloader for desc in batch.descriptor]
    targets = [targ for batch in dataloader for targ in batch.target]
    predictions = model(descriptors)

    for pred, targ in zip(predictions, targets):
        assert metatensor.equal_metadata(pred, targ)

    block_losses = {}
    for key in model._in_metadata.keys:
        block_loss = 0
        for pred, targ in zip(predictions, targets):
            block_loss += torch.nn.MSELoss(reduction="sum")(
                input=pred[key].values, target=targ[key].values
            )
        block_losses[tuple(key)] = block_loss

    return block_losses


def training_loop(
    epochs: List[int],
    model,
    loss_fn,
    optimizer,
    train_loader,
    val_loader,
    test_loader,
    scheduler,
    use_aux: Union[int, None],
    ri_dir: Callable,
    ml_dir: str,
    ml_settings: dict,
    log_path: str,
):
    """
    Runs the training loop for the model. Performs training, validation, and
    evaluation steps, logs results, and saves checkpoint files.
    """

    # Define callable for saving model checkpoints made during training
    if not os.path.exists(os.path.join(ml_dir, "checkpoint")):
        os.makedirs(os.path.join(ml_dir, "checkpoint"))
    def chkpt_dir(epoch):
        return os.path.join(ml_dir, "checkpoint", f"epoch_{epoch}")

    # Define callable for saving model evaluations made during training
    if not os.path.exists(os.path.join(ml_dir, "evaluation")):
        os.makedirs(os.path.join(ml_dir, "evaluation"))
    def eval_dir(A, epoch):
        return os.path.join(ml_dir, "evaluation", f"epoch_{epoch}", f"{A}")

    # Define whether or not to use overlaps in training at each epoch
    if use_aux is None:
        use_ovlps = [True] * len(epochs)
    else:
        if use_aux == -1:
            use_ovlps = [False] * len(epochs)
        elif use_aux == 0:
            use_ovlps = [True] * len(epochs)
        elif use_aux > 0:
            assert use_aux < epochs[-1]
            use_ovlps = [False] * use_aux
            use_ovlps.extend([True] * (len(epochs) - use_aux))

    # Define a log file for writing losses at each epoch
    io.log(log_path, "# epoch train_loss val_loss test_error time lr use_ovlp")

    # Run training loop
    for epoch, use_ovlp in zip(epochs, use_ovlps):

        # ====== Training step ======
        t0 = time.time()
        train_loss_epoch, val_loss_epoch, out_train_pred, out_val_pred = training_step(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            # check_metadata=epoch == 1,  # Check metadata only on 1st epoch
            use_aux=use_ovlp,
        )

        # ====== Evaluation step ======
        test_error_epoch = torch.nan
        if ml_settings.get("evaluation") is not None:
            if epoch % ml_settings["evaluation"]["interval"] == 0:
                loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
                test_error_epoch = evaluation_step(
                    model,
                    dataloader=loaders[ml_settings["evaluation"]["subset"]],
                    save_dir=partial(eval_dir, epoch=epoch),
                    calculate_error=ml_settings["evaluation"]["calculate_error"],
                    target_type=ml_settings["evaluation"]["target_type"],
                    reference_dir=ri_dir,
                )

        # ====== Log results ======
        lr = np.nan
        if scheduler is not None:
            lr = scheduler._last_lr[0]
        dt = time.time() - t0
        if ml_settings["log"].get("interval") is None:
            io.log(
                log_path,
                f"{epoch} {train_loss_epoch} {val_loss_epoch} {test_error_epoch} {dt} {lr} {use_ovlp}",
            )
            if ml_settings["log"].get("block_losses") is True:
                block_losses = get_block_losses(model, val_loader)
                for key, block_loss in block_losses.items():
                    io.log(
                        log_path,
                        f"    key {key} block_loss {block_loss}",
                    )

        else:
            if epoch % ml_settings["log"]["interval"] == 0:
                io.log(
                    log_path,
                    f"{epoch} {train_loss_epoch} {val_loss_epoch} {test_error_epoch} {dt} {lr} {int(use_ovlp)}",
                )
                if ml_settings["log"].get("block_losses") is True:
                    block_losses = get_block_losses(model, val_loader)
                    for key, block_loss in block_losses.items():
                        io.log(
                            log_path,
                            f"    key {key} block_loss {block_loss}",
                        )

        # ====== Save checkpoint ======
        if epoch % ml_settings["training"]["save_interval"] == 0:
            if not os.path.exists(chkpt_dir(epoch)):
                os.makedirs(chkpt_dir(epoch))

            torch.save(model, os.path.join(chkpt_dir(epoch), f"model.pt"))
            torch.save(
                model.state_dict(),
                os.path.join(chkpt_dir(epoch), f"model_state_dict.pt"),
            )
            torch.save(optimizer.state_dict(), os.path.join(chkpt_dir(epoch), f"optimizer.pt"))
            if scheduler is not None:
                torch.save(scheduler.state_dict(), os.path.join(chkpt_dir(epoch), f"scheduler.pt"))

            # for key, block_model in zip(model._in_metadata.keys, model.models):
            #     torch.save(
            #         block_model.state_dict(),
            #         os.path.join(chkpt_dir(epoch), f"model_{tuple(key)}.pt")
            #     )
            # if isinstance(optimizer, List):
            #     for key, optim in zip(model._in_metadata.keys, optimizer):
            #         torch.save(
            #             optim.state_dict(),
            #             os.path.join(chkpt_dir(epoch), f"optimizer_{tuple(key)}.pt")
            #         )
            # else:
            #     torch.save(optimizer.state_dict(), os.path.join(chkpt_dir(epoch), f"optimizer.pt"))
            # if scheduler is not None:
            #     if isinstance(scheduler, List):
            #         for key, sched in zip(model._in_metadata.keys, scheduler):
            #             torch.save(
            #                 sched.state_dict(),
            #                 os.path.join(chkpt_dir(epoch), f"scheduler_{tuple(key)}.pt")
            #             )
            #     else:
            #         torch.save(scheduler.state_dict(), os.path.join(chkpt_dir(epoch), f"scheduler.pt"))

    return


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
            
        f.write(f"#SBATCH --output={os.path.join(run_dir, 'slurm_out', 'slurm_train.out')}\n")
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
