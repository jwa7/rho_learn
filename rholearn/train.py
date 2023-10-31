"""
Contains the function to perform a single training step.
"""
from typing import Tuple
import torch


def training_step(
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler=None,
    check_args=False,
) -> Tuple[torch.Tensor]:
    """
    Perform a single training step using mini-batch gradient descent. Returns
    the train and validation losses for the epoch.
    """

    # Iterate over train batches
    train_loss_epoch = 0
    n_train_epoch = 0
    for train_batch in train_loader:
        # Reset gradients
        optimizer.zero_grad()

        # Unpack train batch, respectively: structure idxs, lsoap, RI coeffs, RI
        # overlap matrices
        idxs_train, in_train, out_train, aux_train = train_batch

        # Make a prediction
        out_train_pred = model(in_train, check_args=check_args)

        # Evaluate training loss
        train_loss_batch = loss_fn(
            input=out_train_pred,
            target=out_train,
            overlap=aux_train,
            check_args=check_args,
        )

        # Calculate gradient and update parameters
        train_loss_batch.backward()
        optimizer.step()

        train_loss_epoch += train_loss_batch
        n_train_epoch += len(idxs_train)

    # Divide by the total number of train structures iterated over in this epoch
    train_loss_epoch /= n_train_epoch

    # Iterate over validation batches
    with torch.no_grad():  # don't track gradients for test loss
        val_loss_epoch = 0
        n_val_epoch = 0
        for val_batch in val_loader:
            # Unpack test batch
            idxs_val, in_val, out_val, aux_val = val_batch

            # Make a prediction
            out_val_pred = model(in_val, check_args=check_args)

            # Evaluate test loss
            val_loss_batch = loss_fn(
                input=out_val_pred,
                target=out_val,
                overlap=aux_val,
                check_args=check_args,
            )
            val_loss_epoch += val_loss_batch
            n_val_epoch += len(idxs_val)

        # Divide by the total number of test structures iterated over in this epoch
        val_loss_epoch /= n_val_epoch


    # Update the learning rate based on the *validation* loss
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss_epoch)  # Needs the metric
        else:
            scheduler.step()  # works on milestones

    return train_loss_epoch, val_loss_epoch
