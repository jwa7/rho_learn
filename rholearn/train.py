"""
Contains function to perform a single training step.
"""
from typing import Tuple
import torch


def training_step(
    train_loader,
    test_loader,
    model,
    loss_fn,
    optimizer,
    scheduler=None,
    check_args=False,
) -> Tuple[torch.Tensor]:
    """
    Perform a single training step using mini-batch gradient descent. Returns
    the train and test losses for the epoch.
    """

    # Iterate over train batches
    train_loss_epoch = 0
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

    train_loss_epoch /= len(idxs_train)

    # Update the learning rate based on the train loss
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss_epoch)  # Needs the metric
        else:
            scheduler.step()  # works on milestones

    # Iterate over test batches: calculate the test loss
    with torch.no_grad():  # don't track gradients for test loss
        test_loss_epoch = 0
        for test_batch in test_loader:
            # Unpack test batch
            idxs_test, in_test, out_test, aux_test = test_batch

            # Make a prediction
            out_test_pred = model(in_test, check_args=check_args)

            # Evaluate test loss
            test_loss_batch = loss_fn(
                input=out_test_pred,
                target=out_test,
                overlap=aux_test,
                check_args=check_args,
            )
            test_loss_epoch += test_loss_batch

        test_loss_epoch /= len(idxs_test)

    return train_loss_epoch, test_loss_epoch
