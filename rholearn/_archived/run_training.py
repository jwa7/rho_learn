import gc
import os
import time
import sys
import warnings

import numpy as np
import torch

import intel_extension_for_pytorch as ipex

import equistore

from rholearn import utils, data, loss, models, training
from settings import data_settings, ml_settings, torch_settings

torch.set_default_dtype(torch_settings["tensor"]["dtype"])
warnings.filterwarnings("ignore")

# Get the training subset index
subset = int(sys.argv[1])

# ===== SET UP LOG FILE =====

# Make a run dir and checkpoint dir for saving results
if not os.path.exists(ml_settings["run_dir"]):
    os.mkdir(ml_settings["run_dir"])

if not os.path.exists(os.path.join(ml_settings["run_dir"], "checkpoints")):
    os.mkdir(os.path.join(ml_settings["run_dir"], "checkpoints"))

# Define a log file
log_file = os.path.join(ml_settings["run_dir"], "log.txt")
utils.log(log_file, "# Start process.")


# ===== DATASET AND TRAIN TEST SPLIT IDXS =====
utils.log(log_file, "# Train/test split")

# Get the grouped indices for train/test(/val) splits
tmp_idxs = data.group_idxs(
    all_idxs=data_settings["all_idxs"],
    n_groups=data_settings["n_groups"],
    group_sizes=data_settings["group_sizes"],
    shuffle=data_settings["shuffle"],
    seed=data_settings["seed"] * (subset + 1),
)
if data_settings["n_groups"] == 1:
    train_idxs, = tmp_idxs
    test_idxs = np.array([])
    val_idxs = np.array([])
elif data_settings["n_groups"] == 2:
    train_idxs, test_idxs = tmp_idxs
    val_idxs = np.array([])
elif data_settings["n_groups"] == 3:
    train_idxs, test_idxs, val_idxs = tmp_idxs

# Define a training subset if applicable
if data_settings.get("subset_sizes") is not None:
    subset_sizes = data_settings["subset_sizes"]
    assert len(subset_sizes) == data_settings["n_train_subsets"]
    train_idxs = train_idxs[: subset_sizes[subset]]
else:
    if data_settings["n_train_subsets"] > 0:
        subset_sizes = data.get_log_subset_sizes(
            len(train_idxs), data_settings["n_train_subsets"]
        )

    train_idxs = train_idxs[: subset_sizes[subset]]

# Define a test subset if not doing test batching
if ml_settings["loading"]["test"]["do_batching"] is False:
    if ml_settings["loading"]["test"]["batch_size"] < len(test_idxs):
        test_idxs = test_idxs[: ml_settings["loading"]["test"]["batch_size"]]

utils.log(log_file, "# Save split data idxs")
np.savez(
    os.path.join(ml_settings["run_dir"], "idxs.npz"),
    train=train_idxs,
    test=test_idxs,
    val=val_idxs,
)
utils.log(log_file, "# Num train structures: " + str(len(train_idxs)))
utils.log(log_file, "# Num test structures: " + str(len(test_idxs)))
utils.log(log_file, "# Num val structures: " + str(len(val_idxs)))

# Build density dataset
utils.log(log_file, "# Building dataset")
rho_data = data.RhoData(
    all_idxs=np.concatenate([train_idxs, test_idxs]),
    train_idxs=train_idxs,
    input_dir=data_settings["input_dir"],
    output_dir=data_settings["output_dir"],
    overlap_dir=data_settings["overlap_dir"],
    keep_in_mem=True,
    calc_out_train_inv_means=data_settings["calc_out_train_inv_means"],
    calc_out_train_std_dev=data_settings["calc_out_train_std_dev"],
    filenames=data_settings["filenames"],
    **torch_settings["tensor"],
)

# Write std dev of output training data to file
if data_settings["calc_out_train_std_dev"] is True:
    out_train_std_dev = rho_data.out_train_std_dev.detach().numpy()
    utils.log(log_file, "# Write out train stddev to file")
    utils.log(log_file, f"# out_train_std_dev: {np.round(out_train_std_dev, 15)}")
    np.savez("std_dev.npz", out_train=out_train_std_dev)


# ===== INIT TORCH OBJECTS =====

# Retrieve and use the invariant means of the output training data only if a)
# they have been calculated and b) we want to train on baselined coefficients
out_train_inv_means = None
if (
    data_settings["calc_out_train_inv_means"]
    and ml_settings["model"]["train_on_baselined_coeffs"]
):
    utils.log(log_file, "# Block models will be trained on baselined coefficients")
    out_train_inv_means = rho_data.out_train_inv_means

# Initialize training objects
restart_epoch = ml_settings["training"]["restart_epoch"]
if restart_epoch == 0:
    utils.log(log_file, "# Initializing training objects")
    model = models.RhoModel(
        model_type=ml_settings["model"]["model_type"],
        input=rho_data[rho_data._all_idxs[0]][1],
        output=rho_data[rho_data._all_idxs[0]][2],
        bias_invariants=ml_settings["model"]["bias_invariants"],
        out_train_inv_means=out_train_inv_means,
        **ml_settings["model"]["args"],
        **torch_settings["tensor"],
    )
    optimizer = ml_settings["optimizer"]["algorithm"](
        model.parameters(), **ml_settings["optimizer"]["args"]
    )
    scheduler = ml_settings["scheduler"]["algorithm"](
        optimizer, **ml_settings["scheduler"]["args"]
    )

# Or load from checkpoint
else:
    utils.log(log_file, f"# Load training objects from checkpoint {restart_epoch}")
    chkpt = training.load_from_checkpoint(
        path=os.path.join(
            ml_settings["run_dir"], "checkpoints", f"checkpoint_{restart_epoch}.pt"
        )
    )
    model, optimizer, scheduler = chkpt["model"], chkpt["optimizer"], chkpt["scheduler"]

# Initialize loss functions
rho_loss_fn = loss.RhoLoss()
coeff_loss_fn = loss.CoeffLoss()

# Reinitialize optimizer
if ml_settings["optimizer"]["reinitialize"] is True:
    utils.log(log_file, f"# Reinitializing optimizer")
    optimizer = ml_settings["optimizer"]["algorithm"](
        model.parameters(),
        **ml_settings["optimizer"]["args"],
    )

# Reinitialize scheduler
if ml_settings["scheduler"]["reinitialize"] is True:
    utils.log(log_file, f"# Reinitializing scheduler")
    scheduler = ml_settings["scheduler"]["algorithm"](
        optimizer=optimizer,
        **ml_settings["scheduler"]["args"],
    )

# Optimize model and optimizer for Intel CPUs
if torch_settings["use_ipex"] is True:
    utils.log(log_file, f"# Optimizing model and optimizer for Intel CPUs")
    model, optimizer = ipex.optimize(model, optimizer=optimizer)

# Initialize the train and test loaders
utils.log(log_file, "# Constructing training data loader")
use_rho_loss = False
if ml_settings["training"]["learn_on_rho_at_epoch"] == 0:
    use_rho_loss = True
train_loader = data.RhoLoader(
    rho_data,
    idxs=train_idxs,
    get_overlaps=use_rho_loss,
    batch_size=ml_settings["loading"]["train"]["batch_size"],
)
utils.log(log_file, "# Constructing testing data loader")
test_loader = data.RhoLoader(
    rho_data,
    idxs=test_idxs,
    get_overlaps=True,
    batch_size=ml_settings["loading"]["test"]["batch_size"],
)

# Pre-collate train data if not performing train batching
if ml_settings["loading"]["train"]["do_batching"] is False:
    utils.log(log_file, "# Pre-collating single batch of train data")
    if use_rho_loss:
        train_batch_idxs, x_train, c_train, s_train = next(iter(train_loader))
    else:
        train_batch_idxs, x_train, c_train = next(iter(train_loader))

# Pre-collate test data if not performing test batching
if ml_settings["loading"]["test"]["do_batching"] is False:
    utils.log(log_file, "# Pre-collating single batch of test data")
    if use_rho_loss:
        test_batch_idxs, x_test, c_test, s_test = next(iter(test_loader))
    else:
        test_batch_idxs, x_test, c_test = next(iter(test_loader))



# ===== TRAINING LOOP =====

# Start training
utils.log(log_file, "# Start model training")
utils.log(log_file, "# epoch train_loss test_loss lr time learning_on_rho grad_norm")
train_losses = []
test_losses = []
for epoch in range(
    ml_settings["training"]["restart_epoch"] + 1,
    ml_settings["training"]["n_epochs"] + 1,
):
    # Start timer
    t0 = time.time()
    gc.collect()

    # Set some epoch-dependent settings
    check_args = (
        True
        if epoch == 0 or epoch == ml_settings["training"]["learn_on_rho_at_epoch"]
        else False
    )

    # If we're now switching to learning on rho, reinitialize the train loader
    if epoch == ml_settings["training"]["learn_on_rho_at_epoch"]:
        use_rho_loss = True
        train_loader = data.RhoLoader(
            rho_data,
            idxs=train_idxs,
            get_overlaps=True,
            batch_size=ml_settings["loading"]["train"]["batch_size"],
        )

    # ===== Evaluate train loss
    train_loss_epoch = 0

    # Option 1) perform train batching
    if ml_settings["loading"]["train"]["do_batching"] is True:
    
        for train_batch in train_loader:
            # Reset gradients
            optimizer.zero_grad()

            # Unpack train batch
            if use_rho_loss:
                train_batch_idxs, x_train, c_train, s_train = train_batch
            else:
                train_batch_idxs, x_train, c_train = train_batch

            # Make a prediction
            c_train_pred = model(x_train, check_args=check_args)

            # Evaluate the loss with either CoeffLoss or RhoLoss
            if use_rho_loss:
                train_loss_batch = rho_loss_fn(
                    c_train_pred, c_train, s_train, check_args=check_args
                )
            else:  # use CoeffLoss
                train_loss_batch = coeff_loss_fn(
                    c_train_pred, c_train, check_args=check_args
                )

            # Calculate gradient and update parameters
            train_loss_batch.backward()
            optimizer.step()

            train_loss_epoch += (train_loss_batch / len(train_batch_idxs))

    # Option 2) use a single train batch, pre-collated
    else:
        # Reset gradients
        optimizer.zero_grad()

        # Make a prediction
        c_train_pred = model(x_train, check_args=check_args)

        # Evaluate the loss with either CoeffLoss or RhoLoss
        if use_rho_loss:
            train_loss_batch = rho_loss_fn(
                c_train_pred, c_train, s_train, check_args=check_args
            )
        else:  # use CoeffLoss
            train_loss_batch = coeff_loss_fn(
                c_train_pred, c_train, check_args=check_args
            )

        # Calculate gradient and update parameters
        train_loss_batch.backward()
        optimizer.step()

        train_loss_epoch += (train_loss_batch / len(train_batch_idxs))


    # Store training loss, divided by the number of structures in the batch
    train_losses.append(train_loss_epoch)

    # ===== Evaluate test loss *on the density*
    with torch.no_grad():
        test_loss_epoch = 0
        # Option 1) perform test batching
        if ml_settings["loading"]["test"]["do_batching"] is True:
            # Iterate over test batches: calculate the test loss
            for test_batch in test_loader:
                # Unpack test batch
                test_batch_idxs, x_test, c_test, s_test = test_batch
                # Make a prediction
                c_test_pred = model(x_test, check_args=check_args)
                # Evaluate and store test loss per structure
                test_loss_batch = rho_loss_fn(
                    c_test_pred, c_test, s_test, check_args=check_args
                )
                test_loss_epoch += (test_loss_batch / len(test_batch_idxs))

        # Option 2) use a single test batch, pre-collated
        else:
            # Make a prediction
            c_test_pred = model(x_test, check_args=check_args)
            # Evaluate and store test loss per structure
            test_loss_epoch = rho_loss_fn(
                c_test_pred, c_test, s_test, check_args=check_args
            ) / len(test_batch_idxs)

        # Store test loss, divided by the number of structures in the batch
        test_losses.append(test_loss_epoch)

    # Calculate gradient norm
    grad_norm = 0
    for p in model.parameters():
        grad_norm += p.grad.detach().data.norm(2).item() ** 2
    grad_norm = grad_norm ** (1. / 2)

    # Perform gradient clipping if applicable
    if ml_settings["optimizer"]["clip_grad_norm"] is True:
        assert isinstance(ml_settings["optimizer"]["max_norm"], float)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=ml_settings["optimizer"]["max_norm"]
        )

    # Update model parameters via the scheduler
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        # Needs the metric
        scheduler.step(test_losses[-1])
    else:  # works on milestones
        scheduler.step()

    # Save checkpoint
    if epoch % ml_settings["training"]["save_interval"] == 0:
        training.save_checkpoint(
            os.path.join(
                ml_settings["run_dir"], "checkpoints", f"checkpoint_{epoch}.pt"
            ),
            model,
            optimizer,
            scheduler=scheduler,
        )

    # Write log for the epoch
    utils.log(
        log_file,
        f"{epoch} "
        f"{np.round(train_losses[-1].detach().numpy(), 15)} "
        f"{np.round(test_losses[-1].detach().numpy(), 15)} "
        f"{np.round(optimizer.param_groups[0]['lr'], 7)} "
        f"{np.round(time.time() - t0, 7)} "
        f"{1 if use_rho_loss else 0} "
        f"{np.round(grad_norm, 7)} "
    )

# All epochs complete
utils.log(log_file, "# Training complete.")
