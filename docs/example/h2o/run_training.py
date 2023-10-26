# Useful standard and scientific ML libraries
import os
import shutil
import time
from typing import Tuple

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol
import torch

# M-Stack packages
import metatensor  # storage format for atomistic ML
import chemiscope  # interactive molecular visualization
import rascaline  # generating structural representations
from metatensor import Labels, TensorBlock, TensorMap
from rascaline.utils import clebsch_gordan

# Interfacing with FHI-aims
from rhocalc.aims import aims_calc, aims_parser

# Torch-based density leaning
from rholearn import io, data, loss, models, predictor, train
from settings import *

# =================================================
# Basic set up
# =================================================

# Log the start of the training
log_path = os.path.join(run_dir, "training.log")
io.log(log_path, "# Starting training")


# Define a function that returns the data directory for a given structure based
# on its structure index
def struct_dir(A):
    return os.path.join(data_dir, f"{A}", "0")


# Create a callable for directories to save predictions by structure index
def pred_dir(A):
    return os.path.join("/scratch/abbott/h2o_homo/ml/predictions", f"{A}")


# Define callable for path to processed data (i.e. TensorMaps)
def processed_dir(A):
    return os.path.join(struct_dir(A), "processed/")


# Define dir for running ML calculations
chkpt_dir = os.path.join(run_dir, "checkpoints")
if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)

# Copy settings file to run dir
shutil.copy(
    os.path.join(os.getcwd(), "settings.py"), 
    os.path.join(run_dir, "settings.py")
)


# =================================================
# Dataset and dataloaders
# =================================================

# Perform a train/test/val split of structure idxs
train_idxs, test_idxs, val_idxs = data.group_idxs(
    idxs=idxs,
    n_groups=crossval_settings["n_groups"],
    group_sizes=crossval_settings["group_sizes"],
    shuffle=crossval_settings["shuffle"],
    seed=crossval_settings["seed"],
)
np.savez(
    os.path.join(run_dir, "idxs.npz"),
    idxs=idxs,
    train_idxs=train_idxs,
    test_idxs=test_idxs,
    val_idxs=val_idxs,
)
io.log(log_path, f"# total num. idxs: {len(idxs)}")

# Construct dataset
rho_data = data.RhoData(
    idxs=np.concatenate([train_idxs, test_idxs, val_idxs]),
    in_path=lambda A: os.path.join(processed_dir(A), "lsoap.npz"),
    out_path=lambda A: os.path.join(processed_dir(A), "ri_coeffs.npz"),
    aux_path=lambda A: os.path.join(processed_dir(A), "ri_ovlp.npz"),
    keep_in_mem=ml_settings["loading"]["train"]["keep_in_mem"],
    **torch_settings,
)

# if crossval_settings["calc_out_train_std_dev"]:
#     out_train_std_dev = rho_data.out_train_std_dev.detach().numpy()
#     io.log(log_path, f"# out_train_std_dev: {out_train_std_dev}")
#     np.savez("ml/std_dev.npz", out_train=out_train_std_dev)

# =================================================
# Initialize model
# =================================================

# If using a non-learnable bias in the form of a invariant baseline, calculate
# it and store it in the model on initialization. This should only be
# calculated on the training data.
if ml_settings["model"]["use_invariant_baseline"]:
    invariant_baseline = rho_data.get_invariant_means(
        idxs=train_idxs, which_data="output"
    )
else:
    invariant_baseline = None

# Calculate the standard deviation of the training data
if data_settings.get("standard_deviation"):
    which_data, which_idxs = data_settings["standard_deviation"]
    tmp_invariant_baseline = invariant_baseline if which_idxs == "train" else None
    which_idxs = {"all": idxs, "train": train_idxs, "test": test_idxs, "val": val_idxs}[which_idxs]
    stddev = rho_data.get_standard_deviation(
        idxs=which_idxs, which_data="output", invariant_baseline=tmp_invariant_baseline,
        use_overlaps=rho_data.aux_path is not None,
    )
    io.log(log_path, f"# stddev: {stddev}")
    np.savez(os.path.join(run_dir, "stddev.npz"), stddev=stddev)


# For descriptor building, we need to store the rascaline settings for
# generating a SphericalExpansion and performing Clebsch-Gordan combinations.
# The `descriptor_builder` function in "predictor.py" contains the 'recipe' for
# using these settings to transform an ASE Atoms object.
descriptor_kwargs = {
    "rascal_settings": rascal_settings,
    "cg_settings": cg_settings,
}

# For target building, the base AIMS settings need to be stored, along with the
# basis set definition.
basis_set = io.unpickle_dict(os.path.join(processed_dir(idxs[0]), "calc_info.pickle"))[
    "basis_set"
]
target_kwargs = {
    "aims_kwargs": {**base_aims_kwargs},
    "basis_set": {**basis_set},
}

# Here we initialize the model with the model architecture options, as well as
# the descriptor/target builder settings needed for end-to-end predictions.
model = models.RhoModel(
    # Required args
    model_type=ml_settings["model"]["model_type"],
    input=rho_data[idxs[0]][1],  # for initializing ...
    output=rho_data[idxs[0]][2],  # ... the metadata of the model
    bias_invariants=ml_settings["model"]["bias_invariants"],
    invariant_baseline=invariant_baseline,
    # For nonlinear model
    hidden_layer_widths=ml_settings["model"].get("hidden_layer_widths"),
    activation_fn=ml_settings["model"].get("activation_fn"),
    bias_nn=ml_settings["model"].get("bias_nn"),
    # For end-to-end predictions
    descriptor_kwargs=descriptor_kwargs,
    target_kwargs=target_kwargs,
    # Torch settings
    **torch_settings,
)

# Settings specific to RI rebuild procedure
ri_kwargs = {
    # Force no SCF
    "sc_iter_limit": 0,
    "postprocess_anyway": True,
    "ri_fit_assume_converged": True,
    # What we want to do
    "ri_fit_rebuild_from_coeffs": True,
    # What we want to output
    "ri_fit_write_rebuilt_field": True,
    "ri_fit_write_rebuilt_field_cube": True,
    "output": ["cube ri_fit"],  # needed for cube files
}

# Update the AIMS and SBATCH kwargs
tmp_aims_kwargs = {**model.target_kwargs["aims_kwargs"]}
tmp_aims_kwargs.update(ri_kwargs)

# Settings for slurm
sbatch_kwargs = {
    "job-name": "h2o-pred",
    "nodes": 1,
    "time": "01:00:00",
    "mem-per-cpu": 2000,
    "partition": "standard",
    "ntasks-per-node": 10,
}
model.update_target_kwargs(
    {
        "aims_path": aims_path,
        "aims_kwargs": tmp_aims_kwargs,
        "sbatch_kwargs": sbatch_kwargs,
    }
)

# Attempt a model I/O
torch.save(model, "_tmp.pt")
torch.load("_tmp.pt")
os.remove("_tmp.pt")


# ======================================================
# Initialize training objects: loaders, loss, optimizers
# ======================================================

# Dataloaders
train_loader = data.RhoLoader(
    rho_data,
    idxs=train_idxs,
    get_aux_data=True,
    batch_size=ml_settings["loading"]["train"]["batch_size"],
)
test_loader = data.RhoLoader(
    rho_data,
    idxs=test_idxs,
    get_aux_data=True,
    batch_size=ml_settings["loading"]["test"]["batch_size"],
)
val_loader = data.RhoLoader(
    rho_data,
    idxs=val_idxs,
    get_aux_data=False,
    batch_size=None,
)

# Loss function, optimizer, scheduler
loss_fn = ml_settings["loss_fn"]["algorithm"]()
optimizer = ml_settings["optimizer"]["algorithm"](
    params=model.parameters(), **ml_settings["optimizer"]["args"]
)
scheduler = ml_settings["scheduler"]["algorithm"](
    optimizer, **ml_settings["scheduler"]["args"]
)

# =================================================
# Training
# =================================================

start_epoch = 1

if ml_settings["training"].get("restart_epoch") is not None:
    start_epoch = ml_settings["training"]["restart_epoch"]
    optimizer.load_state_dict(
        torch.load(os.path.join(chkpt_dir, f"optimizer_{start_epoch}.pt"))
    )
    scheduler.load_state_dict(
        torch.load(os.path.join(chkpt_dir, f"scheduler_{start_epoch}.pt"))
    )
    start_epoch += 1

# Start training loop
io.log(log_path, "# epoch train_loss test_loss train_mae test_mae val_mae time")

for epoch in range(start_epoch, ml_settings["training"]["n_epochs"] + 1):
    # Training step
    t0 = time.time()
    train_loss_epoch, test_loss_epoch = train.training_step(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        check_args=epoch == 1,  # switch off metadata checks after 1st epoch
    )

    # Calculate L1 error against real-space QM scalar fields
    mean_maes = {"train": -1, "test": -1, "val": -1}
    if (epoch - 1) % ml_settings["validation"]["interval"] == 0:
        for tmp_idxs, category in zip(
            [train_idxs, test_idxs],  # val_idxs],
            ["train", "test"],  # "val"]
        ):
            # Get frames and make prediction
            tmp_frames = [all_frames[A] for A in tmp_idxs]
            pred_coeffs, pred_fields = model.predict(
                structure_idxs=tmp_idxs,
                frames=tmp_frames,
                save_dir=pred_dir,
            )

            # Evaluate mean L1 Error
            mean_mae = []
            for A, pred_field in zip(tmp_idxs, pred_fields):
                # Get grids and check they're the same in the SCF and ML directories
                grid = np.loadtxt(os.path.join(struct_dir(A), "partition_tab.out"))
                # assert np.allclose(scf_grid, np.loadtxt(os.path.join(model.target_kwargs["pred_dir"](A), "partition_tab.out")))

                # Get L1 error vs real-space QM scalar field
                target_field = np.loadtxt(os.path.join(struct_dir(A), "rho_ref.out"))
                mae = aims_parser.get_percent_mae_between_fields(
                    input=pred_field,
                    target=target_field,
                    grid=grid,
                )
                mean_mae.append(mae)
            mean_maes[category] = np.mean(mean_mae)

    # Write epoch results
    io.log(
        log_path,
        f"{epoch} {train_loss_epoch} {test_loss_epoch} {mean_maes['train']} {mean_maes['test']} {mean_maes['val']} {time.time() - t0}",
    )
    # Save model, optimizer, scheduler
    if (epoch - 1) % ml_settings["training"]["save_interval"] == 0:
        torch.save(model, os.path.join(chkpt_dir, f"model_{epoch}.pt"))
        torch.save(
            optimizer.state_dict(), os.path.join(chkpt_dir, f"optimizer_{epoch}.pt")
        )
        if scheduler is not None:
            torch.save(
                scheduler.state_dict(), os.path.join(chkpt_dir, f"scheduler_{epoch}.pt")
            )
