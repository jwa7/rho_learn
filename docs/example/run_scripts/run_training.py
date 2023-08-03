import os
import pprint
import numpy as np
import torch

import equistore

import rholearn
from rholearn import io, features, pretraining, training, utils
from settings import data_settings, ml_settings


# Create simulation run directory and save simulation
io.check_or_create_dir(ml_settings["run_dir"])
with open(os.path.join(ml_settings["run_dir"], "ml_settings.txt"), "a+") as f:
    f.write(f"ML Settings:\n{pprint.pformat(ml_settings)}\n\n")

# IMPORTANT! - set the torch default dtype
torch.set_default_dtype(ml_settings["torch"]["dtype"])

# Pre-construct the appropriate torch objects (i.e. models, loss fxns)
pretraining.construct_torch_objects_in_train_dir(
    data_settings["data_dir"], ml_settings["run_dir"], ml_settings, 
)

# Define the training subdirectory
train_rel_dir = ""
train_run_dir = os.path.join(ml_settings["run_dir"], train_rel_dir)

# Load training data and torch objects
data, model, loss_fn, optimizer = pretraining.load_training_objects(
    train_rel_dir, data_settings["data_dir"], ml_settings, ml_settings["training"]["restart_epoch"]
)

# Unpack the data
in_train, in_test, out_train, out_test = data

# Execute model training
training.train(
    in_train=in_train,
    out_train=out_train,
    in_test=in_test,
    out_test=out_test,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    n_epochs=ml_settings["training"]["n_epochs"],
    save_interval=ml_settings["training"]["save_interval"],
    save_dir=train_run_dir,
    restart=ml_settings["training"]["restart_epoch"],
)