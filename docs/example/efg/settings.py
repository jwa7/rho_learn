import os

import ase.io
import numpy as np
import torch

from rholearn import loss


# ====================================================
# ===== Define data settings, including xyz data =====
# ====================================================

# Define the top level dir
top_dir = "/Users/joe.abbott/Documents/phd/code/rho/rho_learn/docs/example/efg"

# Where the generated data should be written
data_dir = os.path.join(top_dir, "data")

# Where ML outputs should be written
ml_dir = os.path.join(top_dir, "ml")

data_settings = {

    # Read in all frames in complete dataset
    "all_frames": ase.io.read(os.path.join(data_dir, "combined_magres_spherical.xyz"), ":"),
    
    # Define a subset of frames
    "n_frames": 30,

    # Define a random seed
    "seed": 12345,
}

# ==========================================================
# ===== Settings for generating structural descriptors =====
# ==========================================================

rascal_settings = {
    "hypers": {
        "cutoff": 3.0,  # Angstrom
        "max_radial": 6,  # Exclusive
        "max_angular": 5,  # Inclusive
        "atomic_gaussian_width": 0.3,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "center_atom_weight": 1.0,
    },
    "compute": {},
}

cg_settings = {
    "angular_cutoff": None,
    "angular_selection": [0, 2],
    "parity_selection": [+1],
}

# =========================================
# ===== Settings for cross validation =====
# =========================================

crossval_settings = {
    # Settings for cross validation
    "n_groups": 3,  # num groups for data split (i.e. 3 for train-test-val)
    "group_sizes": [0.6, 0.2, 0.2],  # the abs/rel group sizes for the data splits
    "shuffle": True,  # whether to shuffle structure indices for the train/test(/val) split
}


# =======================================
# ===== Settings for model training =====
# =======================================

# Setting for torch backend
torch_settings = {
    "dtype": torch.float64,  # important for model accuracy
    "requires_grad": True,
    "device": torch.device(type="cpu"),
}

# Define ML settings
ml_settings = {

    "model": {

        # Model architecture
        "model_type": "linear",  # linear or nonlinear
        "bias_invariants": True,

        # Only if model_type == "nonlinear":
        # "hidden_layer_widths": [64, 64, 64, 64],
        # "activation_fn": torch.nn.SiLU(),
        # "bias_nn": True,
    },
    "loss_fn": {
        "algorithm": loss.L2Loss,
        "args": {

        },
    },
    "optimizer": {
        # "reinitialize": True,
        "algorithm": torch.optim.Adam,
        "args": {
            "lr": 0.001,
        },
        # "clip_grad_norm": True,
        # "max_norm": 0.2,
    },
    "scheduler": {
        # "reinitialize": True,
        "algorithm": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "args": {
            "factor": 0.75,
            "patience": 10,
        },
    },
    "loading": {
        "train": {
            # "do_batching": True,  # whether to batch the train data
            "batch_size": 2,  # number of samples per train batch
            "args": {
                # "num_workers": 0,  # number of workers for data loading
                # "prefetch_factor": None,  # number of batches to prefetch
            },
            "keep_in_mem": True,  # whether to keep the train data in memory or I/O from disk
        },
        "test": {
            # "do_batching": False,  # whether to batch the test data
            "batch_size": 2,  # number of samples per batch
            "args": {
                # "num_workers": 0,  # number of workers for data loading
                # "prefetch_factor": None,  # number of batches to prefetch
            },
        },
    },
    # Parameters for training procedure
    "training": {
        "n_epochs": 30,  # number of total epochs to run
        "save_interval": 5,  # save model and optimizer state every x intervals
        "restart_epoch": None,  # The epoch of the last saved checkpoint. None for no restart
        # "learn_on_rho_at_epoch": 0,  # epoch to start learning on rho instead of coeffs, or 0 to always use it, -1 to never use it.
    },
}
