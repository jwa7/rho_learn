import os
import torch

# Set the rholearn absolute path
RHOLEARN_DIR = "/Users/joe.abbott/Documents/phd/code/rho/rho_learn/"

# Define the rascaline hypers for generating the lambda-SOAP features
RASCAL_HYPERS = {
    "cutoff": 3.0,  # Angstrom
    "max_radial": 6,  # Exclusive
    "max_angular": 5,  # Inclusive
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}

DATA_SETTINGS = {
    # Set path where the data should be stored
    "data_dir": os.path.join(RHOLEARN_DIR, "docs/example/water/data"),
    "axis": "samples",         # which axis to split the data along
    "names": ["structure"],    # what index to split the data by - i.e. "structure"
    "n_groups": 3,             # num groups for data split (i.e. 3 for train-test-val)
    "group_sizes": [0.5, 0.4, 0.1],  # the abs/rel group sizes for the data splits
    "n_exercises": 2,  # the number of learning exercises to perform
    "n_subsets": 3,    # how many subsets to use for each exercise
    "shuffle": True,   # whether to shuffle structure indices for the train/test split
    "seed": 10,        # random seed for data split
}

# Define ML settings
ML_SETTINGS = {
    # Set path where the simulation should be run
    "run_dir": os.path.join(RHOLEARN_DIR, "docs/example/water/runs/demo_linear"),
    "torch": {
        "requires_grad": True,  # needed to track gradients
        "dtype": torch.float64,  # recommended
        "device": torch.device("cpu"),  # which device to load tensors to
    },
    "model": {  # Model architecture
        "type": "nonlinear",  # linear or nonlinear
        "args": {  # if using linear, pass an empty dict
            "hidden_layer_widths": [10, 10, 10],
            "activation_fn": "SiLU"
        },
    },
    "optimizer": {
        "algorithm": torch.optim.AdamW,
        "args": {
            "lr": 0.1,
        },
    },
    "loss": {
        "fn": "MSELoss",  # CoulombLoss or MSELoss
        "args": {
            "reduction": "sum",  # reduction can be used with MSELoss
        },
    },
    "loading": {
        "batch_size": 50,  # number of samples per batch
        # "num_workers": 0,  # number of workers for data loading
        # "prefetch_factor": None,  # number of batches to prefetch
    },
    "training": {
        "n_epochs": 10,  # number of total epochs to run
        "save_interval": 10,  # save model and optimizer state every x intervals
        "restart_epoch": None,  # None, or the epoch checkpoint number if restarting
        "standardize_invariant_features": True,
    },
}
