import os
import torch

# Define the rascaline hypers for generating lambda-SOAP features
rascal_hypers = {
    "cutoff": 3.0,  # Angstrom
    "max_radial": 6,  # Exclusive
    "max_angular": 5,  # Inclusive
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}
# Set the rholearn absolute path and top level data directory
rholearn_dir = "/Users/joe.abbott/Documents/phd/code/rho/rho_learn/"
data_dir = os.path.join(rholearn_dir, "docs/example/water/data")

torch_settings = {
    "requires_grad": True,  # needed to track gradients
    "dtype": torch.float64,  # recommended
    "device": torch.device("cpu"),  # which device to load tensors to
}

# Define data settings
data_settings = {

    # Set path where various data is stored
    "data_dir": data_dir,
    "input_dir": os.path.join(data_dir, "lsoap"),
    "output_dir": os.path.join(data_dir, "rho_std"),
    "overlap_dir": os.path.join(data_dir, "rho"),
    "out_inv_means_path": os.path.join(data_dir, "rho_std/inv_means.npz"),

    # Other settings for splitting data
    "axis": "samples",  # which axis to split the data along
    "names": ["structure"],  # what index to split the data by - i.e. "structure"
    "n_groups": 3,  # num groups for data split (i.e. 3 for train-test-val)
    "group_sizes": [0.5, 0.4, 0.1],  # the abs/rel group sizes for the data splits
    "n_exercises": 2,  # the number of learning exercises to perform
    "n_subsets": 3,  # how many subsets to use for each exercise
    "shuffle": True,  # whether to shuffle structure indices for the train/test split
    "seed": 10,  # random seed for data split

    "n_train_subsets": 5,      # the number of training subsets to use (i.e. for learning exercise)
    "i_train_subset": 0,       # the subset number to run, from 0 to n_train_subsets - 1, inclusive
}

# Define ML settings
ml_settings = {

    # Set path where the simulation should be run
    "run_dir": os.path.join(rholearn_dir, "docs/example/water/runs/demo_linear"),

    # Parameters for training objects
    "model": {  # Model architecture
        "type": "linear",  # linear or nonlinear
        "args": {  # if using linear, pass an empty dict
            # "hidden_layer_widths": [10, 10, 10],
            # "activation_fn": "SiLU",
        },
    },
    "optimizer": {
        "algorithm": torch.optim.AdamW,
        "args": {
            "lr": 0.1,
        },
    },
    "scheduler": {
        # "use_scheduler": False,
        "algorithm": torch.optim.lr_scheduler.MultiStepLR,
        "args": {
            "milestones": [50, 100, 150, 200],
            "gamma": 0.5,
            # "last_epoch": -1
        },
    },
    "loading": {
        "batch_size": 50,  # number of samples per batch
        # "num_workers": 0,  # number of workers for data loading
        # "prefetch_factor": None,  # number of batches to prefetch
    },

    # Parameters for training procedure
    "training": {
        "n_epochs": 300,  # number of total epochs to run
        "save_interval": 10,  # save model and optimizer state every x intervals
        "restart_epoch": 0,  # The epoch checkpoint number if restarting, or 0
        "standardize_out_invariants": True,
        "learn_on_rho_after": 50,  # epoch to start learning on rho instead of coeffs
    },
}
