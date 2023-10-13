import os
import numpy as np
import torch

# Define the rascaline hypers for generating lambda-SOAP features
lsoap_settings = {
    "rascal_hypers": {
        "cutoff": 4.0,  # Angstrom
        "max_radial": 10,  # Exclusive
        "max_angular": 5,  # Inclusive
        "atomic_gaussian_width": 0.3,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
        "center_atom_weight": 1.0,
    },
    "lambdas": [0, 1, 2, 3, 4, 5],
    "sigmas": [+1],
    "lambda_cut": None,
}


# Define PyTorch settings
torch_settings = {
    "tensor": {
        "requires_grad": True,  # needed to track gradients
        "dtype": torch.float64,  # recommended
        "device": torch.device("cpu"),  # which device to load tensors to
    },
    "use_ipex": True,  # whether to use Intel PyTorch Extension (IPEX)
}

# Define data settings
data_dir = os.path.join("/scratch/abbott/rho/silicon_salted/ml_data2/")
all_idxs = np.arange(100)
data_settings = {

    # Pass the indices of the complete dataset
    "all_idxs": all_idxs,

    # Define the number of training subsets to use and which one to run
    "n_train_subsets": 4,      # the number of training subsets to use (i.e. for learning exercise). 0 for no subsets.
    # "i_train_subset": 0,       # the subset number to run, from 0 to n_train_subsets - 1, inclusive
    "subset_sizes": [10, 20, 40, 80],

    # Set paths and names of the input/output/overlap data
    "input_dir": os.path.join(data_dir),
    "output_dir": os.path.join(data_dir),
    "overlap_dir": os.path.join(data_dir),
    "filenames": ["lsoap", "ri_coeffs", "ri_ovlp"],

    # Calculate the invariant means and standard deviation of training data
    "calc_out_train_inv_means": True,
    "calc_out_train_std_dev": True,

    # Settings for shuffling and grouping data indices
    "n_groups": 2,  # num groups for data split (i.e. 3 for train-test-val)
    "group_sizes": [80, 20],  # the abs/rel group sizes for the data splits
    "shuffle": True,  # whether to shuffle structure indices for the train/test(/val) split
    "seed": 100,  # random seed for shuffling data indices
    
}

# Define ML settings
ml_settings = {

    # Set path where the simulation should be run / results saved
    "run_dir": ".",

    # Parameters for training objects
    "model": {  # Model architecture
        "model_type": "nonlinear",  # linear or nonlinear
        "bias_invariants": True,
        "train_on_baselined_coeffs": True,
        "args": {  # if using linear, pass an empty dict
            "hidden_layer_widths": [64, 64, 64],
            "activation_fn": torch.nn.SiLU(),
        },
    },
    "optimizer": {
        "reinitialize": True,
        "algorithm": torch.optim.Adam,
        "args": {
            "lr": 0.0001,
        },
        "clip_grad_norm": True,
        "max_norm": 0.2,
    },
    "scheduler": {
        "reinitialize": True,
        "algorithm": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "args": {
            "factor": 0.75,
            "patience": 10,
        },
    },
    "loading": {
        "train": {
            "do_batching": True,  # whether to batch the train data
            "batch_size": 10,  # number of samples per train batch
            "args": {
                # "num_workers": 0,  # number of workers for data loading
                # "prefetch_factor": None,  # number of batches to prefetch
            },
        },
        "test": {
            "do_batching": False,  # whether to batch the test data
            "batch_size": 20,  # number of samples per batch
            "args": {
                # "num_workers": 0,  # number of workers for data loading
                # "prefetch_factor": None,  # number of batches to prefetch
            },
        },
    },

    # Parameters for training procedure
    "training": {
        "n_epochs": 5000,  # number of total epochs to run
        "save_interval": 1,  # save model and optimizer state every x intervals
        # "restart_epoch": 0,  # The epoch checkpoint number if restarting, or 0 for no restart
        "learn_on_rho_at_epoch": 0,  # epoch to start learning on rho instead of coeffs, or 0 to always use it, -1 to never use it.
    },
}
