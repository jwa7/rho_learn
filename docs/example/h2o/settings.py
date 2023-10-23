import os

import ase.io
import numpy as np
import torch


# =================================================
# ===== Define structural data from .xyz file =====
# =================================================

# Define path to top level of the data dir. Targets and descriptors will be
# saved here (or should live here)
data_dir = "/scratch/abbott/h2o_homo/data"

# The indices of all the structures in the dataset
all_idxs = np.arange(100)
frames = ase.io.read(os.path.join(data_dir, "water_monomers_1k.xyz"), ":")
frames = [frames[A] for A in all_idxs]


# ====================================================
# ===== Settings for generating learning targets =====
# ====================================================

# Path to AIMS binary
aims_path = "/home/abbott/codes/new_aims/FHIaims/build/aims.230905.scalapack.mpi.x"

# Define the AIMS settings that are common to all calculations
base_aims_kwargs = {
    "species_dir": "/home/abbott/codes/new_aims/FHIaims/species_defaults/defaults_2020/tight",
    "xc": "pbe0",
    "spin": "none",
    "relativistic": "none",
    "charge": 0,
    "sc_accuracy_rho": 1e-8,
    "wave_threshold": 1e-8,
}

# Settings specific to SCF
scf_kwargs = {
    "elsi_restart": "write 1",
    "ri_fit_write_orbital_info": True,
}

# Settings for the RI procedure
ri_kwargs = {
    # ===== To restart from a converged density matrix and force no SCF:
    "elsi_restart": "read",
    "sc_iter_limit": 0,
    "postprocess_anyway": True,
    # ===== What we want to fit to:
    "ri_fit_field_from_kso_weights": True,   # allows us to select the HOMO
    # ===== Specific setting for RI fitting
    "ri_fit_ovlp_cutoff_radius": 2.0,
    "ri_fit_assume_converged": True,
    # ===== What to write as output
    "ri_fit_write_coeffs": True,   # RI coeffs (the learning target)
    "ri_fit_write_ovlp": True,     # RI overlap (needed for loss evaluation)
    "ri_fit_write_ref_field": True,           # SCF converged scalar field on AIMS grid
    "ri_fit_write_rebuilt_field": True,       # RI-rebuilt scalar field on AIMS grid
    "output": ["cube ri_fit"],                # Allows output of cube files
    "ri_fit_write_ref_field_cube": True,      # SCF converged scalar field on CUBE grid
    "ri_fit_write_rebuilt_field_cube": True,  # RI-rebuilt scalar field on CUBE grid
}

# Settings for HPC job scheduler
sbatch_kwargs = {
    "job-name": "h2o",
    "nodes": 1,
    "time": "01:00:00",
    "mem-per-cpu": 2000,
    "partition": "bigmem",
    "ntasks-per-node": 10,
}

# ==========================================================
# ===== Settings for generating structural descriptors =====
# ==========================================================

rascal_settings =  {
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

cg_settings =  {
    "angular_cutoff": None,
    "angular_selection": np.arange(9).tolist(),
    "parity_selection": [+1],
}

# =========================================
# ===== Settings for cross validation =====
# =========================================

crossval_settings = {

    # Define the number of training subsets to use and which one to run
    # "n_train_subsets": 4,      # the number of training subsets to use (i.e. for learning exercise). 0 for no subsets.
    # "i_train_subset": 0,       # the subset number to run, from 0 to n_train_subsets - 1, inclusive
    # "subset_sizes": [10, 20, 40, 80],

    # Calculate the invariant means and standard deviation of training data
    # "calc_out_train_inv_means": False,
    # "calc_out_train_std_dev": False,

    # Settings for cross validation
    "n_groups": 3,  # num groups for data split (i.e. 3 for train-test-val)
    "group_sizes": [0.5, 0.3, 0.2],  # the abs/rel group sizes for the data splits
    "shuffle": True,  # whether to shuffle structure indices for the train/test(/val) split
    "seed": 100,  # random seed for shuffling data indices
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

# # Define ML settings
ml_settings = {

    # Set path where the simulation should be run / results saved
    "run_dir": ".",

    # Parameters for training objects
    "model": {  # Model architecture
        "model_type": "linear",  # linear or nonlinear
        "bias_invariants": False,
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
            "keep_in_mem": True,  # whether to keep the train data in memory or I/O from disk
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
