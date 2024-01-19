import os

import ase.io
import numpy as np
import torch

from rholearn import loss


# ====================================================
# ===== Define data settings, including xyz data =====
# ====================================================

# Define the top level dir
TOP_DIR = "/home/abbott/rho/rho_learn/docs/example/field"

# Where the generated data should be written
DATA_DIR = os.path.join(TOP_DIR, "data")

# Where ML outputs should be written
ML_DIR = os.path.join(TOP_DIR, "ml")

DATA_SETTINGS = {

    # Read in all frames in complete dataset
    "all_frames": ase.io.read(os.path.join(DATA_DIR, "water_monomers_1k.xyz"), ":"),
    
    # Define a subset of frames
    "n_frames": 10,

    # Define the index of the RI calculation
    "ri_restart_idx": 0,

    # Global speices required
    "global_species": [1, 8],

    # Define a random seed
    "seed": 314,

    # Calculate the standard deviation?
    # i.e. to calc it for output training data:
    "calc_out_train_std_dev": ("output", "train"),
}


# ====================================================
# ===== Settings for generating learning targets =====
# ====================================================

# Path to AIMS binary
AIMS_PATH = "/home/abbott/codes/new_aims/FHIaims/build/aims.230905.scalapack.mpi.x"

# Define the AIMS settings that are common to all calculations
BASE_AIMS_KWARGS = {
    "species_dir": "/home/abbott/rho/rho_learn/rhocalc/aims/aims_species/tight/default",
    "xc": "pbe0",
    "spin": "none",
    "relativistic": "none",
    "charge": 0,
    "sc_accuracy_rho": 1e-8,
    "wave_threshold": 1e-8,
}

# Settings specific to SCF
SCF_KWARGS = {
    "elsi_restart": "write 1",
    "ri_fit_write_orbital_info": True,
}

# Settings for the RI procedure
RI_KWARGS = {
    # ===== To restart from a converged density matrix and force no SCF:
    "elsi_restart": "read",
    "sc_iter_limit": 0,
    "postprocess_anyway": True,
    # ===== What we want to fit to:
    "ri_fit_field_from_kso_weights": True,  # allows us to select the HOMO
    # ===== Specific setting for RI fitting
    "ri_fit_ovlp_cutoff_radius": 2.0,
    "ri_fit_assume_converged": True,
    "default_max_l_prodbas": 5,
    # ===== What to write as output
    "ri_fit_write_coeffs": True,  # RI coeffs (the learning target)
    "ri_fit_write_ovlp": True,  # RI overlap (needed for loss evaluation)
    "ri_fit_write_ref_field": True,  # SCF converged scalar field on AIMS grid
    "ri_fit_write_rebuilt_field": True,  # RI-rebuilt scalar field on AIMS grid
    "output": ["cube ri_fit"],  # Allows output of cube files
    "ri_fit_write_ref_field_cube": True,  # SCF converged scalar field on CUBE grid
    "ri_fit_write_rebuilt_field_cube": True,  # RI-rebuilt scalar field on CUBE grid
}

# Settings for the RI rebuild procedure
REBUILD_KWARGS = {
    # ===== Force no SCF
    "sc_iter_limit": 0,
    "postprocess_anyway": True,
    "ri_fit_assume_converged": True,
    # ===== What we want to do
    "ri_fit_rebuild_from_coeffs": True,
    # ===== Specific settings for RI rebuild
    "ri_fit_ovlp_cutoff_radius": RI_KWARGS["ri_fit_ovlp_cutoff_radius"],
    "default_max_l_prodbas": RI_KWARGS["default_max_l_prodbas"],
    # ===== What to write as output
    "ri_fit_write_rebuilt_field": True,
    "ri_fit_write_rebuilt_field_cube": True,
    "output": ["cube ri_fit"],  # needed for cube files
}

# Settings for HPC job scheduler
SBATCH_KWARGS = {
    "job-name": "h2o",
    "nodes": 1,
    "time": "01:00:00",
    "mem-per-cpu": 2000,
    "partition": "bigmem",
    "ntasks-per-node": 10,
}

# ==============================
# ===== Settings for torch =====
# ==============================

# Setting for torch backend
TORCH_SETTINGS = {
    "dtype": torch.float64,  # important for model accuracy
    "requires_grad": True,
    "device": torch.device(type="cpu"),
}

# ==========================================================
# ===== Settings for generating structural descriptors =====
# ==========================================================

RASCAL_SETTINGS = {
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

CG_SETTINGS = {
    "correlation_order": 2,
    "angular_cutoff": None,
    "selected_keys": [[l, 1] for l in np.arange(RI_KWARGS["default_max_l_prodbas"] + 1)],
    "skip_redundant": True,
}

# =========================================
# ===== Settings for cross validation =====
# =========================================

CROSSVAL_SETTINGS = {
    # Settings for cross validation
    "n_groups": 3,  # num groups for data split (i.e. 3 for train-test-val)
    "group_sizes": [0.7, 0.2, 0.1],  # the abs/rel group sizes for the data splits
    "shuffle": True,  # whether to shuffle structure indices for the train/test(/val) split
}


# =======================================
# ===== Settings for model training =====
# =======================================

# Define ML settings
ML_SETTINGS = {

    "model": {

        # Model architecture
        "model_type": "nonlinear",  # linear or nonlinear
        "bias_invariants": False,
        "use_invariant_baseline": True,
        # Only if model_type == "nonlinear":
        "hidden_layer_widths": [64, 64, 64],
        "activation_fn": torch.nn.SiLU(),
        "bias_nn": True,
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
        "batch_size": 5,  # number of samples per train batch
        "args": {
            # "num_workers": 0,  # number of workers for data loading
            # "prefetch_factor": None,  # number of batches to prefetch
        },
        "keep_in_mem": True,  # whether to keep the train data in memory or I/O from disk
    },
    # Parameters for training procedure
    "training": {
        "n_epochs": 5,  # number of total epochs to run
        "save_interval": 5,  # save model and optimizer state every x intervals
        "restart_epoch": None,  # The epoch of the last saved checkpoint, or None for no restart
        # "learn_on_rho_at_epoch": 0,  # epoch to start learning on rho instead of coeffs, or 0 to always use it, -1 to never use it.
    },
    "calc_mae": {
        "interval": 2,  # validate every x epochs against real-space SCF field
        "which": ["test"],
        "return_targets": False,  # whether to wait for AIMS calcs to finish
    },
}
