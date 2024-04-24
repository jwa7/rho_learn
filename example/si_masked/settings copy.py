from functools import partial
from os.path import join, exists

import ase.io
import torch
import metatensor.torch as mts


# ===== SETUP =====
SEED = 42
TOP_DIR = "/work/cosmo/abbott/april-24/si_masked"
DATA_DIR = join(TOP_DIR, "data")
FIELD_NAME = "edensity"
RI_FIT_ID = "edensity"
ML_RUN_ID = "unmasked"
ML_DIR = join(TOP_DIR, "ml", RI_FIT_ID, ML_RUN_ID)


# ===== DATA =====
ALL_STRUCTURE = ase.io.read(join(DATA_DIR, "si_slabs_dimer_z_depth_geomopt_distorted.xyz"), ":")
ALL_STRUCTURE_ID = np.arange(len(ALL_FRAMES))
# np.random.default_rng(seed=SEED).shuffle(ALL_STRUCTURE_ID)  # shuffle first?
STRUCTURE_ID = ALL_STRUCTURE_ID[6::13]
STRUCTURE = [ALL_STRUCTURE[A] for A in STRUCTURE_ID]


# ===== HPC
SBATCH = {
    "job-name": "si_mask",
    "nodes": 1,
    "time": "12:00:00",
    "mem-per-cpu": 0,
    "partition": "standard",
    "ntasks-per-node": 72,
}
HPC = {
    "load_modules": ["intel", "intel-oneapi-mpi", "intel-oneapi-mkl"],
    "export_vars": [
        "OMP_NUM_THREADS=1",
        "MKL_DYNAMIC=FALSE",
        "MKL_NUM_THREADS=1",
        "I_MPI_FABRICS=shm",
    ],
    "run_command": "srun",
}


# ===== MODEL & TRAINING =====
DTYPE = torch.float64
DEVICE = "cpu"
TORCH = {
    "dtype": DTYPE,  # important for model accuracy
    "device": torch.device(type=DEVICE),
}
DESCRIPTOR_HYPERS = {
    "spherical_expansion_hypers": {
        "cutoff": 4.0,  # Angstrom
        "max_radial": 10,  # Exclusive
        "max_angular": 3,  # Inclusive
        "atomic_gaussian_width": 0.3,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
        "center_atom_weight": 1.0,
    },
    "density_correlations_hypers": {
        "max_angular": 10,
        "correlation_order": 2,
        "angular_cutoff": 3,
        "selected_keys": mts.Labels(
            names=["spherical_harmonics_l", "inversion_sigma"],
            values=torch.tensor(
                [[l, 1] for l in np.arange(RI_KWARGS["default_max_l_prodbas"] + 1)]
            ),
        ),
        "skip_redundant": True,
    }
}
# TODO: move partially initialized architecture to here once updated
# MODEL_HYPERS = {
#     "nn": [
#         partial(nn.EquiLayerNorm, ,
#         nn.EquiLinear,
#     ]
# }

CROSSVAL = {
    # Settings for cross validation
    "n_groups": 3,  # num groups for data split (i.e. 3 for train-test-val)
    "group_sizes": [len(STRUCTURE_ID), 0, 0],  # the abs/rel group sizes for the data splits
    "shuffle": True,  # whether to shuffle structure indices for the train/test(/val) split
}
TRAIN = {
    # Training
    "batch_size": 1,
    "n_epochs": 3001,  # number of total epochs to run. End in a 1, i.e. 20001.
    "restart_epoch": 2000,  # The epoch of the last saved checkpoint, or None for no restart
    "use_overlap": 0,  # True/False, or the epoch to start using it at
    # Evaluation
    "eval_interval": None,  # validate every x epochs against real-space SCF field. None to turn off evaluation
    "eval_subset": "train",
    "calculate_error": False,  # whether to wait for AIMS calcs to finish
    "target_type": "ri",  # evaluate against QM ("ref") or RI ("ri") scalar field
    # Saving and logging
    "checkpoint_interval": 250,  # save model and optimizer state every x intervals
    "log_interval": 250,  # how often to log the loss
    "log_block_loss": True,  # whether to log the block losses
    # Masked learning
    "masked_learning": True,
    "slab_depth": 4.0,  # Ang
    "interphase_depth": 1.0,  # Ang
}
OPTIMIZER = partial(
    torch.optim.Adam, 
    **{
        "lr": 1e-3,
    }
)
# SCHEDULER = None
SCHEDULER = partial(
    torch.optim.lr_scheduler.MultiStepLR,
    **{
        "milestones": [2250, 2500, 2750],
        "gamma": 0.1,
    }
)


# ===== Define and create relative directories
SCF_DIR = lambda A: join(DATA_DIR, f"{A}")
RI_DIR = lambda A: join(SCF_DIR(A), RI_FIT_ID)
REBUILD_DIR = lambda A: join(RI_DIR(A), "rebuild")
PROCESSED_DIR = lambda A: join(RI_DIR(A), "processed")
CHKPT_DIR = lambda epoch: join(ML_DIR, "checkpoint", f"epoch_{epoch}")
EVAL_DIR = lambda A, epoch: join(ML_DIR, "evaluation", f"epoch_{epoch}", f"{A}")
# Create dirs
if not exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not exists(ML_DIR):
    os.makedirs(ML_DIR)
if not exists(join(ML_DIR, "checkpoint")):
    os.makedirs(join(ML_DIR, "checkpoint"))
if not exists(join(ML_DIR, "evaluation")):
    os.makedirs(join(ML_DIR, "evaluation"))