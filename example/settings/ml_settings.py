"""
Module containing global variables for running model training.
"""

from functools import partial
import os
from os.path import join, exists
from typing import List

import ase.io
import numpy as np
import torch
import metatensor.torch as mts

from metatensor.torch.learn import nn

from rholearn.loss import L2Loss
from rholearn.utils import timestamp, unpickle_dict

# ===== SETUP =====
SEED = 42
FIELD_NAME = "edensity"
RI_FIT_ID = "edensity"
DATA_DIR = "/work/cosmo/abbott/april-24/si_masked/data"
ML_DIR = "/home/abbott/may-24/si_clean_distortions/run_1/masked/2"

# ===== DATA =====
ALL_SYSTEM = ase.io.read(
    join(DATA_DIR, "si_slabs_dimer_z_depth_geomopt_distorted.xyz"), ":"
)
ALL_SYSTEM_ID = np.arange(len(ALL_SYSTEM))
# np.random.default_rng(seed=SEED).shuffle(ALL_SYSTEM_ID)  # shuffle first?
SYSTEM_ID = ALL_SYSTEM_ID[6::13]
SYSTEM = [ALL_SYSTEM[A] for A in SYSTEM_ID]

ALL_SUBSET_ID = [
    SYSTEM_ID[:8],  # train subsets: [2, 4, 8, 16]
    SYSTEM_ID[-5:-3],  # 2 x val
    SYSTEM_ID[-3:],  # 3 x test
]
CROSSVAL = None

# ALL_SUBSET_ID = None
# CROSSVAL = {
#     "n_groups": 2,  # num groups for data split (i.e. 3 for train-test-val)
#     "group_sizes": [1, 1],  # the abs/rel group sizes for the data splits
#     "shuffle": True,  # whether to shuffle structure indices for the train/test(/val) split
# }

# ===== RELATIVE DIRS =====
SCF_DIR = lambda A: join(DATA_DIR, f"{A}")
RI_DIR = lambda A: join(SCF_DIR(A), RI_FIT_ID)
REBUILD_DIR = lambda A: join(RI_DIR(A), "rebuild")
PROCESSED_DIR = lambda A: join(RI_DIR(A), "processed")
CHKPT_DIR = lambda epoch: join(ML_DIR, "checkpoint", f"epoch_{epoch}")
EVAL_DIR = lambda A, epoch: join(ML_DIR, "evaluation", f"epoch_{epoch}", f"{A}")

# ===== CREATE DIRS =====
if not exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not exists(ML_DIR):
    os.makedirs(ML_DIR)
if not exists(join(ML_DIR, "checkpoint")):
    os.makedirs(join(ML_DIR, "checkpoint"))
if not exists(join(ML_DIR, "evaluation")):
    os.makedirs(join(ML_DIR, "evaluation"))


# ===== HPC =====
SBATCH = {
    "job-name": "si_mask",
    "nodes": 1,
    "time": "1:00:00",
    "mem-per-cpu": 2000,
    "partition": "standard",
    "ntasks-per-node": 20,
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
TARGET_BASIS = unpickle_dict(  # Load the RI basis set definition
    join(PROCESSED_DIR(SYSTEM_ID[0]), "calc_info.pickle")
)["basis_set"]
LMAX = max(TARGET_BASIS["lmax"].values())
DESCRIPTOR_HYPERS = {
    "spherical_expansion_hypers": {
        "cutoff": 6.0,  # Angstrom
        "max_radial": 10,  # Exclusive
        "max_angular": LMAX,  # Inclusive
        "atomic_gaussian_width": 0.3,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
        "center_atom_weight": 1.0,
    },
    "density_correlations_hypers": {
        "max_angular": LMAX * 2,
        "correlation_order": 2,
        "angular_cutoff": None,
        "selected_keys": mts.Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[l, 1] for l in np.arange(LMAX + 1)]),
        ),
        "skip_redundant": True,
    },
    "atom_types": [1, 14],
    "mask_descriptor": False,
    "slab_depth": 4.0,  # Ang
    "interphase_depth": 1.0,  # Ang
}


def net(
    in_keys: mts.Labels,
    invariant_key_idxs: List[int],
    in_properties: mts.Labels,
    out_properties: mts.Labels,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    """Builds a NN sequential ModuleMap"""
    return nn.Sequential(
        in_keys,
        nn.InvariantLayerNorm(
            in_keys=in_keys,
            invariant_key_idxs=invariant_key_idxs,
            in_features=[
                len(in_props)
                for i, in_props in enumerate(in_properties)
                if i in invariant_key_idxs
            ],
            dtype=dtype,
            device=device,
        ),
        nn.EquivariantLinear(
            in_keys=in_keys,
            invariant_key_idxs=invariant_key_idxs,
            in_features=[len(in_props) for in_props in in_properties],
            out_features=128,
            bias=True,  # equivariant bias
            dtype=dtype,
            device=device,
        ),
        nn.InvariantSiLU(in_keys=in_keys, invariant_key_idxs=invariant_key_idxs),
        nn.EquivariantLinear(
            in_keys=in_keys,
            invariant_key_idxs=invariant_key_idxs,
            in_features=128,
            out_properties=out_properties,
            bias=True,  # equivariant bias
            dtype=dtype,
            device=device,
        ),
    )


NET = net
TRAIN = {
    # Training
    "batch_size": 1,
    "n_epochs": 2001,  # number of total epochs to run. End in a 1, i.e. 20001.
    "restart_epoch": None,  # The epoch of the last saved checkpoint, or None for no restart
    "use_overlap": False,  # True/False, or the epoch to start using it at
    # Saving and logging
    "checkpoint_interval": 500,  # save model and optimizer state every x intervals
    "log_interval": 500,  # how often to log the loss
    "log_block_loss": True,  # whether to log the block losses
}
OPTIMIZER = partial(torch.optim.Adam, **{"lr": 1e-2})
# SCHEDULER = None
SCHEDULER = partial(
    torch.optim.lr_scheduler.MultiStepLR,
    **{
        "milestones": [500, 1000, 1500],
        "gamma": 0.1,
    },
)
LOSS_FN = L2Loss
EVAL = {  # Evaluation
    "eval_id": ALL_SUBSET_ID[2],
    "eval_epoch": 2000,
    "target_type": "ri",  # evaluate MAE against QM ("ref") or RI ("ri") scalar field
    "evaluate_on": "surface",  # "surface" or "slab"
    "stm": {
        "center_coord": 0,  # Angstrom, offset from max Z coord
        "thickness": 4.0,  # Angstrom
    },
}


# ===== FINAL DICT =====
ML_SETTINGS = {
    "SEED": SEED,
    "TOP_DIR": TOP_DIR,
    "DATA_DIR": DATA_DIR,
    "FIELD_NAME": FIELD_NAME,
    "RI_FIT_ID": RI_FIT_ID,
    "ML_DIR": ML_DIR,
    "ALL_SYSTEM": ALL_SYSTEM,
    "ALL_SYSTEM_ID": ALL_SYSTEM_ID,
    "SYSTEM_ID": SYSTEM_ID,
    "SYSTEM": SYSTEM,
    "ALL_SUBSET_ID": ALL_SUBSET_ID,
    "CROSSVAL": CROSSVAL,
    "SBATCH": SBATCH,
    "HPC": HPC,
    "DTYPE": DTYPE,
    "DEVICE": DEVICE,
    "TORCH": TORCH,
    "TARGET_BASIS": TARGET_BASIS,
    "DESCRIPTOR_HYPERS": DESCRIPTOR_HYPERS,
    "NET": NET,
    "TRAIN": TRAIN,
    "OPTIMIZER": OPTIMIZER,
    "SCHEDULER": SCHEDULER,
    "LOSS_FN": LOSS_FN,
    "SCF_DIR": SCF_DIR,
    "RI_DIR": RI_DIR,
    "REBUILD_DIR": REBUILD_DIR,
    "PROCESSED_DIR": PROCESSED_DIR,
    "CHKPT_DIR": CHKPT_DIR,
    "EVAL_DIR": EVAL_DIR,
    "EVAL": EVAL,
}
