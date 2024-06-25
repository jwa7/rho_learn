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

from rhocalc import convert
from rholearn.loss import L2Loss
from rholearn.utils import timestamp, unpickle_dict

from dft_settings import SCF_DIR, RI_DIR, PROCESSED_DIR, REBUILD_DIR, MASK

# ===== SETUP =====
SEED = 42
FIELD_NAME = "edensity"
RI_FIT_ID = "edensity"
DATA_DIR = "/work/cosmo/abbott/june-24/ArgAu/data"
ML_DIR = os.getcwd()

# ===== DATA =====
ALL_SYSTEM = ase.io.read(join(DATA_DIR, "ArgHAu_geomopt_energies_forces.xyz"), ":")
ALL_SYSTEM_ID = np.arange(len(ALL_SYSTEM))
SYSTEM_ID = np.array(list(range(80)) + list(range(721, 801)))
np.random.default_rng(seed=SEED).shuffle(SYSTEM_ID)  # shuffle first?
SYSTEM = [ALL_SYSTEM[A] for A in SYSTEM_ID]
N_TRAIN = 128
ALL_SUBSET_ID = [
    SYSTEM_ID[:N_TRAIN],  # train subsets
    SYSTEM_ID[-50:-40],  # 10 x val
    np.array(list(range(80, 90)) + list(range(801, 811))),  # 20 x test
]
CROSSVAL = None

# ALL_SUBSET_ID = None
# CROSSVAL = {
#     "n_groups": 2,  # num groups for data split (i.e. 3 for train-test-val)
#     "group_sizes": [1, 1],  # the abs/rel group sizes for the data splits
#     "shuffle": True,  # whether to shuffle structure indices for the train/test(/val) split
# }

# ===== RELATIVE DIRS =====
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
    "job-name": "ArgAu_ML",
    "nodes": 1,
    "time": "6:00:00",
    # "mem-per-cpu": 2000,
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
TARGET_BASIS = unpickle_dict(  # Load the RI basis set definition
    join(PROCESSED_DIR(SYSTEM_ID[0]), "calc_info.pickle")
)["basis_set"]
ATOM_TYPES = [1, 6, 7, 8, 79]  # H C N O Au
for a in ATOM_TYPES:  # need a basis set definition for all global atoms
    assert convert.NUM_TO_SYM[a] in TARGET_BASIS["lmax"]
LMAX = max(TARGET_BASIS["lmax"].values())
DESCRIPTOR_HYPERS = {
    "spherical_expansion_hypers": {
        "cutoff": 10.0,  # Angstrom
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
            values=torch.tensor([[l, 1] for l in torch.arange(LMAX + 1)]),
        ),
        "skip_redundant": True,
    },
    "atom_types": ATOM_TYPES,
    "mask_descriptor": True,
}
if MASK is not None and DESCRIPTOR_HYPERS["mask_descriptor"] is True:
    DESCRIPTOR_HYPERS.update(MASK)  # i.e. `surface_depth` and `buffer_depth`


def net(
    in_keys: mts.Labels,
    in_properties: mts.Labels,
    out_properties: mts.Labels,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    """
    Builds a NN sequential ModuleMap
    """
    all_invariant_key_idxs = [
        i for i, key in enumerate(in_keys) if key["o3_lambda"] == 0
    ]
    # au_invariant_key_idxs = [
    #     i
    #     for i, key in enumerate(in_keys)
    #     if key["o3_lambda"] == 0 and key["center_type"] == 79
    # ]
    sequential = nn.Sequential(
        in_keys,
        nn.InvariantLayerNorm(  # for all invariants
            in_keys=in_keys,
            invariant_key_idxs=all_invariant_key_idxs,
            in_features=[
                len(in_props)
                for i, in_props in enumerate(in_properties)
                if i in all_invariant_key_idxs
            ],
            dtype=dtype,
            device=device,
        ),
        nn.EquivariantLinear(  # Linear layer for all blocks
            in_keys=in_keys,
            invariant_key_idxs=all_invariant_key_idxs,
            in_features=[len(in_props) for in_props in in_properties],
            # out_features=32,
            out_properties=out_properties,
            bias=True,  # equivariant bias
            dtype=dtype,
            device=device,
        ),
        # nn.InvariantTanh(  # activation only for Au invariants
        #     in_keys=in_keys, invariant_key_idxs=au_invariant_key_idxs, out_properties=out_properties,
        # ),
        # nn.EquivariantLinear(
        #     in_keys=in_keys,
        #     invariant_key_idxs=all_invariant_key_idxs,
        #     in_features=32,
        #     out_properties=out_properties,
        #     bias=True,  # equivariant bias
        #     dtype=dtype,
        #     device=device,
        # ),
    )

    # for key, block_nn in zip(in_keys, sequential.module_map):

        # # Initialize Au weights to zero.
        # if key["center_type"] == 79:
        #     for i, param in enumerate(block_nn.parameters()):
        #         param.data = torch.nn.Parameter(torch.zeros_like(param))

        # # Freeze the other weights
        # else:
        #     for param in block_nn.parameters():
        #         param.requires_grad = False

    return sequential


NET = net
PRETRAINED_MODEL = None
TRAIN = {
    # Training
    "batch_size": 4,
    "n_epochs": 10001,  # number of total epochs to run. End in a 1, i.e. 20001.
    "restart_epoch": None,  # The epoch of the last saved checkpoint, or None for no restart
    "use_overlap": False,  # True/False, or the epoch to start using it at
    # "overlap_type": "diag",
    # Saving and logging
    "checkpoint_interval": 500,  # save model and optimizer state every x intervals
    "log_interval": 500,  # how often to log the loss
    "log_block_loss": True,  # whether to log the block losses
}
OPTIMIZER = partial(torch.optim.Adam, **{"lr": 1e-1})
# SCHEDULER = None
SCHEDULER = partial(torch.optim.lr_scheduler.StepLR, **{"step_size": 1000, "gamma": 0.1})
TRAIN_LOSS_FN = L2Loss(overlap_type=None)
VAL_LOSS_FN = L2Loss(overlap_type="on-site")
EVAL = {  # Evaluation
    "eval_id": ALL_SUBSET_ID[2],
    "eval_epoch": 10000,
    "target_type": "ri",  # evaluate MAE against QM ("ref") or RI ("ri") scalar field
    "evaluate_on": "surface",  # "surface" or "slab"
}
DATA_NAMES = {
    "target": (
        "ri_coeffs.npz"
        if DESCRIPTOR_HYPERS["mask_descriptor"] is False
        else f"ri_coeffs_masked_{MASK['surface_depth']}_{MASK['buffer_depth']}.npz"
    ),
    "aux": (
        "ri_ovlp.npz"
        if DESCRIPTOR_HYPERS["mask_descriptor"] is False
        else f"ri_ovlp_masked_{MASK['surface_depth']}_{MASK['buffer_depth']}.npz"
    ),
}
if TRAIN.get("overlap_type") == "diag":
    DATA_NAMES["aux_diag"] = DATA_NAMES["aux"].replace(".npz", "_diag.npz")

# ===== FINAL DICT =====
ML_SETTINGS = {
    "SEED": SEED,
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
    "PRETRAINED_MODEL": PRETRAINED_MODEL,
    "TRAIN": TRAIN,
    "OPTIMIZER": OPTIMIZER,
    "SCHEDULER": SCHEDULER,
    "TRAIN_LOSS_FN": TRAIN_LOSS_FN,
    "VAL_LOSS_FN": VAL_LOSS_FN,
    "DATA_NAMES": DATA_NAMES,
    "EVAL": EVAL,
    "SCF_DIR": SCF_DIR,
    "RI_DIR": RI_DIR,
    "PROCESSED_DIR": PROCESSED_DIR,
    "REBUILD_DIR": REBUILD_DIR,
    "CHKPT_DIR": CHKPT_DIR,
    "EVAL_DIR": EVAL_DIR,
}
