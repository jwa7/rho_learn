"""
Script for running model training
"""
from functools import partial
import time

import numpy as np
import torch

import metatensor
from metatensor import Labels, TensorMap
from metatensor.learn.data import (
    Dataset, DataLoader, IndexedDataset, group
)

from rholearn import io, data, loss, models, train
from settings import *


# =================
# ===== Setup =====
# =================

t0 = time.time()

# Create ML run dir
ml_dir = os.path.join(ML_DIR, DATA_SETTINGS["ri_restart_idx"])
if not os.path.exists(ml_dir):
    os.makedirs(ml_dir)

# Callables for the SCF, RI, and processed data directories
def scf_dir(A):
    return os.path.join(DATA_DIR, f"{A}")

def ri_dir(A, restart_idx):
    return os.path.join(scf_dir(A), f"{restart_idx}")
    
def processed_dir(A, restart_idx):
    return os.path.join(scf_dir(A), f"{restart_idx}", "processed")

log_path = os.path.join(ml_dir, "training.log")
io.log(log_path, f"# Top directory defined as: {TOP_DIR}")


# ==========================
# ===== Define Systems =====
# ==========================
io.log(log_path, "# System setup")

# Load the frames in the complete dataset
all_frames = DATA_SETTINGS["all_frames"]

# Shuffle the total set of structure indices
idxs = np.arange(len(all_frames))
np.random.default_rng(seed=DATA_SETTINGS["seed"]).shuffle(idxs)

# Take a subset of the frames if desired
idxs = idxs[:DATA_SETTINGS["n_frames"]]
frames = [all_frames[A] for A in idxs]


# ================================
# ===== Train/Test/Val Split =====
# ================================
io.log(log_path, "# train/test/val split")

all_subset_idxs = data.group_idxs(
    idxs=idxs,
    n_groups=CROSSVAL_SETTINGS["n_groups"],
    group_sizes=CROSSVAL_SETTINGS["group_sizes"],
    shuffle=CROSSVAL_SETTINGS["shuffle"],
    seed=DATA_SETTINGS["seed"],
)
train_idxs, test_idxs, val_idxs = all_subset_idxs[0], [], []

io.log(
    log_path,
    f"# num train_idxs: {len(train_idxs)}"
    f"   num test_idxs: {len(test_idxs)}"
    f"   num val_idxs: {len(val_idxs)}",
)
np.savez(
    os.path.join(ml_dir, "idxs.npz"),
    idxs=idxs,
    train_idxs=train_idxs,
    test_idxs=test_idxs,
    val_idxs=val_idxs,
)


# ====================
# ===== Datasets =====
# ====================
io.log(log_path, "# Build datasets")

def load_to_torch(
    path: str, torch_kwargs: dict, drop_blocks: bool = False
) -> TensorMap:
    """Loads a TensorMap from file and converts its backend to torch"""
    tensor = metatensor.io.load_custom_array(
        path,
        create_array=metatensor.io.create_torch_array,
    )
    if drop_blocks:
        tensor = metatensor.drop_blocks(
            tensor, 
            keys=Labels(
                names=["spherical_harmonics_l", "species_center"],
                values=np.array([[5, 1]]),
            ),
        )
    tensor = tensor.to(**torch_kwargs)
    tensor = metatensor.requires_grad(tensor, True)
    return tensor

datasets = [
    IndexedDataset(
        sample_ids=subset_idxs,
        frames=[all_frames[A] for A in subset_idxs],
        descriptors=[
            load_to_torch(
                os.path.join(
                    processed_dir(A, DATA_SETTINGS["ri_restart_idx"]), 
                    "lsoap.npz"
                ), 
                TORCH_SETTINGS, 
                # drop_blocks=True,
            )
            for A in subset_idxs
        ],
        targets=[
            load_to_torch(
                os.path.join(
                    processed_dir(A, DATA_SETTINGS["ri_restart_idx"]), 
                    "ri_coeffs.npz"
                ), 
                TORCH_SETTINGS
            )
            for A in subset_idxs
        ],
        auxiliaries=[
            load_to_torch(
                os.path.join(
                    processed_dir(A, DATA_SETTINGS["ri_restart_idx"]), 
                    "ri_ovlp.npz"
                ), 
                TORCH_SETTINGS
            )
            for A in subset_idxs
        ],
    ) for subset_idxs in all_subset_idxs
]

train_dataset, test_dataset, val_dataset = datasets[0], None, None


# ===============================
# ===== Baseline and stddev =====
# ===============================
io.log(log_path, "# Calculate invariant baseline and stddev")

if ML_SETTINGS["model"]["use_invariant_baseline"]:
    invariant_baseline = data.get_dataset_invariant_means(
        train_dataset, field="targets", torch_kwargs=TORCH_SETTINGS,
    )
else:
    invariant_baseline = None

stddev = data.get_standard_deviation(
    dataset=train_dataset,
    field="targets",
    torch_kwargs=TORCH_SETTINGS,
    invariant_baseline=invariant_baseline,
    overlap_field="auxiliaries",
)
np.savez(os.path.join(ml_dir, "stddev.npz"), stddev=stddev.detach().numpy())


# =======================
# ===== DataLoaders =====
# =======================
io.log(log_path, "# Construct dataloaders")

# Construct dataloaders
train_loader = metatensor.learn.data.DataLoader(
    train_dataset,
    collate_fn=group,
    batch_size=ML_SETTINGS["loading"]["batch_size"],
    **ML_SETTINGS["loading"]["args"],
)
# val_loader = metatensor.learn.data.DataLoader(
#     val_dataset,
#     collate_fn=group,
#     batch_size=ML_SETTINGS["loading"]["batch_size"],
#     **ML_SETTINGS["loading"]["args"],
# )

# test_loader = metatensor.learn.data.DataLoader(
#     test_dataset,
#     collate_fn=group,
#     batch_size=ML_SETTINGS["loading"]["batch_size"],
#     **ML_SETTINGS["loading"]["args"],
# )
val_loader = None
test_loader = None


# =================
# ===== Model =====
# =================
io.log(log_path, "# Init model")

descriptor_kwargs = {
    "rascal_settings": RASCAL_SETTINGS,
    "cg_settings": CG_SETTINGS,
}

basis_set = io.unpickle_dict(
    os.path.join(
        processed_dir(train_idxs[0], DATA_SETTINGS["ri_restart_idx"]), 
        "calc_info.pickle"
    )
)["basis_set"]

target_kwargs = {
    "aims_kwargs": {**BASE_AIMS_KWARGS},
    "aims_path": AIMS_PATH,
    "basis_set": {**basis_set},
    "sbatch_kwargs": SBATCH_KWARGS,
}
target_kwargs["aims_kwargs"].update(REBUILD_KWARGS)

model = models.RhoModel(
    # Standard model architecture
    model_type=ML_SETTINGS["model"]["model_type"],  # "linear" or "nonlinear"
    input=train_dataset[0].descriptors,   # example input data for init metadata
    output=train_dataset[0].targets,  # example output data for init metadata
    bias_invariants=ML_SETTINGS["model"]["bias_invariants"],

    # Nonlinear model settings
    hidden_layer_widths=ML_SETTINGS["model"].get("hidden_layer_widths"),
    activation_fn=ML_SETTINGS["model"].get("activation_fn"),
    bias_nn=ML_SETTINGS["model"].get("bias_nn"),

    # Invariant baselining
    invariant_baseline=invariant_baseline,

    # Settings for descriptor/target building
    descriptor_kwargs=descriptor_kwargs,
    target_kwargs=target_kwargs,

    # Torch tensor settings
    **TORCH_SETTINGS,
)


# ======================================
# ===== Loss, Optimizer, Scheduler =====
# ======================================
io.log(log_path, "# Init loss, opt, scheduler")

loss_fn = loss.L2Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = None


# =========================
# ===== Training Loop =====
# =========================
io.log(log_path, "# Training loop")

train.training_loop(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    scheduler=scheduler,
    ri_dir=partial(
        ri_dir, restart_idx=DATA_SETTINGS["ri_restart_idx"]
    ),
    ml_dir=ml_dir,
    ml_settings=ML_SETTINGS,
    log_path=log_path,
)


# ==================
# ===== Finish =====
# ==================

training_time = time.time() - t0
if training_time <= 60:
    io.log(log_path, f"# Training complete in {training_time} seconds.")
elif 60 < training_time <= 3600:
    io.log(log_path, f"# Training complete in {training_time / 60} minutes.")
else:
    io.log(log_path, f"# Training complete in {training_time / 3600} hours.")