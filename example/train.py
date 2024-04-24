from os.path import exists, join

import torch

import metatensor.torch as mts
from metatensor.torch.learn.data import Dataloader, IndexedDataset, group

from model import DescriptorCalculator, Model
from train_utils import *
from settings import *


def train():

    # ===== Setup =====
    t0_training = time.time()
    torch.manual_seed(SEED)
    torch.set_default_dtype(DTYPE)

    log_path = os.path.join(ML_DIR, "training.log")
    io.log(log_path, f"Setup")
    io.log(log_path, f"Top directory defined as: {TOP_DIR}")

    # ===== Create datasets and dataloaders =====
    io.log(log_path, f"Calculate descriptors")
    all_subset_id = data.group_idxs(  # cross-validation split of idxs
        sample_id=IDXS,
        n_groups=CROSSVAL_SETTINGS["n_groups"],
        group_sizes=CROSSVAL_SETTINGS["group_sizes"],
        shuffle=CROSSVAL_SETTINGS["shuffle"],
        seed=SEED,
    )
    io.log(log_path, f"Init descriptor calculator")
    descriptor_calculator = DescriptorCalculator(**DESCRIPTOR_HYPERS)
    datasets = []
    for subset_id in all_subset_id:  # build each cross-val dataset
        io.log(log_path, f"Build dataset for structures {subset_id}")
        selected_samples = get_selected_samples(  # for structure and/or atomic subsets
            structure=[ALL_STRUCTURE[A] for A in subset_id],
            structure_id=subset_id,
            masked_learning=TRAIN["masked_learning"],
            slab_depth=TRAIN.get("slab_depth"),
            interphase_depth=TRAIN.get("interphase_depth"),
        )
        io.log(log_path, f"Calculate descriptors for subset {subset_id}")
        descriptor = calculate_descriptors(  # pre-compute descriptors
            descriptor_calculator=descriptor_calculator,
            all_structure=ALL_STRUCTURE,
            all_structure_id=ALL_STRUCTURE_ID,
            structure_id=subset_id,
            correlate_what="pxp",
            selected_samples=selected_samples,
        )
        io.log(log_path, "Load targets and auxiliaries")
        target = [  # load RI coeffs
            load_dft_data(join(PROCESSED_DIR(A), "ri_coeffs.npz"))
            for A in subset_id
        ]
        aux = [  # load overlaps
            load_dft_data(join(PROCESSED_DIR(A), "ri_ovlp.npz"))
            for A in subset_id
        ]
        if TRAIN["masked_learning"]:  # mask the RI coeffs and overlaps for bulk atoms
            io.log(
                log_path, 
                f"Mask targets and auxiliaries based on slab/interphase depths of {[TRAIN.get('slab_depth'), TRAIN.get('interphase_depth')]}"
            )
            target, aux = mask_dft_data(structure=[ALL_STRUCTURE[A] for A in subset_id], target=target, aux=aux)

        assert mts.equal_metadata(descriptor[0], target[0], check=["samples", "components"])

        datasets.append(
            IndexedDataset(
                sample_id=subset_id,
                structure=[ALL_STRUCTURE[A] for A in subset_id],
                descriptor=descriptor,
                target=target,
                aux=aux,
            )
        )
    dataloaders = [  # build dataloaders
        DataLoader(
            dset,
            collate_fn=group,
            batch_size=TRAIN["batch_size"],
        )
        for dset in datasets
    ]

    # ===== Model, optimizer, scheduler, loss fn =====
    if TRAIN.get("restart_epoch") is None:  # initialize
        epochs = torch.arange(TRAIN["n_epochs"] + 1)
        io.log(log_path, f"Epochs: {epochs[0]} -> {epochs[-1]}")
        io.log(log_path, "Initializing model, optimizer, loss fn")
        in_keys = descriptor[0].keys
        invariant_key_idxs = [
            i for i, key in enumerate(in_keys) if key["spherical_harmonics_l"] == 0
        ]
        in_properties = [descriptor[0][key].properties for key in in_keys]
        out_properties = [target[0][key].properties for key in in_keys]
        model = Model(
            in_keys=in_keys,
            in_properties=in_properties,
            out_properties=out_properties,
            descriptor_calculator=descriptor_calculator,
            nn=get_nn(in_keys, invariant_key_idxs, in_properties, out_properties, DTYPE),
            **TORCH,
        )
        optimizer = OPTIMIZER(model.params())
        scheduler = None
        if SCHEDULER is not None:
            io.log(log_path, "Using LR scheduler")
            scheduler = SCHEDULER(optimizer)
        
    else:  # load
        epochs = torch.arange(TRAIN["restart_epoch"] + 1, TRAIN["n_epochs"] + 1)
        io.log(log_path, f"Epochs: {epochs[0]} -> {epochs[-1]}")
        io.log(log_path, "Loading model, optimizer, loss fn")
        model = torch.load(join(CHKPT_DIR(TRAIN["restart_epoch"]), "model.pt"))
        optimizer = torch.load(join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt"))
        scheduler = None
        if exists(join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt")):
            io.log(log_path, "Using LR scheduler")
            scheduler = torch.load(join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt"))

    loss_fn = LOSS_FN()

    # Try a model save/load
    torch.save(model, os.path.join(ML_DIR, "model.pt"))
    torch.load(os.path.join(ML_DIR, "model.pt"))
    os.remove(os.path.join(ML_DIR, "model.pt"))
    io.log(log_path, "Model Architecture:")
    io.log(str(model).replace("\n", "\n#"))

    #


if __name__ == "__name__":

    train()
