from os.path import exists, join
import time

import torch

import metatensor.torch as mts
from metatensor.torch.learn.data import DataLoader, IndexedDataset, group

from rhotrain.model import DescriptorCalculator, Model
from rhotrain.train_utils import *

def set_settings_gloablly(settings: dict):
    # global SEED, DTYPE, ML_DIR, TOP_DIR, IDXS, CROSSVAL_SETTINGS, DESCRIPTOR_HYPERS, ALL_STRUCTURE, ALL_STRUCTURE_ID, PROCESSED_DIR, TRAIN, TORCH, OPTIMIZER, SCHEDULER, CHKPT_DIR, LOSS_FN
    
    SEED = settings["SEED"]
    DTYPE = settings["DTYPE"]
    ML_DIR = settings["ML_DIR"]
    TOP_DIR = settings["TOP_DIR"]
    IDXS = settings["IDXS"]
    CROSSVAL_SETTINGS = settings["CROSSVAL_SETTINGS"]
    DESCRIPTOR_HYPERS = settings["DESCRIPTOR_HYPERS"]
    ALL_STRUCTURE = settings["ALL_STRUCTURE"]
    ALL_STRUCTURE_ID = settings["ALL_STRUCTURE_ID"]
    PROCESSED_DIR = settings["PROCESSED_DIR"]
    TRAIN = settings["TRAIN"]
    TORCH = settings["TORCH"]
    OPTIMIZER = settings["OPTIMIZER"]
    SCHEDULER = settings["SCHEDULER"]
    CHKPT_DIR = settings["CHKPT_DIR"]
    LOSS_FN = settings["LOSS_FN"]


def train(settings: dict):

    set_settings_gloablly(settings)

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
            load_dft_data(join(PROCESSED_DIR(A), "ri_coeffs.npz")) for A in subset_id
        ]
        aux = [  # load overlaps
            load_dft_data(join(PROCESSED_DIR(A), "ri_ovlp.npz")) for A in subset_id
        ]
        if TRAIN["masked_learning"]:  # mask the RI coeffs and overlaps for bulk atoms
            io.log(
                log_path,
                f"Mask targets and auxiliaries based on slab/interphase depths of {[TRAIN.get('slab_depth'), TRAIN.get('interphase_depth')]}",
            )
            target, aux = mask_dft_data(
                structure=[ALL_STRUCTURE[A] for A in subset_id], target=target, aux=aux
            )

        assert mts.equal_metadata(
            descriptor[0], target[0], check=["samples", "components"]
        )

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
        io.log(log_path, "Initializing model, optimizer, loss fn")
        epochs = torch.arange(TRAIN["n_epochs"] + 1)
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
            nn=get_nn(
                in_keys, invariant_key_idxs, in_properties, out_properties, DTYPE
            ),
            **TORCH,
        )
        optimizer = OPTIMIZER(model.params())
        scheduler = None
        if SCHEDULER is not None:
            io.log(log_path, "Using LR scheduler")
            scheduler = SCHEDULER(optimizer)

    else:  # load
        io.log(log_path, "Loading model, optimizer, loss fn")
        epochs = torch.arange(TRAIN["restart_epoch"] + 1, TRAIN["n_epochs"] + 1)
        model = torch.load(join(CHKPT_DIR(TRAIN["restart_epoch"]), "model.pt"))
        optimizer = torch.load(join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt"))
        scheduler = None
        if exists(join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt")):
            io.log(log_path, "Using LR scheduler")
            scheduler = torch.load(
                join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt")
            )

    loss_fn = LOSS_FN()

    # Try a model save/load
    torch.save(model, os.path.join(ML_DIR, "model.pt"))
    torch.load(os.path.join(ML_DIR, "model.pt"))
    os.remove(os.path.join(ML_DIR, "model.pt"))
    io.log(log_path, "Model Architecture:")
    io.log(str(model).replace("\n", "\n#"))

    # ===== Training loop =====
    io.log(log_path, f"Start training loops. Epochs: {epochs[0]} -> {epochs[-1]}")
    use_overlap = parse_use_overlap_setting(TRAIN["use_overlap"], epochs)

    for epoch, use_ovlp in zip(epochs, use_ovlps):

        t0 = time.time()
        train_loss = training_step(  # train subset
            dataloaders[0],
            model,
            optimizer,
            loss_fn,
            check_metadata=epoch == 0,
            use_aux=use_ovlp,
        )
        val_losses = []
        if len(dataloaders) > 1:  # i.e. val and test subsets, if present
            val_losses = [
                validation_step(dloader, model, loss_fn) for dloader in dataloaders[1:]
            ]
            if scheduler is not None:  # update learning rate based on the validation loss
                scheduler.step(val_losses[0])
                lr = scheduler._last_lr[0]
            else:
                lr = torch.nan
        
        # Log general info and losses
        if epoch % TRAIN("log_interval") == 0:
            log_msg = f"epoch {epoch} lr {lr} time {time.time() - t0} use_ovlp {use_ovlp} train_loss {train_loss}"
            if len(val_losses) > 0:
                for i, tmp_val_loss in enumerate(val_losses):
                    log_msg += f"val_loss_{i} {tmp_val_loss}"
            io.log(log_path, log_msg)

            if TRAIN("log_block_loss"):
                block_losses = get_block_losses(model, train_loader)
                for key, block_loss in block_losses.items():
                    io.log(log_path, f"    key {key} block_loss {block_loss}")

        # Save checkpoint
        if epoch % TRAIN_["checkpoint_interval"] == 0:
            save_checkpoint(model, optimizer, scheduler, chkpt_dir=CHKPT_DIR(epoch))


    # Finish
    dt_training = time.time() - t0_training
    if dt_training <= 60:
        io.log(log_path, f"Training complete in {dt_training} seconds.")
    elif 60 < dt_training <= 3600:
        io.log(log_path, f"Training complete in {dt_training / 60} minutes.")
    else:
        io.log(log_path, f"Training complete in {dt_training / 3600} hours.")

