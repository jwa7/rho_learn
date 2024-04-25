from os.path import exists, join
import time

import torch

import metatensor.torch as mts
from metatensor.torch.learn.data import DataLoader, IndexedDataset, group

from rholearn import io, utils
from rhotrain.model import DescriptorCalculator, Model
from rhotrain.train_utils import *


def set_settings_gloablly(ml_settings: dict):
    global_vars = globals()
    for key, value in ml_settings.items():
        global_vars[key] = value


def train(ml_settings: dict):

    # ===== Setup =====
    t0_training = time.time()
    set_settings_gloablly(ml_settings)
    torch.manual_seed(SEED)
    torch.set_default_dtype(DTYPE)

    log_path = os.path.join(ML_DIR, "training.log")
    utils.log(log_path, f"Setup")
    utils.log(log_path, f"Top directory defined as: {TOP_DIR}")

    # ===== Create datasets and dataloaders =====
    utils.log(log_path, f"Split structure ids into subsets")
    all_subset_id = data.group_idxs(  # cross-validation split of idxs
        idxs=STRUCTURE_ID,
        n_groups=CROSSVAL["n_groups"],
        group_sizes=CROSSVAL["group_sizes"],
        shuffle=CROSSVAL["shuffle"],
        seed=SEED,
    )
    utils.log(log_path, f"Init descriptor calculator")
    descriptor_calculator = DescriptorCalculator(**DESCRIPTOR_HYPERS)
    datasets = []
    for subset_id in all_subset_id:  # build each cross-val dataset
        utils.log(log_path, f"Build dataset for structures {subset_id}")
        selected_samples = get_selected_samples(  # for structure and/or atomic subsets
            structure=[ALL_STRUCTURE[A] for A in subset_id],
            structure_id=subset_id,
            masked_learning=TRAIN["masked_learning"],
            slab_depth=TRAIN.get("slab_depth"),
            interphase_depth=TRAIN.get("interphase_depth"),
        )
        utils.log(log_path, f"Calculate descriptors for subset {subset_id}")
        descriptor = calculate_descriptors(  # pre-compute descriptors
            descriptor_calculator=descriptor_calculator,
            all_structure=ALL_STRUCTURE,
            all_structure_id=ALL_STRUCTURE_ID,
            structure_id=subset_id,
            correlate_what="pxp",
            selected_samples=selected_samples,
        )
        utils.log(log_path, "Load targets and auxiliaries")
        target = [  # load RI coeffs
            load_dft_data(join(PROCESSED_DIR(A), "ri_coeffs.npz"), torch_kwargs=TORCH)
            for A in subset_id
        ]
        aux = [  # load overlaps
            load_dft_data(join(PROCESSED_DIR(A), "ri_ovlp.npz"), torch_kwargs=TORCH)
            for A in subset_id
        ]
        if TRAIN["masked_learning"]:  # mask the RI coeffs and overlaps for bulk atoms
            utils.log(
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
        utils.log(log_path, "Initializing model, optimizer, loss fn")
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
            utils.log(log_path, "Using LR scheduler")
            scheduler = SCHEDULER(optimizer)

    else:  # load
        utils.log(log_path, "Loading model, optimizer, loss fn")
        epochs = torch.arange(TRAIN["restart_epoch"] + 1, TRAIN["n_epochs"] + 1)
        model = torch.load(join(CHKPT_DIR(TRAIN["restart_epoch"]), "model.pt"))
        optimizer = torch.load(join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt"))
        scheduler = None
        if exists(join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt")):
            utils.log(log_path, "Using LR scheduler")
            scheduler = torch.load(
                join(CHKPT_DIR(TRAIN["restart_epoch"]), "optimizer.pt")
            )

    loss_fn = LOSS_FN()

    # Try a model save/load
    torch.save(model, os.path.join(ML_DIR, "model.pt"))
    torch.load(os.path.join(ML_DIR, "model.pt"))
    os.remove(os.path.join(ML_DIR, "model.pt"))
    utils.log(log_path, "Model Architecture:")
    utils.log(str(model).replace("\n", "\n#"))

    # ===== Training loop =====
    utils.log(log_path, f"Start training loops. Epochs: {epochs[0]} -> {epochs[-1]}")
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
            if (
                scheduler is not None
            ):  # update learning rate based on the validation loss
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
            utils.log(log_path, log_msg)

            if TRAIN("log_block_loss"):
                block_losses = get_block_losses(model, train_loader)
                for key, block_loss in block_losses.items():
                    utils.log(log_path, f"    key {key} block_loss {block_loss}")

        # Save checkpoint
        if epoch % TRAIN_["checkpoint_interval"] == 0:
            save_checkpoint(model, optimizer, scheduler, chkpt_dir=CHKPT_DIR(epoch))

    # Finish
    dt_training = time.time() - t0_training
    if dt_training <= 60:
        utils.log(log_path, f"Training complete in {dt_training} seconds.")
    elif 60 < dt_training <= 3600:
        utils.log(log_path, f"Training complete in {dt_training / 60} minutes.")
    else:
        utils.log(log_path, f"Training complete in {dt_training / 3600} hours.")
