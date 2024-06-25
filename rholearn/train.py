from functools import partial
from os.path import exists, join
import time

import torch

import metatensor.torch as mts
from metatensor.torch.learn.data import (
    DataLoader,
    IndexedDataset,
    group,
    group_and_join,
)

from rholearn import utils
from rholearn.loss import L2Loss
from rholearn.model import DescriptorCalculator, Model
from rholearn.train_utils import *


def set_settings_gloablly(ml_settings: dict):
    """Sets the ML settings globally."""
    global_vars = globals()
    for key, value in ml_settings.items():
        global_vars[key] = value


def get_train_dataset(
    subset_id: List[int],
    descriptor_calculator: torch.nn.Module,
    log_path: str,
) -> torch.nn.Module:
    """Builds the training dataset and dataloader."""

    utils.log(log_path, f"Handling training subset {subset_id}")

    # Load targets
    utils.log(log_path, f"Load training targets with name '{DATA_NAMES['target']}'")
    target = [  # load RI coeffs
        load_dft_data(join(PROCESSED_DIR(A), DATA_NAMES["target"]), torch_kwargs=TORCH)
        for A in subset_id
    ]

    # Load overlaps
    if TRAIN["use_overlap"] is False:
        utils.log(log_path, f"Training overlaps will not be loaded")
        aux = None
    else:
        if TRAIN["overlap_type"] == "diag":  # diagonal overlap
            train_ovlp_name = DATA_NAMES["aux_diag"]
        else:  # full overlap
            train_ovlp_name = DATA_NAMES["aux"]
        utils.log(log_path, f"Load training auxiliaries with name '{train_ovlp_name}'")
        aux = [  # load overlaps
            load_dft_data(join(PROCESSED_DIR(A), train_ovlp_name), torch_kwargs=TORCH)
            for A in subset_id
        ]

    # Calculate descriptors
    utils.log(log_path, f"Calculate training descriptors")
    descriptor = descriptor_calculator(
        system=rascaline.torch.systems_to_torch([ALL_SYSTEM[A] for A in subset_id]),
        system_id=torch.tensor(subset_id, dtype=torch.int32),
        split_by_system=True,
        drop_empty_blocks=False,
    )

    # Check metadata
    utils.log(log_path, f"Check metadata")
    new_descriptor = []
    for desc, targ in zip(descriptor, target):  # check metadata match
        # Drop blocks not present in the target
        keys_to_drop = [key for key in desc.keys if key not in targ.keys]
        if len(keys_to_drop) > 0:
            desc = mts.drop_blocks(
                desc,
                keys=mts.Labels(
                    names=keys_to_drop[0].names,
                    values=torch.tensor([[i for i in k.values] for k in keys_to_drop]),
                ),
            )
        mts.equal_metadata_raise(desc, targ, check=["samples", "components"])
        new_descriptor.append(desc)
    descriptor = new_descriptor

    # Log mean sizes of descriptor, target, aux
    for data_field, data_name in zip(
        [descriptor, target, aux], ["descriptor", "target", "aux"]
    ):
        if data_field is None:
            assert data_name == "aux"
            continue
        mean_field_size = torch.tensor(
            [
                torch.tensor(
                    [
                        block.values.element_size() * block.values.nelement() * 1e-6
                        for block in tensor
                    ]
                ).sum()
                for tensor in data_field
            ]
        ).mean()

        utils.log(
            log_path,
            f"Mean training {data_name} size in MB (block tensors only, no metadata): {mean_field_size:.5f}",
        )

    # Build dataset
    utils.log(log_path, f"Build training dataset")
    aux = {"aux": aux} if aux is not None else {}
    train_dataset = IndexedDataset(
        sample_id=list(subset_id),
        frame=[ALL_SYSTEM[A] for A in subset_id],
        descriptor=[mts.sort(d) for d in descriptor],
        target=target,
        **aux,
    )

    return train_dataset


def get_train_dataloader(train_dataset: torch.nn.Module) -> torch.nn.Module:
    """Builds the training dataloader."""

    train_dataloader = {}
    if TRAIN["use_overlap"] != True:  # could be False or an int
        # Training at some point will not use the overlap for evaluating the training
        # loss. Overlaps are not
        train_dataloader[False] = DataLoader(
            train_dataset,
            collate_fn=partial(
                group_and_join, join_kwargs={"remove_tensor_name": True}
            ),
            batch_size=TRAIN["batch_size"],
        )

    if TRAIN["use_overlap"] != False:  # could be True or an int
        # Training at some point will use the overlap for evaluating the training loss.
        if TRAIN["overlap_type"] == "diag":  # can still use group_and_join
            train_dataloader[True] = DataLoader(
                train_dataset,
                collate_fn=partial(
                    group_and_join, join_kwargs={"remove_tensor_name": True}
                ),
                batch_size=TRAIN["batch_size"],
            )
        else:  # full matrix: must use group
            train_dataloader[True] = DataLoader(
                train_dataset,
                collate_fn=group,
                batch_size=TRAIN["batch_size"],
            )

    return train_dataloader


def get_val_dataset(
    subset_id: List[int],
    descriptor_calculator: torch.nn.Module,
    log_path: str,
) -> torch.nn.Module:
    """Builds the validation dataset and dataloader."""

    utils.log(log_path, f"Handling validation subset {subset_id}")

    # Load targets
    utils.log(log_path, f"Load validation targets with name '{DATA_NAMES['target']}'")
    target = [  # load RI coeffs
        load_dft_data(join(PROCESSED_DIR(A), DATA_NAMES["target"]), torch_kwargs=TORCH)
        for A in subset_id
    ]

    # Load overlaps
    val_ovlp_name = DATA_NAMES["aux"]
    utils.log(log_path, f"Load validation auxiliaries with name '{val_ovlp_name}'")
    aux = [  # load overlaps
        load_dft_data(join(PROCESSED_DIR(A), val_ovlp_name), torch_kwargs=TORCH)
        for A in subset_id
    ]

    # Calculate descriptors
    utils.log(log_path, f"Calculate validation descriptors")
    descriptor = descriptor_calculator(
        system=rascaline.torch.systems_to_torch([ALL_SYSTEM[A] for A in subset_id]),
        system_id=torch.tensor(subset_id, dtype=torch.int32),
        split_by_system=True,
        drop_empty_blocks=False,
    )

    # Check metadata
    utils.log(log_path, f"Check metadata")
    new_descriptor = []
    for desc, targ in zip(descriptor, target):  # check metadata match
        # Drop blocks not present in the target
        keys_to_drop = [key for key in desc.keys if key not in targ.keys]
        if len(keys_to_drop) > 0:
            desc = mts.drop_blocks(
                desc,
                keys=mts.Labels(
                    names=keys_to_drop[0].names,
                    values=torch.tensor([[i for i in k.values] for k in keys_to_drop]),
                ),
            )
        mts.equal_metadata_raise(desc, targ, check=["samples", "components"])
        new_descriptor.append(desc)
    descriptor = new_descriptor

    # Log mean sizes of descriptor, target, aux
    for data_field, data_name in zip(
        [descriptor, target, aux], ["descriptor", "target", "aux"]
    ):
        if data_field is None:
            assert data_name == "aux"
            continue
        mean_field_size = torch.tensor(
            [
                torch.tensor(
                    [
                        block.values.element_size() * block.values.nelement() * 1e-6
                        for block in tensor
                    ]
                ).sum()
                for tensor in data_field
            ]
        ).mean()

        utils.log(
            log_path,
            f"Mean validation {data_name} size in MB (block tensors only, no metadata): {mean_field_size:.5f}",
        )

    # Build dataset
    utils.log(log_path, f"Build validation dataset")
    val_dataset = IndexedDataset(
        sample_id=list(subset_id),
        frame=[ALL_SYSTEM[A] for A in subset_id],
        descriptor=[mts.sort(d) for d in descriptor],
        target=target,
        aux=aux,
    )

    return val_dataset


def get_val_dataloader(val_dataset: torch.nn.Module) -> torch.nn.Module:
    """Builds the validation dataloader."""

    # Full overlaps are always used for validation, so collate_fn=group must be used.
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=group,
        batch_size=val_dataset._size,
    )

    return val_dataloader


def train(ml_settings: dict):
    """
    Runs model training in the following steps:
        1. Split structures into subsets
        2. Calculate descriptors and oad targets and auxiliaries, masking if appropriate
        3. Build cross-validation datasets and dataloaders
        4. Initialize model, optimizer, scheduler, loss function
        5. Iteratively train, validate, and checkpoint
    """

    # ===== Setup =====
    t0_training = time.time()
    set_settings_gloablly(ml_settings)
    torch.manual_seed(SEED)
    torch.set_default_dtype(DTYPE)

    log_path = os.path.join(ML_DIR, "logs/train.log")
    losses_log_path = os.path.join(ML_DIR, "logs/losses.log")
    block_losses_log_path = os.path.join(ML_DIR, "logs/block_losses.log")
    os.makedirs(os.path.join(ML_DIR, "logs"), exist_ok=True)
    utils.log(log_path, f"===== BEGIN =====")
    utils.log(log_path, f"Working directory: {ML_DIR}")

    # ===== Create datasets and dataloaders =====
    if ALL_SUBSET_ID is None:
        utils.log(log_path, f"Split structure ids into subsets")
        all_subset_id = data.group_idxs(  # cross-validation split of idxs
            idxs=SYSTEM_ID,
            n_groups=CROSSVAL["n_groups"],
            group_sizes=CROSSVAL["group_sizes"],
            shuffle=CROSSVAL["shuffle"],
            seed=SEED,
        )
    else:
        all_subset_id = ALL_SUBSET_ID
    utils.log(log_path, f"Subset IDs are: {all_subset_id}")
    utils.log(log_path, f"Init descriptor calculator")
    descriptor_calculator = DescriptorCalculator(**DESCRIPTOR_HYPERS)

    # Training dataset and dataloader
    utils.log(log_path, str(TRAIN))
    train_dataset = get_train_dataset(all_subset_id[0], descriptor_calculator, log_path)
    train_dataloader = get_train_dataloader(train_dataset)

    # Validation dataset and dataloader
    val_dataset = get_val_dataset(all_subset_id[1], descriptor_calculator, log_path)
    val_dataloader = get_val_dataloader(val_dataset)

    # ===== Model, optimizer, scheduler, loss fn =====
    if TRAIN.get("restart_epoch") is None:  # initialize
        epochs = torch.arange(TRAIN["n_epochs"] + 1)
        if PRETRAINED_MODEL is None:  # initialize model from scratch
            utils.log(log_path, f"Target basis set definition: {TARGET_BASIS}")
            utils.log(log_path, "Initializing model, optimizer, loss fn")
            model = Model(
                target_basis=TARGET_BASIS,
                descriptor_calculator=descriptor_calculator,
                net=NET,
                **TORCH,
            )

        else:  # Use pre-trained model
            utils.log(
                log_path, "Using pre-trained model, initializing optimizer and loss fn"
            )
            model = PRETRAINED_MODEL
        # optimizer = OPTIMIZER(model.parameters())
        optimizer = OPTIMIZER(filter(lambda p: p.requires_grad, model.parameters()))
        scheduler = None
        if SCHEDULER is not None:
            utils.log(log_path, "Using LR scheduler")
            scheduler = SCHEDULER(optimizer)

    else:  # load from checkpoint for restart
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

    # Try a model save/load
    torch.save(model, os.path.join(ML_DIR, "model.pt"))
    torch.load(os.path.join(ML_DIR, "model.pt"))
    os.remove(os.path.join(ML_DIR, "model.pt"))
    utils.log(log_path, "Model Architecture:")
    utils.log(log_path, str(model).replace("\n", "\n#"))

    # ===== Training loop =====
    utils.log(log_path, f"Start training loops. Epochs: {epochs[0]} -> {epochs[-1]}")
    use_overlap = parse_use_overlap_setting(TRAIN["use_overlap"], epochs)

    with open(losses_log_path, "a") as f:
        f.write("epoch train_loss val_loss val_loss_onsite val_loss_offsite val_loss_full\n")
    key_tuples = [tuple(key) for key in val_dataset[0].target.keys]
    with open(block_losses_log_path, "a") as f:
        f.write(f"epoch {' '.join([str(key).replace(' ', '') for key in key_tuples])}\n")

    best_val_loss = torch.tensor(float("inf"))
    for epoch, use_ovlp in zip(epochs, use_overlap):

        # Train step
        t0_train = time.time()
        train_loss = training_step(  # train subset
            train_dataloader[use_ovlp],
            model,
            optimizer,
            TRAIN_LOSS_FN,
            check_metadata=epoch == 0,
            use_aux=use_ovlp,
        )
        t1_train = time.time()

        # Val step - onsite loss
        t0_val = time.time()
        # val_loss = validation_step(
        #     val_dataloader, model, VAL_LOSS_FN, use_aux=True
        # )
        val_loss = torch.nan
        t1_val = time.time()

        # Step scheduler params based on val loss
        lr = torch.nan
        if scheduler is not None:
            step_scheduler(val_loss, scheduler)
            lr = scheduler._last_lr[0]

        # Log some results
        if epoch % TRAIN["log_interval"] == 0:

            # Val step - onsite loss
            t0_val_onsite = time.time()
            val_loss_onsite = validation_step(
                val_dataloader, model, L2Loss(overlap_type="on-site"), use_aux=True
            )
            # val_loss_onsite = torch.nan
            t1_val_onsite = time.time()

            # Val step - offsite loss
            t0_val_offsite = time.time()
            val_loss_offsite = validation_step(
                val_dataloader, model, L2Loss(overlap_type="off-site"), use_aux=True
            )
            # val_loss_offsite = torch.nan
            t1_val_offsite = time.time()

            # Val step - full loss
            t0_val_full = time.time()
            val_loss_full = validation_step(
                val_dataloader, model, L2Loss(overlap_type=None), use_aux=True
            )
            # val_loss_full = torch.nan
            t1_val_full = time.time()

            # Log general info and losses
            log_msg = (
                f"epoch {epoch}"
                f" train_loss {train_loss}"
                f" val_loss {val_loss}"
                f" val_loss_onsite {val_loss_onsite}"
                f" val_loss_offsite {val_loss_offsite}"
                f" val_loss_full {val_loss_full}"
                f" train_time {(t1_train - t0_train):.3f}"
                f" val_time {(t1_val - t0_val):.3f}"
                f" val_time_onsite {(t1_val_onsite - t0_val_onsite):.3f}"
                f" val_time_offsite {(t1_val_offsite - t0_val_offsite):.3f}"
                f" val_time_full {(t1_val_full - t0_val_full):.3f}"
                f" lr {lr}"
                f" use_ovlp {use_ovlp}"
            )
            utils.log(log_path, log_msg)
            with open(losses_log_path, "a") as f:
                f.write(f"{epoch} {train_loss} {val_loss_onsite} {val_loss_offsite} {val_loss_full}\n")

            if TRAIN["log_block_loss"]:

                block_losses_train = get_block_losses(model, train_dataloader[use_ovlp])
                block_losses_val = get_block_losses(model, val_dataloader)
                utils.log(log_path, "  Train block losses:")
                for key, block_loss in block_losses_train.items():
                    utils.log(log_path, f"    key {key} block_loss {block_loss}")
                utils.log(log_path, "  Validation block losses:")
                for key, block_loss in block_losses_val.items():
                    utils.log(log_path, f"    key {key} block_loss {block_loss}")

                with open(block_losses_log_path, "a") as f:
                    f.write(f"{epoch} {' '.join([str(block_losses_val[key].item()) for key in key_tuples])}\n")

        # Save checkpoint
        if epoch % TRAIN["checkpoint_interval"] == 0:
            save_checkpoint(model, optimizer, scheduler, chkpt_dir=CHKPT_DIR(epoch))

        # Save checkpoint if best validation loss
        if val_loss_full < best_val_loss:
            best_val_loss = val_loss_full
            save_checkpoint(model, optimizer, scheduler, chkpt_dir=CHKPT_DIR("best"))


    # Finish
    dt_training = time.time() - t0_training
    if dt_training <= 60:
        utils.log(log_path, f"Training complete in {dt_training} seconds. best_val_loss {best_val_loss:.5f}")
    elif 60 < dt_training <= 3600:
        utils.log(log_path, f"Training complete in {dt_training / 60} minutes. best_val_loss {best_val_loss:.5f}")
    else:
        utils.log(log_path, f"Training complete in {dt_training / 3600} hours. best_val_loss {best_val_loss:.5f}")
