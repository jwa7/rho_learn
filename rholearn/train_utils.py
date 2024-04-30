"""
Module containing functions to perform model training and evaluation steps.
"""

import os
from os.path import exists, join
import time
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import ase
import numpy as np
import torch

import metatensor.torch as mts
from metatensor.torch.learn.data import IndexedDataset
from metatensor.torch.learn.nn import Sequential

import rascaline.torch

from rhocalc.aims import aims_fields
from rhocalc.ase import structure_builder
from rholearn import data
from rholearn.model import DescriptorCalculator


def get_selected_samples(
    structure: List[ase.Atoms],
    structure_id: List[int],
    masked_learning: bool = False,
    slab_depth: Optional[float] = None,
    interphase_depth: Optional[float] = None,
) -> mts.Labels:
    """
    Generates a `mts.Labels` object of the samples to compute when calling the
    DescriptorCalculator. There are 2 uses of this:

    1. Passing a global set of frames to, i.e. rascaline.SphericalExpansion, to then
       compute only a subset. This ensures the feature space has the global dimension.
    2. Computing the atom-centered density correlations for a subset of atoms within
       each frame. This is useful for masked learning, i.e. for learning the surfaces of
       slabs.

    If `masked_learning` is true, `slab_depth` and `interphase_depth` must be passed.
    """
    # Normal use case: subset of structures
    if not masked_learning:
        return mts.Labels(
            names=["system"],
            values=torch.tensor(structure_id).reshape(-1, 1),
        )

    # Extended use case: subset of structures and subset of atoms within them
    selected_samples = []
    for A, frame in zip(structure_id, structure):
        # Partition atoms into S, I, B regions
        idxs_surface, idxs_interphase, idxs_bulk = (
            structure_builder.get_atom_idxs_by_region(
                frame, slab_depth, interphase_depth
            )
        )
        # Keep S + I atoms
        for atom_i in list(idxs_surface) + list(idxs_interphase):
            selected_samples.append([A, atom_i])

    return mts.Labels(names=["system", "atom"], values=torch.tensor(selected_samples))


def calculate_descriptors(
    descriptor_calculator: torch.nn.Module,
    all_structure: List[ase.Atoms],
    all_structure_id: List[int],
    structure_id: List[int],
    correlate_what: str = "pxp",
    selected_samples: Optional[mts.Labels] = None,
    drop_empty_blocks: bool = True,
) -> List[torch.ScriptObject]:
    """
    Takes an initialized DescriptorCalculator and calculates descriptors for the frames
    in `all_structure` corresponding to those indexed in `structure_id`.
    """
    if selected_samples:
        compute_args = {"selected_samples": selected_samples}
    else:
        compute_args = {}
    descriptor = descriptor_calculator(
        system=rascaline.torch.systems_to_torch(all_structure),
        structure_id=structure_id,
        correlate_what=correlate_what,
        spherical_expansion_compute_args=compute_args,
        lode_spherical_expansion_compute_args=compute_args,
        density_correlations_compute_args={},
    )
    if drop_empty_blocks:
        descriptor_dropped = []
        for desc in descriptor:
            # Find empty blocks
            keys_to_drop = []
            for key, block in desc.items():
                if block.values.shape[0] == 0:  # has been sliced to zero samples
                    keys_to_drop.append(key)

            if len(keys_to_drop) > 0:  # Drop empty blocks
                desc_dropped = mts.drop_blocks(
                    desc,
                    keys=mts.Labels(
                        names=keys_to_drop[0].names,
                        values=torch.tensor(
                            [[i for i in k.values] for k in keys_to_drop]
                        ),
                    ),
                )
            else:  # no empty blocks to drop
                desc_dropped = desc
            descriptor_dropped.append(desc_dropped)

        descriptor = descriptor_dropped

    return descriptor


def load_dft_data(path: str, torch_kwargs: dict) -> torch.ScriptObject:
    """Loads a TensorMap from file and converts its backend to torch"""
    return mts.load(path).to(**torch_kwargs)


def mask_dft_data(
    structure: List[ase.Atoms],
    target: List[torch.ScriptObject],
    aux: List[torch.ScriptObject],
    slab_depth: float,
    interphase_depth: float,
) -> Tuple[torch.ScriptObject]:
    """
    Masks the RI coefficients and overlap TensorMaps according to the slab/interphase
    depth
    """
    target_masked, aux_masked = [], []
    for frame, t, o in zip(structure, target, aux):
        idxs_surface, idxs_interphase, idxs_bulk = (
            structure_builder.get_atom_idxs_by_region(
                frame, slab_depth, interphase_depth
            )
        )
        idxs_to_keep = list(idxs_surface) + list(idxs_interphase)

        target_masked.append(data.mask_coeff_vector_tensormap(t, idxs_to_keep))
        aux_masked.append(data.mask_ovlp_matrix_tensormap(o, idxs_to_keep))

    return target_masked, aux_masked


def parse_use_overlap_setting(
    use_overlap: Union[bool, int], epochs: torch.Tensor
) -> List[bool]:
    """
    Returns a list of boolean values, indicating whether to use the auxiliary
    data (i.e. overlaps) in loss evaluation at each of the epochs in `epochs`.
    """
    if isinstance(use_overlap, bool):
        return [use_overlap] * len(epochs)

    assert isinstance(use_overlap, int)

    parsed_use_overlap = []
    for epoch in epochs:
        if epoch < use_overlap:
            parsed_use_overlap.append(False)
        else:
            parsed_use_overlap.append(True)

    return parse_use_overlap


def training_step(
    train_loader,
    model,
    optimizer,
    loss_fn,
    check_metadata: bool = False,
    use_aux: bool = False,
) -> Tuple[torch.Tensor]:
    """
    Performs a single epoch of training by minibatching.
    """
    model.train()

    train_loss_epoch, n_train_epoch = 0, 0
    for train_batch in train_loader:

        optimizer.zero_grad()  # zero grads

        id_train, struct_train, in_train, out_train, aux_train = (
            train_batch  # unpack batch
        )
        if not use_aux:
            aux_train = None

        out_train_pred = model(  # forward pass
            descriptor=in_train, check_metadata=check_metadata
        )

        # Drop the blocks from the prediction that aren't part of the target
        if isinstance(out_train_pred, torch.ScriptObject):
            out_train_pred = mts.TensorMap(
                keys=out_train.keys,
                blocks=[out_train_pred[key].copy() for key in out_train.keys],
            )
        else:
            out_train_pred = [
                mts.TensorMap(
                    keys=out_train[i].keys,
                    blocks=[out_train_pred[i][key].copy() for key in out_train[i].keys],
                )
                for i in range(len(out_train_pred))
            ]

        train_loss_batch = loss_fn(  # train loss
            input=out_train_pred,
            target=out_train,
            overlap=aux_train,
            check_metadata=check_metadata,
        )
        train_loss_batch.backward()  # backward pass
        optimizer.step()  # update parameters
        train_loss_epoch += train_loss_batch  # store loss
        n_train_epoch += len(id_train)  # accumulate num structures in epoch

    train_loss_epoch /= n_train_epoch  # normalize loss by num structures

    return train_loss_epoch


def validation_step(
    val_loader, model, loss_fn, check_metadata: bool = False, use_aux: bool = False
) -> torch.Tensor:
    """
    Performs a single validation step
    """
    with torch.no_grad():
        val_loss_epoch, out_val_pred = torch.nan, None
        val_loss_epoch = 0
        n_val_epoch = 0
        for val_batch in val_loader:  # minibatches

            id_val, struct_val, in_val, out_val, aux_val = val_batch  # unpack batch
            if not use_aux:
                aux_val = None

            out_val_pred = model(descriptor=in_val, check_metadata=check_metadata)
            val_loss_batch = loss_fn(  # validation loss
                input=out_val_pred,
                target=out_val,
                overlap=aux_val,
                check_metadata=check_metadata,
            )
            val_loss_epoch += val_loss_batch  # store loss
            n_val_epoch += len(id_val)

        val_loss_epoch /= n_val_epoch  # normalize loss

        return val_loss_epoch


def step_scheduler(val_loss: torch.Tensor, scheduler) -> None:
    """Updates the scheduler parameters based on the validation loss"""
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_losses[0])
    else:
        scheduler.step()


def evaluation_step(
    model,
    dataloader,
    save_dir: Callable,
    calculate_error: bool = False,
    target_type: Optional[str] = None,
    reference_dir: Optional[Callable] = None,
) -> Union[None, float]:
    """
    Evaluates the model by making a prediction (with no gradient tracking) and
    rebuilding the scalar field from these coefficients by calling AIMS. Rebuilt
    scalar fields are saved to `save_dir`, a callable called with each structure
    index as an argument.

    If `calculate_error` is set to true, the % MAE (normalized by the number of
    electrons) of the rebuilt scalar field relative to either the DFT scalar
    field (`target_type="ref"`) or the RI scalar field (`target_type="ri"`) is
    returned.

    In this case, the directories where the reference DFT calculation files are
    stored must be specified in `reference_dir`. This is again a callable,
    called with each structure index.
    """
    model.eval()

    assert target_type in ["ref", "ri"]

    # Compile relevant data from all minibatches
    structure_id, structure, descriptor = [], [], []
    for batch in dataloader:
        for idx in batch.sample_id:
            structure_id.append(idx)
        for struct in batch.structure:
            structure.append(struct)
        for desc in batch.descriptor:
            descriptor.append(desc)

    # Make prediction with the model
    with torch.no_grad():
        prediction = model(  # return a list of TensorMap (or ScriptObject)
            descriptor=descriptor, check_metadata=True
        )

    if not calculate_error:
        return np.nan

    assert reference_dir is not None

    percent_maes = []
    for A, frame, prediction in zip(structure_id, structure, predictions):

        grid = np.loadtxt(  # integration weights
            join(reference_dir(A), "partition_tab.out")
        )
        target = np.loadtxt(  # target scalar field
            join(reference_dir(A), f"rho_{target_type}.out")
        )
        percent_mae = aims_fields.get_percent_mae_between_fields(  # calc MAE
            input=prediction,
            target=target,
            grid=grid,
        )
        percent_maes.append(percent_mae)

    return np.mean(percent_maes)


def get_block_losses(model, dataloader):
    """
    For all structures in the dataloader, sums the squared errors on the
    predictions from the model and returns the total SE for each block in a
    dictionary.
    """
    with torch.no_grad():
        # Get all the descriptors structures
        descriptor = [desc for batch in dataloader for desc in batch.descriptor]
        target = [targ for batch in dataloader for targ in batch.target]
        prediction = model(descriptor=descriptor, check_metadata=False)
        keys = prediction[0].keys

        block_losses = {tuple(key): 0 for key in keys}
        for key in keys:
            for pred, targ in zip(prediction, target):
                if key not in targ.keys:  # target does not have this key
                    continue
                # Remove predicted blocks that aren't in the target
                pred = mts.TensorMap(
                    keys=targ.keys, blocks=[pred[key].copy() for key in targ.keys]
                )
                assert mts.equal_metadata(pred, targ)
                block_losses[tuple(key)] += torch.nn.MSELoss(reduction="sum")(
                    pred[key].values, targ[key].values
                )

    return block_losses


def save_checkpoint(model: torch.nn.Module, optimizer, scheduler, chkpt_dir: str):
    """
    Saves model object, model state dict, optimizer state dict, scheduler state dict,
    to file.
    """
    if not exists(chkpt_dir):  # create chkpoint dir
        os.makedirs(chkpt_dir)

    torch.save(model, join(chkpt_dir, f"model.pt"))  # model obj
    torch.save(  # model state dict
        model.state_dict(),
        join(chkpt_dir, f"model_state_dict.pt"),
    )
    # Optimizer and scheduler
    torch.save(optimizer.state_dict(), join(chkpt_dir, f"optimizer.pt"))
    if scheduler is not None:
        torch.save(
            scheduler.state_dict(),
            join(chkpt_dir, f"scheduler.pt"),
        )


def run_training_sbatch(run_dir: str, python_command: str, **kwargs) -> None:
    """
    Writes a bash script to `fname` that allows running of model training.
    `run_dir` must contain two files; "run_training.py" and "settings.py".
    """
    top_dir = os.getcwd()

    # Copy training script and settings
    shutil.copy(join(top_dir, "ml_settings.py"), join(run_dir, "ml_settings.py"))

    os.chdir(run_dir)
    fname = "run_training.sh"

    with open(join(run_dir, fname), "w+") as f:
        # Make a dir for the slurm outputs
        if not exists(join(run_dir, "slurm_out")):
            os.mkdir(join(run_dir, "slurm_out"))

        f.write("#!/bin/bash\n")  # Write the header
        for tag, val in kwargs.items():  # Write the sbatch parameters
            f.write(f"#SBATCH --{tag}={val}\n")
        f.write(f"#SBATCH --output={join(run_dir, 'slurm_out', 'slurm_train.out')}\n")
        f.write("#SBATCH --get-user-env\n\n")

        # Define the run directory, cd to it, run command
        f.write(f"RUNDIR={run_dir}\n")
        f.write("cd $RUNDIR\n\n")
        f.write(f"{python_command}\n")

    os.system(f"sbatch {fname}")
    os.chdir(top_dir)


def run_training_local(run_dir: str, python_command: str) -> None:
    """
    Runs the training loop in the local environment. `run_dir` must contain this
    file "run_training.py", and the settings file "settings.py".
    """
    top_dir = os.getcwd()
    os.chdir(run_dir)
    os.system(f"{python_command}")
    os.chdir(top_dir)
