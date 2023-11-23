"""
Module for user-defined functions.
"""
import os
import shutil
from typing import List, Callable

import ase
import numpy as np

import metatensor
import rascaline
from metatensor import Labels, TensorMap
from rascaline.utils import clebsch_gordan

from rhocalc import convert
from rhocalc.aims import aims_calc, aims_parser


def _template_descriptor_builder(frames: List[ase.Atoms], **kwargs) -> TensorMap:
    """
    Function to build a descriptor for input into a ML model. Must take as input
    a list of ASE Atoms objects, and return a TensorMap. What happens within the
    function body is completely customizable, but must be deterministic based on
    the inputs.
    """
    descriptor = None

    # Code here

    return descriptor


def _template_target_builder(target: TensorMap, **kwargs):
    """
    Function to transform the raw prediction of a metatensor/torch model into
    the desired format. This may include converting the TensorMap prediction
    into a format suitable for re-integration into a QC code and calculating a
    derived quantity.
    """
    # Code here

    return target


def descriptor_builder(frames: List[ase.Atoms], **kwargs) -> TensorMap:
    """
    Function to build a descriptor for input into a ML model. This function
    generates a spherical expansion with rascaline, then perfroms a single
    Clebsch-Gordan combination step to build a lambda-SOAP vector.
    """
    # Get relevant kwargs
    global_species = kwargs.get("global_species")
    rascal_settings = kwargs.get("rascal_settings")
    cg_settings = kwargs.get("cg_settings")
    torch_settings = kwargs.get("torch_settings")

    # Build spherical expansion
    nu1_tensor = rascaline.SphericalExpansion(**rascal_settings["hypers"]).compute(
        frames, **rascal_settings["compute"]
    )
    nu1_tensor = nu1_tensor.keys_to_properties(
        keys_to_move=Labels(
            names=["species_neighbor"], values=np.array(global_species).reshape(-1, 1)
        )
    )

    # Build lambda-SOAP vector
    lsoap = clebsch_gordan.single_center_combine_to_order(
        nu1_tensor, 
        **cg_settings,
    )

    # Convert to torch backend and return
    if torch_settings is not None:
        lsoap = metatensor.to(lsoap, "torch", **torch_settings)

    return lsoap


def target_builder(
    structure_idxs: List[int],
    frames: List[ase.Atoms],
    predictions: List[TensorMap],
    save_dir: Callable,
    return_targets: bool = True,
    **kwargs
):
    """
    Takes the RI coefficients predicted by the model. Converts it from TensorMap
    to numpy format, reorders the array according to the AIMS convention, then
    rebuilds the scalar field by calling AIMS.

    if `return_targets` is True, then this function waits for the AIMS
    calcualtions to finish then returns rebuilt scalar fields.
    """
    calcs = {
        A: {"atoms": frame, "run_dir": save_dir(A)}
        for A, frame in zip(structure_idxs, frames)
    }

    # Convert to a list of numpy arrays
    for A, frame, pred in zip(structure_idxs, frames, predictions):
        pred_np = convert.coeff_vector_tensormap_to_ndarray(
            frame=frame,
            tensor=pred,
            lmax=kwargs["basis_set"]["def"]["lmax"],
            nmax=kwargs["basis_set"]["def"]["nmax"],
        )
        # Convert to AIMS ordering and save to "ri_coeffs.in"
        pred_aims = aims_parser.coeff_vector_ndarray_to_aims_coeffs(
            coeffs=pred_np,
            basis_set_idxs=kwargs["basis_set"]["idxs"],
            save_dir=save_dir(A),
        )
    
    # If an "aims.out" file already exists in the save directory, delete it.
    # This prevents this function from incorrectly determining that the AIMS
    # calculation has finished, while not being able to delete the directory
    all_aims_outs = [os.path.join(save_dir(A), "aims.out") for A in structure_idxs]
    for aims_out in all_aims_outs:
        if os.path.exists(aims_out):
            os.remove(aims_out)

    # Run AIMS to build the target scalar field for each structure
    aims_calc.run_aims_array(
        calcs=calcs,
        aims_path=kwargs["aims_path"],
        aims_kwargs=kwargs["aims_kwargs"],
        sbatch_kwargs=kwargs["sbatch_kwargs"],
        run_dir=save_dir,  # must be a callable
    )

    if not return_targets:
        return
    
    # Wait until all AIMS calcs have finished, then read in and return the
    # target scalar fields
    all_finished = False
    while len(all_aims_outs) > 0:
        for aims_out in all_aims_outs:
            if os.path.exists(aims_out):
                with open(aims_out, "r") as f:
                    # Basic check to see if AIMS calc has finished
                    if "Leaving FHI-aims." in f.read():
                        all_aims_outs.remove(aims_out)

    targets = []
    for A in structure_idxs:
        target = np.loadtxt(
            os.path.join(save_dir(A), "rho_rebuilt.out"), 
            usecols=(0, 1, 2, 3),
        )
        targets.append(target)

    return targets
