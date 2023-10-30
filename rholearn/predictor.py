"""
Module for user-defined functions.
"""
import os
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
    nu_1_tensor = rascaline.SphericalExpansion(**rascal_settings["hypers"]).compute(
        frames, **rascal_settings["compute"]
    )
    nu_1_tensor = nu_1_tensor.keys_to_properties(
        keys_to_move=Labels(
            names=["species_neighbor"], values=np.array(global_species).reshape(-1, 1)
        )
    )

    # Build lambda-SOAP vector
    lsoap = clebsch_gordan.lambda_soap_vector(nu_1_tensor, **cg_settings)

    # Convert to torch backend and return
    if torch_settings is not None:
        lsoap = metatensor.to(lsoap, "torch", **torch_settings)

    return lsoap


def target_builder(
    structure_idxs: List[int],
    frames: List[ase.Atoms],
    predictions: List[TensorMap],
    save_dir: Callable,
    **kwargs
):
    """
    Takes the RI coefficients predicted by the model. Converts it from TensorMap
    to numpy format, reorders the array according to the AIMS convention, then
    rebuilds the scalar field by calling AIMS.
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

    # Run AIMS to build the target scalar field for each structure
    aims_calc.run_aims_array(
        calcs=calcs,
        aims_path=kwargs["aims_path"],
        aims_kwargs=kwargs["aims_kwargs"],
        sbatch_kwargs=kwargs["sbatch_kwargs"],
        run_dir=save_dir,  # must be a callable
    )
    
    # Wait until all AIMS calcs have finished, then read in and return the
    # target scalar fields
    all_finished = False
    while not all_finished:
        calcs_finished = []
        for A in structure_idxs:
            aims_out_path = os.path.join(save_dir(A), "aims.out")
            if os.path.exists(aims_out_path):
                with open(aims_out_path, "r") as f:
                    # Basic check to see if AIMS calc has finished
                    calcs_finished.append("Leaving FHI-aims." in f.read())
            else:
                calcs_finished.append(False)
        all_finished = np.all(calcs_finished)

    targets = []
    for A in structure_idxs:
        targets.append(np.loadtxt(os.path.join(save_dir(A), "rho_rebuilt.out")))

    return targets
