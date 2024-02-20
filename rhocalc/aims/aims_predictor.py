"""
Module for taking an ML prediction of the RI coefficients and rebuilding the scalar
field in FHI-aims.
"""

import os
from typing import Callable, List

import ase
import numpy as np

import metatensor as mts

from rhocalc import convert
from rhocalc.aims import aims_calc


def target_builder(
    structure_id: List[int],
    structure: List[ase.Atoms],
    prediction: List[mts.TensorMap],
    save_dir: Callable,
    return_targets: bool = True,
    **kwargs
):
    """
    Takes the RI coefficients predicted by the model. Converts it from TensorMap to
    numpy format, reorders the array according to the AIMS convention, then rebuilds the
    scalar field by calling AIMS.

    if `return_targets` is True, then this function waits for the AIMS calculations to
    finish then returns rebuilt scalar fields.
    """
    calcs = {
        A: {"atoms": struct, "run_dir": save_dir(A)}
        for A, struct in zip(structure_id, structure)
    }

    # Add tailored cube edges for each structure if applicable
    if kwargs["aims_kwargs"].get("output") == ["cube ri_fit"]:
        if kwargs["cube"].get("slab") is True:
            for A in structure_id:
                calcs[A]["aims_kwargs"] = aims_calc.get_aims_cube_edges_slab(
                    calcs[A]["atoms"], kwargs["cube"].get("n_points")
                )
        else:
            for A in structure_id:
                calcs[A]["aims_kwargs"] = aims_calc.get_aims_cube_edges(
                    calcs[A]["atoms"], kwargs["cube"].get("n_points")
                )

    # Convert to a list of numpy arrays
    for A, struct, pred in zip(structure_id, structure, prediction):
        pred_np = convert.coeff_vector_tensormap_to_ndarray(
            struct=struct,
            tensor=pred,
            lmax=kwargs["basis_set"]["lmax"],
            nmax=kwargs["basis_set"]["nmax"],
        )
        # Save coeffs to "ri_coeffs.in"
        if not os.path.exists(save_dir(A)):
            os.makedirs(save_dir(A))
        np.savetxt(os.path.join(save_dir(A), "ri_coeffs.in"), pred_np)

    # If an "aims.out" file already exists in the save directory, delete it.
    # This prevents this function from incorrectly determining that the AIMS
    # calculation has finished, while not being able to delete the directory
    all_aims_outs = [os.path.join(save_dir(A), "aims.out") for A in structure_id]
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
        load_modules=kwargs["hpc_kwargs"]["load_modules"],
        run_command=kwargs["hpc_kwargs"]["run_command"],
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
    for A in structure_id:
        target = np.loadtxt(
            os.path.join(save_dir(A), "rho_rebuilt.out"),
            usecols=(0, 1, 2, 3),  # grid x, grid y, grid z, value
        )
        targets.append(target)

    return targets
