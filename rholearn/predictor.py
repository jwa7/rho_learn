"""
Module for user-defined functions.
"""
from typing import List

import ase
import numpy as np

import metatensor
import rascaline
from metatensor import Labels, TensorMap
from rascaline.utils import clebsch_gordan


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

    # Convert to torch backedn and return
    if torch_settings is not None:
        lsoap = metatensor.to(lsoap, "torch", **torch_settings)

    return lsoap


def target_builder(target: TensorMap, **kwargs):
    """
    Takes the RI coefficients predicted by the model. Converts it from TensorMap
    to numpy format, reorders the array according to the AIMS convention, then
    rebuilds the scalar field by calling AIMS.

    Calls AIMS to output the scalar field rebuilt from the RI coefficients, both
    on the AIMS and CUBE grids. Also outputs the integration weights.
    """
    

    return target
