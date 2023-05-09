"""
Generates features vectors for equivariant structural representations.
Currently implemented:
    - lambda-SOAP
"""
import os
import pickle
from typing import List, Optional

import numpy as np

import rascaline
import equistore
from equistore import Labels, TensorMap

from rholearn import io, spherical, utils


def lambda_feature_vector(
    frames: list,
    hypers: dict,
    calc: str,
    lambda_max: Optional[int] = None,
    neighbor_species: Optional[List[int]] = None,
    even_parity_only: bool = False,
    save_dir: Optional[str] = None,
) -> TensorMap:
    """
    Takes a list of frames of ASE loaded structures and a dict of Rascaline
    hyperparameters and generates a lambda (i.e. nu=2) representation of the
    data. Either short-ranged SOAP or long-range LODE atom centered density
    correlations can be calculated and combined to form the equivariant
    lambda-features.

    :param frames: a list of structures generated by the ase.io function.
    :param hypers: a dict of hyperparameters used to calculate the atom density
        correlation calculated with either the Rascaline "SphericalExpansion" or
        "LodeSphericalExpansion" calculator.
    :param calc: a str of the calculator used to calculate the atom density
        correlations, must be either "SphericalExpansion" (i.e. for short-range)
        or "LodeSphericalExpansion" (for long-range).
    :param lambda_max: an int of the maximum lambda value to include in the
        final lambda representation. If none, the 'max_angular' value in
        `hypers` will be used instead.
    :param neighbor_species: a list of int that correspond to the atomic charges
        of all the neighbour species that you want to be in your properties (or
        features) dimension. This list may contain charges for atoms that don't
        appear in ``frames``, but are included anyway so that the one can
        enforce consistent properties dimension size with other lambda-feature
        vectors.
    :param even_parity_only: a bool that determines whether to only include the
        key/block pairs with even parity under rotation, i.e. sigma = +1.
        Defaults to false, where both parities are included.
    :param save_dir: a str of the absolute path to the directory where the
        TensorMap of the calculated lambda-SOAP representation and pickled
        ``hypers`` dict should be written. If none, the TensorMap will not be
        saved.

    :return: a TensorMap of the lambda-representation vector of the input
        frames.
    """
    # Create save directory
    if save_dir is not None:
        io.check_or_create_dir(save_dir)

    # Generate Rascaline hypers and Clebsch-Gordon coefficients
    if calc == "SphericalExpansion":
        calculator = rascaline.SphericalExpansion(**hypers)
    elif calc == "LodeSphericalExpansion":
        calculator = rascaline.LodeSphericalExpansion(**hypers)
    else:
        raise ValueError(
            f"calc must be either 'SphericalExpansion' or 'LodeSphericalExpansion',"
            + f" not {calc}."
        )
    if lambda_max is None:
        lambda_max = hypers["max_angular"]
    else:
        if lambda_max > 2 * hypers["max_angular"]:
            raise ValueError(
                "As this function generates 2-body features (nu=2), `lambda_max` must"
                f" be <= 2 x hypers['max_angular'] `hypers`. Received {lambda_max}."
            )
    # Build ClebschGordan object
    cg = spherical.ClebschGordanReal(l_max=lambda_max)

    # Generate descriptor via Spherical Expansion
    print("Computing spherical expansion")
    acdc_nu1 = calculator.compute(frames)

    # nu=1 features
    print("Standardizing keys")
    acdc_nu1 = spherical.acdc_standardize_keys(acdc_nu1)

    # Move "species_neighbor" sparse keys to properties with enforced atom
    # charges if ``neighbor_species`` is specified. This is required as the CG
    # iteration code currently does not handle neighbour species padding
    # automatically.
    keys_to_move = "species_neighbor"
    if neighbor_species is not None:
        keys_to_move = Labels(
            names=(keys_to_move,),
            values=np.array(neighbor_species).reshape(-1, 1),
        )
    print("Moving keys to properties")
    acdc_nu1 = acdc_nu1.keys_to_properties(keys_to_move=keys_to_move)

    # Combined nu=1 features to generate nu=2 features. lambda-SOAP is defined
    # as just the nu=2 features.
    print("Performing CG iteration to generate nu=2 features")
    acdc_nu2 = spherical.cg_increment(
        acdc_nu1,
        acdc_nu1,
        clebsch_gordan=cg,
        lcut=lambda_max,
        other_keys_match=["species_center"],
    )
    # Release acdc_nu1 from memory
    del acdc_nu1
    utils.trim_memory()

    # Clean the lambda-SOAP TensorMap. Drop the order_nu key name as this is by
    # definition 2 for all keys.
    print("Dropping 'order_nu' from the key names")
    acdc_nu2 = utils.drop_key_name(acdc_nu2, key_name="order_nu")
    utils.trim_memory()

    if even_parity_only:
        # Drop all odd parity keys/blocks
        print("Dropping blocks with odd parity")
        keys_to_drop = acdc_nu2.keys[acdc_nu2.keys["inversion_sigma"] == -1]
        acdc_nu2 = equistore.drop_blocks(acdc_nu2, keys=keys_to_drop)
        utils.trim_memory()

        # Drop the inversion_sigma key name as this is now +1 for all blocks
        print("Dropping 'inversion_sigma' from the key names")
        acdc_nu2 = utils.drop_key_name(acdc_nu2, key_name="inversion_sigma")
        utils.trim_memory()

    if save_dir is not None:  # Write hypers and features to file
        with open(os.path.join(save_dir, f"hypers_{calc}.pickle"), "wb") as handle:
            pickle.dump(hypers, handle, protocol=pickle.HIGHEST_PROTOCOL)
        equistore.save(os.path.join(save_dir, f"range_{calc}.npz"), acdc_nu2)

    return acdc_nu2


def lambda_feature_kernel(lsoap_vector: TensorMap) -> TensorMap:
    """
    Takes a lambda-feature vector (i.e. lambda-SOAP or lambda-LODE) as a
    TensorMap and takes the relevant inner products to form a lambda-feature
    kernel, returned as a TensorMap.
    """
    return
