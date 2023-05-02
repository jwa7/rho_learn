"""
Module containing functions to calculate error metrics on predictions
"""
from typing import Optional
import numpy as np
import torch

import equistore
from equistore import TensorMap

from rholearn import utils


def absolute_error(input: TensorMap, target: TensorMap) -> float:
    """
    Calculates the absolute error between 2 :py:class:`TensorMap` objects.

    First, performs ``input`` minus ``target``. Then, takes the absolute values
    of these residuals , and sums them to get a single float value of the
    absolute error.
    """
    # Check metadata equivalence
    if not equistore.equal_metadata(input, target):
        raise ValueError("Input and target TensorMaps must have equal metadata")

    # Calculate the residuals and find the absolute values of them
    abs_diff = equistore.abs(equistore.subtract(input, target))

    # Sum the abs residuals
    abs_error = 0.0
    for key, block in abs_diff:
        vals = block.values
        if isinstance(vals, np.ndarray):
            abs_error += np.sum(vals)
        elif isinstance(vals, torch.Tensor):
            abs_error += torch.sum(vals).detach().numpy()
        else:
            raise TypeError(
                "TensorMap block values must be numpy.ndarray or torch.Tensor"
            )

    return abs_error


def mean_absolute_error(input: TensorMap, target: TensorMap) -> float:
    """
    Calculates the mean absolute error between 2 :py:class:`TensorMap` objects.

    First, performs ``input`` minus ``target``. Then, takes the absolute values
    of these residuals , and sums them to get a single float value of the
    absolute error. Divides this value by the number of elements.
    """
    return absolute_error(input, target) / utils.num_elements_tensormap(input)


def relative_errors_a_b(
    a: TensorMap, b: TensorMap, epsi: Optional[float] = None
) -> dict:
    """
    Takes 2 TensorMaps, ``a`` and ``b``, and returns a dictionary of the
    absolute flattened values of ``a`` and ``b``, as well as the relative errors
    between them. The returned dictionary consists of nested dicts for each l
    value, where the keys of the nested dicts are: ``a_vals``, ``b_vals`` and
    "rel_error".

    The relative error is calculated as:

        rel_error = | (a - b) / (a + b) |

    and is calcualated only for values of ``a`` and ``b`` that are both greater
    than ``epsi``.

    :param a: The first TensorMap
    :param b: The second TensorMap
    :param epsi: The threshold for values to include in the relative error
        calculation.
    """
    # Check that the metadata is the same
    assert equistore.equal_metadata(a, b)

    # Initialize the results dict
    results = {
        l: {"a_vals": np.array([]), "b_vals": np.array([]), "rel_error": np.array([])}
        for l in a.keys["spherical_harmonics_l"]
    }

    for key in a.keys:
        # Retrieve the l value and block values
        l = key["spherical_harmonics_l"]
        a_vals = a[key].values.flatten()
        b_vals = b[key].values.flatten()

        # Store the absolute flattened values of a and b
        results[l]["a_vals"] = np.concatenate([results[l]["a_vals"], np.abs(a_vals)])
        results[l]["b_vals"] = np.concatenate([results[l]["b_vals"], np.abs(b_vals)])

        # Keep values of of a and b that are both below epsilon to calculate the
        # relative error
        if epsi is not None:
            if epsi <= 0.0:
                raise ValueError("epsi must be a float greater than zero")
            above_epsi_idxs = np.union1d(
                np.where(a_vals > epsi), np.where(b_vals > epsi)
            )
            a_vals = a_vals[above_epsi_idxs]
            b_vals = b_vals[above_epsi_idxs]

        # Calculate the relative difference between the two blocks
        rel_error = np.abs((a_vals - b_vals) / (a_vals + b_vals))

        results[l]["rel_error"] = np.concatenate([results[l]["rel_error"], rel_error])

    return results
