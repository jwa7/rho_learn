"""
Module containing functions to calculate error metrics on predictions
"""
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
