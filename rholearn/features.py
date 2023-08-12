"""
Generates features vectors for equivariant structural representations.
Currently implemented:
    - lambda-SOAP
"""
import os
import pickle
from typing import Sequence, Optional

import ase
import numpy as np

import equistore
from equistore import Labels, TensorMap

import rascaline
from rascaline.utils import clebsch_gordan

from rholearn import io, spherical, utils


def lambda_feature_kernel(lsoap_vector: TensorMap) -> TensorMap:
    """
    Takes a lambda-feature vector (i.e. lambda-SOAP or lambda-LODE) as a
    TensorMap and takes the relevant inner products to form a lambda-feature
    kernel, returned as a TensorMap.
    """
    raise NotImplementedError



