"""
Module for user-defined functions.
"""
import os
import shutil
from typing import Callable, List, Optional

import ase
import numpy as np

import metatensor
import rascaline
from metatensor import Labels, TensorMap
from rascaline.utils import clebsch_gordan

from rhocalc import convert
from rhocalc.aims import aims_calc, aims_parser



