"""
Functions to process results data from ML training.
"""
import os
from typing import List

import numpy as np


def compile_loss_data(run_dir: str, exercises: List[int], subsets: List[int]) -> dict:
    """
    Reads the losses.npz files in each of the training subdirectories in parent
    directory ``run_dir`` and compiles them into a nested dict of arrays, where
    each array is accessed by compiled_loss_data[exercise_i][subset_j][<str>],
    where <str> is "train" or "test" and ``compiled_loss_data`` is the return
    dict.
    """
    compiled_train_losses, compiled_test_losses = {}, {}
    for exercise in exercises:
        ex_dict_train, ex_dict_test = {}, {}
        for subset in subsets:
            losses = np.load(
                os.path.join(
                    run_dir, f"exercise_{exercise}", f"subset_{subset}", "losses.npz"
                )
            )
            ex_dict_train[subset] = losses["train"]
            ex_dict_test[subset] = losses["test"]
        compiled_train_losses[exercise] = ex_dict_train
        compiled_test_losses[exercise] = ex_dict_test
    return compiled_train_losses, compiled_test_losses


def average_losses(compiled_losses: dict) -> dict:
    """
    Takes a dict of compiled loss arrays for each learning exercise and training
    subset (i.e. as produced by the ``compile_loss_data`` function) and averages
    over the learning exercises. Returns a dict of similar nested structure to
    ``compiled_losses``, but with the exercise layer of keys removed, such that
    each array in the output dict is accessed by averaged_losses[subset_j].
    """
    exercises = compiled_losses.keys()
    subsets = list(compiled_losses.values())[0].keys()
    averaged_losses = {
        subset: np.mean(
            [compiled_losses[exercise][subset] for exercise in exercises], axis=0
        )
        for subset in subsets
    }

    return averaged_losses
