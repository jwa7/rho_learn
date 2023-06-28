"""
A module containing helper functions specific to the example workflow.
"""
import os
import numpy as np

import equistore
from equistore import Labels, TensorMap


def recombine_coulomb_metrics(data_dir: str):
    """
    The TensorMap of coulomb matrices for the azoswitch workflow example is too
    large to store as a single file in the GitHub repository. As such, the
    TensorMap has been split into 3 smaller TensorMaps and saved under the names
    "coulomb_matrices_i.npz", where i = 0, 1, or 2, each comprised of 120
    blocks. This function recombines them into a single TensorMap, saves it to
    the same directory under the filename "coulomb_matrices_full.npz", and
    returns it.
    """
    # Load the individual TensorMaps into a list
    cm_list = [
        equistore.load(
            os.path.join(data_dir, "coulomb_metrics/", f"coulomb_metrics_{i}.npz")
        )
        for i in range(6)
    ]

    # Combine them
    combined_keys, combined_blocks = [], []
    for tensor in cm_list:
        for key, block in tensor.items():
            combined_keys.append([k for k in key])
            combined_blocks.append(block.copy())

    # Create a Labels object for the combined keys
    combined_keys = Labels(names=cm_list[0].keys.names, values=np.array(combined_keys))

    # Build the combined TensorMap and save to file
    combined_tensor = TensorMap(
        keys=combined_keys,
        blocks=combined_blocks,
    )
    equistore.save(os.path.join(data_dir, "coulomb_metrics.npz"), combined_tensor)

    return combined_tensor
