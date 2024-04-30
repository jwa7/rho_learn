"""
Module containing dummy TensorMaps with the naming convention used in rho_learn.
"""

import numpy as np
import metatensor as mts


COEFF_VECTOR = mts.TensorMap(
    keys=mts.Labels(
        names=[
            "o3_lambda",
            "center_type",
        ],
        values=np.array([[0, 0]]),
    ),
    blocks=[
        mts.TensorBlock(
            values=np.array([0]).reshape(1, 1, 1),
            samples=mts.Labels(
                names=["system", "atom"],
                values=np.array([[0, 0]]),
            ),
            components=[
                mts.Labels(
                    names=[f"o3_mu"],
                    values=np.array([[0]]),
                )
            ],
            properties=mts.Labels(
                names=["n"],
                values=np.array([[0]]),
            ),
        )
    ],
)


OVERLAP_MATRIX = mts.TensorMap(
    keys=mts.Labels(
        names=[
            "o3_lambda_1",
            "o3_lambda_2",
            "center_1_type",
            "center_2_type",
        ],
        values=np.array([[0, 0, 0, 0]]),
    ),
    blocks=[
        mts.TensorBlock(
            values=np.array([0]).reshape(1, 1, 1, 1),
            samples=mts.Labels(
                names=["system", "atom_1", "atom_2"],
                values=np.array([[0, 0, 0]]),
            ),
            components=[
                mts.Labels(
                    names=[f"o3_mu_1"],
                    values=np.array([[0]]),
                ),
                mts.Labels(
                    names=[f"o3_mu_2"],
                    values=np.array([[0]]),
                ),
            ],
            properties=mts.Labels(
                names=["n_1", "n_2"],
                values=np.array([[0, 0]]),
            ),
        )
    ],
)
