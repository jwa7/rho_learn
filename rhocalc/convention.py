"""
Module containing dummy TensorMaps with the naming convention used in rho_learn.
"""
import numpy as np
import metatensor
from metatensor import Labels, TensorBlock, TensorMap


COEFF_VECTOR = TensorMap(
    keys=Labels(
        names=[
            "spherical_harmonics_l",
            "species_center",
        ],
        values=np.array([[0, 0]]),
    ),
    blocks=[
        TensorBlock(
            values=np.array([0]).reshape(1, 1, 1),
            samples=Labels(
                names=["structure", "center"],
                values=np.array([[0, 0]]),
            ),
            components=[
                Labels(
                    names=[f"spherical_harmonics_m"],
                    values=np.array([[0]]),
                )
            ],
            properties=Labels(
                names=["n"],
                values=np.array([[0]]),
            ),
        )
    ],
)


OVERLAP_MATRIX = TensorMap(
    keys=Labels(
        names=[
            "spherical_harmonics_l1",
            "spherical_harmonics_l2",
            "species_center_1",
            "species_center_2",
        ],
        values=np.array([[0, 0, 0, 0]]),
    ),
    blocks=[
        TensorBlock(
            values=np.array([0]).reshape(1, 1, 1, 1),
            samples=Labels(
                names=["structure", "center_1", "center_2"],
                values=np.array([[0, 0, 0]]),
            ),
            components=[
                Labels(
                    names=[f"spherical_harmonics_m{c}"],
                    values=np.array([[0]]),
                )
                for c in [1, 2]
            ],
            properties=Labels(
                names=["n1", "n2"],
                values=np.array([[0, 0]]),
            ),
        )
    ],
)
