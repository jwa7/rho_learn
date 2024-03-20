"""
Module containing dummy TensorMaps with the naming convention used in rho_learn.
"""
import numpy as np
import metatensor as mts


COEFF_VECTOR = mts.TensorMap(
    keys=mts.Labels(
        names=[
            "spherical_harmonics_l",
            "species_center",
        ],
        values=np.array([[0, 0]]),
    ),
    blocks=[
        mts.TensorBlock(
            values=np.array([0]).reshape(1, 1, 1),
            samples=mts.Labels(
                names=["structure", "center"],
                values=np.array([[0, 0]]),
            ),
            components=[
                mts.Labels(
                    names=[f"spherical_harmonics_m"],
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
            "spherical_harmonics_l1",
            "spherical_harmonics_l2",
            "species_center_1",
            "species_center_2",
        ],
        values=np.array([[0, 0, 0, 0]]),
    ),
    blocks=[
        mts.TensorBlock(
            values=np.array([0]).reshape(1, 1, 1, 1),
            samples=mts.Labels(
                names=["structure", "center_1", "center_2"],
                values=np.array([[0, 0, 0]]),
            ),
            components=[
                mts.Labels(
                    names=[f"spherical_harmonics_m{c}"],
                    values=np.array([[0]]),
                )
                for c in [1, 2]
            ],
            properties=mts.Labels(
                names=["n1", "n2"],
                values=np.array([[0, 0]]),
            ),
        )
    ],
)
