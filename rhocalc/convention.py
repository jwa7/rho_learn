"""
Module containing dummy TensorMaps with the naming convention used in rho_learn.
"""
import metatensor.torch as mts


COEFF_VECTOR = mts.TensorMap(
    keys=mts.Labels(
        names=[
            "spherical_harmonics_l",
            "species_center",
        ],
        values=torch.tensor([[0, 0]]),
    ),
    blocks=[
        mts.TensorBlock(
            values=torch.tensor([0]).reshape(1, 1, 1),
            samples=mts.Labels(
                names=["structure", "center"],
                values=torch.tensor([[0, 0]]),
            ),
            components=[
                mts.Labels(
                    names=[f"spherical_harmonics_m"],
                    values=torch.tensor([[0]]),
                )
            ],
            properties=mts.Labels(
                names=["n"],
                values=torch.tensor([[0]]),
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
        values=torch.tensor([[0, 0, 0, 0]]),
    ),
    blocks=[
        mts.TensorBlock(
            values=torch.tensor([0]).reshape(1, 1, 1, 1),
            samples=mts.Labels(
                names=["structure", "center_1", "center_2"],
                values=torch.tensor([[0, 0, 0]]),
            ),
            components=[
                mts.Labels(
                    names=[f"spherical_harmonics_m{c}"],
                    values=torch.tensor([[0]]),
                )
                for c in [1, 2]
            ],
            properties=mts.Labels(
                names=["n1", "n2"],
                values=torch.tensor([[0, 0]]),
            ),
        )
    ],
)
