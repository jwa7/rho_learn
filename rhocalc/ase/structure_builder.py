import itertools
import ase
import ase.io
import ase.build
import numpy as np
import chemiscope



def perturb_slab(slab, noise_scale_factor: float = 0.25):
    """
    Applies Gaussian noise to all lattice positions of the input `slab`, except the
    bottom layer and its passivating Hydrogens. Assumes that the only Hydrogen present
    in the slab are the ones passivating the bottom layer.

    Returns the perturbed slab, leaving the positions of the input `slab` unchanged.

    `noise_scale_factor` scales the Gaussian noise applied to the unconstrained atoms.
    """
    perturbed_slab = slab.copy()

    # Identify the indices of the Si atoms on the bottom layer
    z_min = min(
        [
            position
            for position, symbol in zip(
                slab.positions[:, 2], perturbed_slab.get_chemical_symbols()
            )
            if symbol == "Si"
        ]
    )
    idxs_layer0 = np.where(np.abs(perturbed_slab.positions[:, 2] - z_min) < 0.1)[0]
    idxs_hydrogen = np.array(
        [i for i, sym in enumerate(perturbed_slab.get_chemical_symbols()) if sym == "H"]
    )

    idxs_to_perturb = [
        i
        for i in range(perturbed_slab.get_global_number_of_atoms())
        if i not in idxs_layer0 and i not in idxs_hydrogen
    ]

    for idx in idxs_to_perturb:
        perturbed_slab.positions[idx] += np.random.rand(*perturbed_slab.positions[idx].shape) * noise_scale_factor

    return perturbed_slab


def perturb_slab_constrain_bottom_layers(
    slab, constrained_slice_distance: float = None, noise_scale_factor: float = 0.25
):
    """
    Applies Gaussian noise to all lattice positions of the input `slab`, except those
    with a z-coordinate equal in the range [`z_min`, `z_min +
    constrained_slice_distance`], where `z_min` is the minimum z-coordinate in the
    system, and `constrained_slice_distance` is the thickness of the slab in the
    z-direction that should be constrained.

    Returns the perturbed slab, leaving the positions of the input `slab` unchanged.

    `noise_scale_factor` scales the Gaussian noise applied to the unconstrained atoms.
    """
    perturbed_slab = slab.copy()

    if constrained_slice_distance is None:
        idxs_constrained = []
    else:
        # Identify the indices of the Si atoms on the bottom layer
        z_min = np.min(perturbed_slab.positions[:, 2])
        idxs_constrained = np.where(perturbed_slab.positions[:, 2] <= z_min + constrained_slice_distance)[0]
        idxs_constrained = np.array(idxs_constrained)

    # Perturb the unconstrained atoms
    idxs_to_perturb = [
        i
        for i in range(perturbed_slab.get_global_number_of_atoms())
        if i not in idxs_constrained
    ]

    for idx in idxs_to_perturb:
        perturbed_slab.positions[idx] += np.random.rand(*perturbed_slab.positions[idx].shape) * noise_scale_factor

    return perturbed_slab


def adsorb_h2_on_slab(slab, height: float, lattice_param: float = 5.431020511):
    """
    Randomly places a H2 molecule above the top layer of the `slab` at  height
    that is a random pertubation of the chosen `height`.
    """
    # Build the H2 molecule: bond length is 0.74 Angstrom
    h2_bond = 0.74
    h2 = ase.Atoms("H2", positions=[[0, 0, 0], [h2_bond + 2 * np.random.rand(1)[0], 0, 0]])

    ase.build.add_adsorbate(
        slab,
        h2,
        position=np.random.rand(2) * lattice_param,  # randomnly places on cell surrface
        height=1.5 + np.random.rand(1)[0],  # height randomly perturbed from 1.5 above surface
    )

    return slab


def build_pristine_bulk_si(
    size: tuple, lattice_param: float = 5.431020511,
):
    bulk = ase.build.bulk("Si", crystalstructure="diamond", a=lattice_param)
    bulk = bulk.repeat((2, 2, 2))
    bulk.info["size"] = size
    bulk.info["lattice_param"] = lattice_param
    bulk.info["category"] = "pristine"

    return bulk
    

def build_pristine_passivated_si_slab(
    size: tuple, lattice_param: float = 5.431020511, passivate: bool = True
):
    slab = ase.build.diamond100("Si", size=size, vacuum=None, a=lattice_param)
    if passivate:
        slab = h_passivate_slab_underside(slab)

    slab.cell[2] = [0, 0, 100]

    slab.info["size"] = size
    slab.info["lattice_param"] = lattice_param
    slab.info["passivated"] = passivate
    slab.info["category"] = "pristine"
    slab.info["supercell"] = False

    return slab


def h_passivate_slab_underside(slab: ase.Atoms) -> ase.Atoms:

    # Identify the indices of the atoms on the bottom layer
    z_min = min(slab.positions[:, 2])
    idxs_layer0 = np.where(np.abs(slab.positions[:, 2] - z_min) < 0.1)[0]

    si_h_bond = 1.476
    h_si_h_angle = 170
    y_displacement = si_h_bond * np.arcsin(np.pi * h_si_h_angle / (2 * 360))
    z_displacement = si_h_bond * np.arccos(np.pi * h_si_h_angle / (2 * 360))

    # Add 2 Hydrogens to passivate each Si atom on the bottom layer
    for idx in idxs_layer0:
        # Calculate the positions of the Hydrogens
        h1_position = slab[idx].position - [0, +y_displacement, z_displacement]
        h2_position = slab[idx].position - [0, -y_displacement, z_displacement]

        # Add adatoms
        slab.append(ase.Atom("H", position=h1_position))
        slab.append(ase.Atom("H", position=h2_position))

    return slab


def make_slab_topside_dimer(
    slab: ase.Atoms,
    desired_dimer_distance: float = 2.24,
    buckled=None,
    aligned=True,
) -> ase.Atoms:
    # Identify the indices of the atoms on the top layer
    z_max = max(slab.positions[:, 2])
    idxs_top_layer = np.where(np.abs(slab.positions[:, 2] - z_max) < 0.1)[0]
    assert len(idxs_top_layer) == 2

    # There should be a 'square' of atoms on this layer. Move the y-coodinates closer
    # together.
    idx1, idx2 = idxs_top_layer[0], idxs_top_layer[1]

    current_dimer_distance = np.abs(slab[idx1].position[1] - slab[idx2].position[1])
    slab[idx1].position[1] += (current_dimer_distance - desired_dimer_distance) / 2
    slab[idx2].position[1] -= (current_dimer_distance - desired_dimer_distance) / 2

    if buckled:
        slab[idx1].position[1] += buckled
        slab[idx2].position[1] -= buckled
        slab[idx1].position[2] += buckled
        slab[idx2].position[2] -= buckled

    return slab


def make_slab_topside_dimer_pair(
    slab: ase.Atoms,
    desired_dimer_distance: float = 2.24,
    buckled=None,
    aligned=True,
) -> ase.Atoms:
    # Identify the indices of the atoms on the top layer
    z_max = max(slab.positions[:, 2])
    idxs_top_layer = np.where(np.abs(slab.positions[:, 2] - z_max) < 0.1)[0]
    assert len(idxs_top_layer) == 4

    # There should be a 'square' of atoms on this layer. Move the y-coodinates closer
    # together.
    dimer_pair_idxs = [
        (idxs_top_layer[0], idxs_top_layer[2]),
        (idxs_top_layer[1], idxs_top_layer[3]),
    ]
    if buckled is not None:
        if aligned:
            tilt_factors = [1, 1]
        else:
            tilt_factors = [1, -1]
    else:
        tilt_factors = [0, 0]
    for (idx1, idx2), tilt_factor in zip(dimer_pair_idxs, tilt_factors):
        current_dimer_distance = np.abs(slab[idx1].position[1] - slab[idx2].position[1])
        slab[idx1].position[1] += (current_dimer_distance - desired_dimer_distance) / 2
        slab[idx2].position[1] -= (current_dimer_distance - desired_dimer_distance) / 2

        if buckled:
            slab[idx1].position[1] += buckled
            slab[idx2].position[1] -= buckled
            slab[idx1].position[2] += (buckled * tilt_factor)
            slab[idx2].position[2] -= (buckled * tilt_factor)

    return slab