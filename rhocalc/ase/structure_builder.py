import itertools
import ase
import ase.io
import ase.build
import numpy as np
import chemiscope


def build_pristine_passivated_si_slab(size: tuple, lattice_param: float = 5.431020511):
    """
    Builds a Si diamond (100) slab of the chosen `size` with `lattice_param` a
    = 5.431020511 (default for Si), and passivates each Si in the bottom layer with 2 Hydrogen
    atoms.
    """
    # Build a Si(100) slab
    slab = ase.build.diamond100("Si", size=size, vacuum=None, a=lattice_param)

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


def perturb_slab(slab):
    """
    Applies Gaussian noise to all lattice positions of the input `slab`, except
    the bottom layer and its passivating Hydrogens. Assumes that the only
    Hydrogen present in the slab are the ones passivating the bottom layer.
    """
    # Identify the indices of the Si atoms on the bottom layer
    z_min = min(
        [
            position
            for position, symbol in zip(
                slab.positions[:, 2], slab.get_chemical_symbols()
            )
            if symbol == "Si"
        ]
    )
    idxs_layer0 = np.where(np.abs(slab.positions[:, 2] - z_min) < 0.1)[0]
    idxs_hydrogen = np.array(
        [i for i, sym in enumerate(slab.get_chemical_symbols()) if sym == "H"]
    )

    idxs_to_perturb = [
        i
        for i in range(slab.get_global_number_of_atoms())
        if i not in idxs_layer0 and i not in idxs_hydrogen
    ]

    for idx in idxs_to_perturb:
        slab.positions[idx] += np.random.rand(*slab.positions[idx].shape) * 0.25

    return slab


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


if __name__ == "__main__":
        

    # Define sizes of slabs
    sizes = [(2, 2, 4), (2, 2, 6), (2, 2, 8), (3, 3, 4)]

    # For each size, generate 90 perturbed slabs
    frames = []
    for size in sizes:
        pristine_slab = build_pristine_passivated_si_slab(size)
        # For each size, generate 100 structures
        for i in range(100):
            # Slab 0: pristine, no adsorbate
            if i == 0:
                slab = pristine_slab.copy()
            # Slabs 1 - 10: pristine, with adsorbate
            elif 0 < i < 10:
                slab = adsorb_h2_on_slab(pristine_slab.copy(), 2)
            # Slabs 11 - 40: perturbed, no adsorbate
            elif 10 <= i < 40:
                slab = perturb_slab(pristine_slab.copy())
            else:
                assert 40 <= i < 100
                slab = adsorb_h2_on_slab(perturb_slab(pristine_slab.copy()), 2)
            frames.append(slab)


    for frame in frames:
        frame.pbc = True
        frame.cell[2] = [0, 0, 100]
        frame.positions[:, 2] += 5

    # Save
    ase.io.write("sih2_small.xyz", frames)
    ase.io.write("sih2_small.cif", frames[:10])
    print("Num frames:", len(frames))