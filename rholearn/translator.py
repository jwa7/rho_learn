"""
Module for performing data format translations between various packages and
rholearn.
"""
import itertools
import numpy as np

import equistore
from equistore import Labels, TensorBlock, TensorMap

from rholearn import utils

# TODO:
# - check correct ordering of properties

SYM_TO_NUM = {"H": 1, "C": 6, "O": 8, "N": 7}
NUM_TO_SYM = {1: "H", 6: "C", 8: "O", 7: "N"}


def salted_coeffs_to_tensormap(
    frames: list,
    coeffs: np.ndarray,
    lmax: dict,
    nmax: dict,
) -> TensorMap:
    """
    Convert the flat vector of SALTED electron density coefficients to a
    TensorMap. Assumes the order of the coefficients, for a given molecule, is:

    .. math::

        c_v = \sum_{i \in atoms(A)} \sum_{l}^{l_max} \sum_{n}^{n_max} \sum_{m
        \in [-l, +l]} c^i_{lnm}

    :param frames: List of ASE Atoms objects for each xyz structure in the data
        set.
    :param coeffs: np.ndarray of shape (n_structures, n_coefficients), where
        each flat vector of coefficients for a single structure is ordered with
        indices iterating over i, l, n, m, in that nested order - as in the
        equation above.
    :param lmax: dict of maximum l values for each species in the data set that
        wass used to construct the projections passed in `coeffs`. Chemical
        symbols are the keys, and the values are the maximum l values. For
        instance, if the data set contains only oxygen and hydrogen atoms, a
        valid lmax dict would be: {"H": 4, "O": 8}.
    :param nmax: dict of maximum n values for each species and l values. For
        instance, {('H', 0): 8, ('H', 1): 6, ('H', 2): 6, ... ('O', 0): 9, ('O',
        1): 10, ('O', 2): 9, ...}. The keys are tuples of the form (symbol, l),
        and the values are the maximum n values for that species and l value.

    :return: TensorMap of the coefficients, with the appropriate indices named
        as follows: Keys: l -> "spherical_harmonics_l", a -> "species_center".
        Samples: A -> "structure", i -> "center". Components: m ->
        "spherical_harmonics_m". Properties: n -> "n".
    """

    # Define some useful constants
    n_frames = len(frames)
    n_atoms_per_frame = len(frames[0].get_positions())
    species_symbols = np.array(frames[0].get_chemical_symbols())

    # Define some useful dimensions
    n_coeffs_by_l = {
        symbol: [(2 * l + 1) * nmax[(symbol, l)] for l in range(lmax[symbol] + 1)]
        for symbol in species_symbols
    }
    n_coeffs_by_symbol = {
        symbol: np.sum(n_coeffs_by_l[symbol]) for symbol in species_symbols
    }
    n_coeffs_per_atom = np.array(
        [n_coeffs_by_symbol[symbol] for symbol in species_symbols]
    )
    n_coeffs_per_frame = np.sum(n_coeffs_per_atom)
    assert coeffs.shape == (n_frames, n_coeffs_per_frame)

    # Split the projections into a list of arrays, one for each atom. Go up to
    # -1 to remove the last empty array, created by numpy.split when using exact
    # indices.
    split_by_atom = np.split(coeffs, np.cumsum(n_coeffs_per_atom), axis=1)[:-1]
    assert len(split_by_atom) == n_atoms_per_frame

    # Split each of these arrays into a list of arrays, one for each l
    tmp_atomic = []
    for symbol, atomic_array in zip(species_symbols, split_by_atom):
        assert atomic_array.shape == (n_frames, n_coeffs_by_symbol[symbol])

        split_by_l = np.split(atomic_array, np.cumsum(n_coeffs_by_l[symbol]), axis=1)[
            :-1
        ]
        assert lmax[symbol] + 1 == len(split_by_l)

        # Reshape the arrays such that the m-components are along the 1st axis
        tmp_l = []
        for l, l_array in zip(range(lmax[symbol] + 1), split_by_l):
            # Don't need to check the shape here as np.reshape implicitly does that
            # for us.
            tmp_l.append(l_array.reshape(n_frames, 2 * l + 1, nmax[(symbol, l)]))

        tmp_atomic.append(tmp_l)

    split_by_atom = tmp_atomic

    # Initialize dict of the form {(l, symbol): array}
    results_dict = {}
    for symbol in np.unique(species_symbols):
        for l in range(lmax[symbol] + 1):
            results_dict[(l, symbol)] = {}

    # Store the arrays by l and species
    for atom_idx, (symbol, atomic_array) in enumerate(
        zip(species_symbols, split_by_atom)
    ):
        for l, l_array in zip(range(lmax[symbol] + 1), atomic_array):
            results_dict[(l, symbol)][atom_idx] = l_array

    # Contruct the TensorMap
    keys = Labels(
        names=["spherical_harmonics_l", "species_center"],
        values=np.array([[l, SYM_TO_NUM[symbol]] for l, symbol in results_dict.keys()]),
    )

    blocks = []
    for l, species_center in keys:
        symbol = NUM_TO_SYM[species_center]
        raw_block = results_dict[(l, symbol)]
        atom_idxs = np.sort(list(raw_block.keys()))
        blocks.append(
            TensorBlock(
                samples=Labels(
                    names=["structure", "center"],
                    values=np.array(
                        [
                            [A, i]
                            for i, A in itertools.product(atom_idxs, range(n_frames))
                        ]
                    ),
                ),
                components=[
                    Labels(
                        names=["spherical_harmonics_m"],
                        values=np.arange(-l, +l + 1).reshape(-1, 1),
                    ),
                ],
                properties=Labels(
                    names=["n"], values=np.arange(nmax[(symbol, l)]).reshape(-1, 1)
                ),
                values=np.concatenate(
                    [raw_block[atom_idx] for atom_idx in atom_idxs], axis=0
                ),
            )
        )
    tensor = TensorMap(keys=keys, blocks=blocks)
    assert utils.num_elements_tensormap(tensor) == np.prod(coeffs.shape)
    return tensor


def salted_overlaps_to_tensormap(
    frames: list,
    overlaps: np.ndarray,
    lmax: dict,
    nmax: dict,
) -> TensorMap:
    """
    Convert the flat vector of SALTED electron density coefficients to a
    TensorMap. Assumes the order of the coefficients, for a given molecule, is:

    .. math::


    :param frames: List of ASE Atoms objects for each xyz structure in the data
        set.
    :param overlaps: np.ndarray of shape (n_structures, n_coefficients,
        n_coefficients), where each 2D-matrix of overlap elements for a single
        structure is ordered with indices iterating over i1, l1, n1, m1, along
        one axis and i2, l2, n2, m2, in that nested order - as in the equation
        above.
    :param basis_str: str of the basis set that the density has been expanded
        onto, e.g. "RI-cc-pvqz".

    :return: TensorMap of the density basis set overlap matrix elements, with
        the appropriate indices named as follows: Keys: l1 ->
        "spherical_harmonics_l1", l2 -> "spherical_harmonics_l2", a1 ->
        "species_center_1", a2 -> "species_center_2". Samples: A -> "structure",
        i1 -> "center_1", i2 -> "center_2". Components: m1 ->
        "spherical_harmonics_m2", m1 -> "spherical_harmonics_m2". Properties: n1
        -> "n1", n2 -> "n2".
    """

    # Define some useful constants
    n_frames = len(frames)
    n_atoms_per_frame = len(frames[0].get_positions())
    species_symbols = np.array(frames[0].get_chemical_symbols())

    # Define some useful dimensions
    n_coeffs_by_l = {
        symbol: [(2 * l + 1) * nmax[(symbol, l)] for l in range(lmax[symbol] + 1)]
        for symbol in species_symbols
    }
    n_coeffs_by_symbol = {
        symbol: np.sum(n_coeffs_by_l[symbol]) for symbol in species_symbols
    }
    n_coeffs_per_atom = np.array(
        [n_coeffs_by_symbol[symbol] for symbol in species_symbols]
    )
    n_coeffs_per_frame = np.sum(n_coeffs_per_atom)

    assert overlaps.shape == (n_frames, n_coeffs_per_frame, n_coeffs_per_frame)

    # Split the overlaps into a list of matrices along axis 1, one for each atom
    split_by_i1 = np.split(overlaps, np.cumsum(n_coeffs_per_atom), axis=1)[:-1]
    assert len(split_by_i1) == n_atoms_per_frame

    # Split the overlaps into a list of matrices along axis 2, one for each atom
    tmp_i1 = []
    for i1, i1_matrix in enumerate(split_by_i1):
        assert i1_matrix.shape == (
            n_frames,
            n_coeffs_by_symbol[species_symbols[i1]],
            n_coeffs_per_frame,
        )

        # Split along the axis corresponding to the second atom
        split_by_i2 = np.split(i1_matrix, np.cumsum(n_coeffs_per_atom), axis=2)[:-1]
        assert len(split_by_i2) == n_atoms_per_frame

        # Split by l1
        tmp_i2 = []
        for i2, i2_matrix in enumerate(split_by_i2):
            assert i2_matrix.shape == (
                n_frames,
                n_coeffs_by_symbol[species_symbols[i1]],
                n_coeffs_by_symbol[species_symbols[i2]],
            )
            split_by_l1 = np.split(
                i2_matrix, np.cumsum(n_coeffs_by_l[species_symbols[i1]]), axis=1
            )[:-1]
            assert len(split_by_l1) == lmax[species_symbols[i1]] + 1

            # Split by l2
            tmp_l1 = []
            for l1, l1_matrix in enumerate(split_by_l1):
                assert l1_matrix.shape == (
                    n_frames,
                    n_coeffs_by_l[species_symbols[i1]][l1],
                    n_coeffs_by_symbol[species_symbols[i2]],
                )
                split_by_l2 = np.split(
                    l1_matrix, np.cumsum(n_coeffs_by_l[species_symbols[i2]]), axis=2
                )[:-1]
                assert len(split_by_l2) == lmax[species_symbols[i2]] + 1

                # Now reshape the matrices such that the m-components are expanded
                tmp_l2 = []
                for l2, l2_matrix in enumerate(split_by_l2):
                    assert l2_matrix.shape == (
                        n_frames,
                        n_coeffs_by_l[species_symbols[i1]][l1],
                        n_coeffs_by_l[species_symbols[i2]][l2],
                    )

                    l2_matrix = l2_matrix.reshape(
                        n_frames,
                        2 * l1 + 1,
                        nmax[(species_symbols[i1], l1)],
                        n_coeffs_by_l[species_symbols[i2]][l2],
                    )

                    l2_matrix = l2_matrix.reshape(
                        n_frames,
                        2 * l1 + 1,
                        nmax[(species_symbols[i1], l1)],
                        2 * l2 + 1,
                        nmax[(species_symbols[i2], l2)],
                    )
                    # Reorder the axes such that m-components are in the
                    # intermediate dimensions
                    l2_matrix = np.swapaxes(l2_matrix, 2, 3)

                    # Reshape matrix such that both n1 and n2 are along the final
                    # axes
                    l2_matrix = l2_matrix.reshape(
                        n_frames,
                        2 * l1 + 1,
                        2 * l2 + 1,
                        nmax[(species_symbols[i1], l1)]
                        * nmax[(species_symbols[i2], l2)],
                    )

                    assert l2_matrix.shape == (
                        n_frames,
                        2 * l1 + 1,
                        2 * l2 + 1,
                        nmax[(species_symbols[i1], l1)]
                        * nmax[(species_symbols[i2], l2)],
                    )
                    tmp_l2.append(l2_matrix)
                tmp_l1.append(tmp_l2)
            tmp_i2.append(tmp_l1)
        tmp_i1.append(tmp_i2)
    split_by_i1 = tmp_i1

    # Initialize dict of the form {(l, symbol): array}
    results_dict = {}
    for symbol_1, symbol_2 in itertools.product(np.unique(species_symbols), repeat=2):
        for l1 in range(lmax[symbol_1] + 1):
            for l2 in range(lmax[symbol_2] + 1):
                results_dict[(l1, l2, symbol_1, symbol_2)] = {}

    # Store the arrays by l1, l2, species
    for i1, (symbol_1, i1_matrix) in enumerate(zip(species_symbols, split_by_i1)):
        for i2, (symbol_2, i2_matrix) in enumerate(zip(species_symbols, i1_matrix)):
            for l1, l1_matrix in zip(range(lmax[symbol_1] + 1), i2_matrix):
                for l2, l2_matrix in zip(range(lmax[symbol_2] + 1), l1_matrix):
                    results_dict[(l1, l2, symbol_1, symbol_2)][i1, i2] = l2_matrix

    # Contruct the TensorMap
    keys = Labels(
        names=[
            "spherical_harmonics_l1",
            "spherical_harmonics_l2",
            "species_center_1",
            "species_center_2",
        ],
        values=np.array(
            [
                [l1, l2, SYM_TO_NUM[symbol_1], SYM_TO_NUM[symbol_2]]
                for l1, l2, symbol_1, symbol_2 in results_dict.keys()
            ]
        ),
    )

    blocks = []
    for l1, l2, species_center_1, species_center_2 in keys:

        # Get the chemical symbols for the corresponding atomic numbers
        symbol_1 = NUM_TO_SYM[species_center_1]
        symbol_2 = NUM_TO_SYM[species_center_2]

        # Retrieve the raw block of data for the l and symbols
        raw_block = results_dict[(l1, l2, symbol_1, symbol_2)]

        # Get the atomic indices
        atom_idxs = np.array(list(raw_block.keys()))

        # Build the TensorBlock
        values = np.concatenate([raw_block[i1, i2] for i1, i2 in atom_idxs], axis=0)
        block = TensorBlock(
            samples=Labels(
                names=["structure", "center_1", "center_2"],
                values=np.array(
                    [
                        [A, i1, i2]
                        for (i1, i2), A in itertools.product(atom_idxs, range(n_frames))
                    ]
                ),
            ),
            components=[
                Labels(
                    names=["spherical_harmonics_m1"],
                    values=np.arange(-l1, +l1 + 1).reshape(-1, 1),
                ),
                Labels(
                    names=["spherical_harmonics_m2"],
                    values=np.arange(-l2, +l2 + 1).reshape(-1, 1),
                ),
            ],
            properties=Labels(
                names=["n1", "n2"],
                values=np.array(
                    [
                        [n1, n2]
                        for n1, n2 in itertools.product(
                            np.arange(nmax[(symbol_1, l1)]),
                            np.arange(nmax[(symbol_2, l2)]),
                        )
                    ]
                ),
            ),
            values=values,
        )
        blocks.append(block)

    # Build TensorMap and check num elements same as input
    tensor = TensorMap(keys=keys, blocks=blocks)
    assert utils.num_elements_tensormap(tensor) == np.prod(overlaps.shape)

    return tensor


def overlap_is_symmetric(tensor: TensorMap) -> bool:
    """
    Returns true if the overlap matrices stored in TensorMap form are symmetric,
    false otherwise. Assumes the following data strutcure of the TensorMap:
    - keys: spherical_harmonics_l1, spherical_harmonics_l2, species_center_1,
    species_center_2
    - blocks: samples: structure, center_1, center_2
              components: [spherical_harmonics_m1, spherical_harmonics_m2]
              properties: n1, n2
    """
    keys = tensor.keys
    checked = set()
    # Iterate over the keys
    for key in keys:
        l1, l2, a1, a2 = key
        if (l1, l2, a1, a2) in checked or (l2, l1, a2, a1) in checked:
            continue

        # Get 2 blocks for comparison, permuting l and a. If this is a diagonal
        # block, block 1 and block 2 will be the same object
        block1 = tensor.block(
            spherical_harmonics_l1=l1,
            spherical_harmonics_l2=l2,
            species_center_1=a1,
            species_center_2=a2,
        )
        block2 = tensor.block(
            spherical_harmonics_l1=l2,
            spherical_harmonics_l2=l1,
            species_center_1=a2,
            species_center_2=a1,
        )
        if not overlap_is_symmetric_block(block1, block2):
            return False
        # Only track checking of off-diagonal blocks
        if not (l1 == l2 and a1 == a2):
            checked.update({(l1, l2, a1, a2)})

    return True


def overlap_is_symmetric_block(block1: TensorBlock, block2: TensorBlock) -> bool:
    """
    Returns true if the overlap matrices stored in the input blocks are
    symmetric,
    false otherwise. Assumes the following data structure of the TensorMap:
    """
    # Create a samples filter for how the samples map to eachother between blocks
    samples_filter = [
        block2.samples.position((A, i2, i1)) for [A, i1, i2] in block1.samples
    ]
    # Create a properties filter for how the properties map to eachother between
    # blocks
    properties_filter = [
        block2.properties.position((n2, n1)) for [n1, n2] in block1.properties
    ]
    # Broadcast the values array using the filters, and swap the components axes
    reordered_block2 = np.swapaxes(
        block2.values[samples_filter][..., properties_filter], 1, 2
    )

    return np.all(block1.values == reordered_block2)


def drop_off_diagonal_blocks(tensor: TensorMap) -> TensorMap:
    """
    Given an input TensorMap with keys (l1, l2, a1, a2), returns a new TensorMap
    where off-diagonal blocks (i.e. where (l1 != l2 or a1 != a2) indexed by (l1,
    l2, a1, a2) are kept but blocks indexed by (l2, l1, a2, a1) are dropped.
    Diagonal blocks (i.e. where (l1 == l2 and a1 == a2) are kept.
    """
    keys = tensor.keys
    keys_to_drop = set()

    for key in keys:
        l1, l2, a1, a2 = key
        if (l1, l2, a1, a2) in keys_to_drop or (l2, l1, a2, a1) in keys_to_drop:
            continue

        # Don't drop keys for diagonal blocks
        if not (l1 == l2 and a1 == a2):
            keys_to_drop.update({(l1, l2, a1, a2)})

    new_tensor = utils.drop_blocks(
        tensor,
        keys=Labels(
            names=tensor.keys.names,
            values=np.array([np.array(k) for k in keys_to_drop]),
        ),
    )
    K_old = len(tensor.keys)
    K_new = len(new_tensor.keys)
    assert K_new == np.sqrt(K_old) / 2 * (np.sqrt(K_old) + 1)

    return new_tensor


def drop_redundant_samples_diagonal_blocks(tensor: TensorMap) -> TensorMap:
    """
    Given an input TensorMap with keys (l1, l2, a1, a2), returns a new TensorMap
    where diagonal blocks have redundant samples dropped. For instance, for
    blocks where l1 = l2 and a1 = a2, samples of the form (A, i1, i2) are kept
    but samples of the form (A, i2, i1) are dropped.
    """
    keys = tensor.keys
    assert np.all(
        keys.names
        == (
            "spherical_harmonics_l1",
            "spherical_harmonics_l2",
            "species_center_1",
            "species_center_2",
        )
    )
    blocks = []
    for key in keys:

        # Unpack the key
        l1, l2, a1, a2 = key

        # If an off-diagonal block, just store it
        if not (l1 == l2 and a1 == a2):
            blocks.append(tensor[key].copy())
            continue

        # Otherwise, manipulate the samples of the diagonal block
        block = tensor[key]

        # Create a samples filter
        samples_filter = []
        s_checked = set()
        for sample in block.samples:
            A, i1, i2 = sample
            if (A, i1, i2) in s_checked or (A, i2, i1) in s_checked:
                samples_filter.append(False)
            else:
                samples_filter.append(True)
                s_checked.add((A, i1, i2))

        # Create and store the new block
        blocks.append(
            TensorBlock(
                values=block.values[samples_filter],
                samples=block.samples[samples_filter],
                components=block.components,
                properties=block.properties,
            )
        )
    return TensorMap(keys=keys, blocks=blocks)
