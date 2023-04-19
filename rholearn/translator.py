"""
Module for performing data format translations between various packages and
rholearn.
"""
import itertools
import numpy as np

import ase

import equistore
from equistore import Labels, TensorBlock, TensorMap

from rholearn import utils


SYM_TO_NUM = {"H": 1, "C": 6, "O": 8, "N": 7}
NUM_TO_SYM = {1: "H", 6: "C", 8: "O", 7: "N"}


def get_flat_index(
    symbol_list: list, lmax: dict, nmax: dict, i: int, l: int, n: int, m: int
) -> int:
    """
    Get the flat index of the coefficient pointed to by the basis function
    indices.

    :param lmax : A dictionary containing the maximum spherical harmonics (l)
        value for each atom type.
    :param nmax: A dictionary containing the maximum radial channel (n) value
        for each combination of atom type and l.
    :param i: int, the atom index.
    :param l: int, the spherical harmonics index.
    :param n: int, the radial channel index.
    :param m: int, the spherical harmonics component index.
    :param tests: int, the number of coefficients to randomly compare between
        the raw input array and processed TensorMap to check for correct
        conversion.

    :return int: the flat index of the coefficient pointed to by the input
        indices.
    """
    # Define the atom offset
    atom_offset = 0
    for atom_index in range(i):
        symbol = symbol_list[atom_index]
        for l_tmp in range(lmax[symbol] + 1):
            atom_offset += (2 * l_tmp + 1) * nmax[(symbol, l_tmp)]

    # Define the l offset
    l_offset = 0
    symbol = symbol_list[i]
    for l_tmp in range(l):
        l_offset += (2 * l_tmp + 1) * nmax[(symbol, l_tmp)]

    # Define the n offset
    n_offset = (2 * l + 1) * n

    # Define the m offset
    m_offset = m + l

    return atom_offset + l_offset + n_offset + m_offset


def coeff_vector_to_tensormap(
    frame: ase.Atoms,
    coeffs: np.ndarray,
    structure_idx: int,
    lmax: dict,
    nmax: dict,
    tests: int = 0,
) -> TensorMap:
    """
    Convert a set of coefficients to a TensorMap.
    """
    # Define some useful variables
    symbols = frame.get_chemical_symbols()
    uniq_symbols = np.unique(symbols)
    tot_atoms = len(symbols)
    tot_coeffs = len(coeffs)

    # First, confirm the length of the flat array is as expected
    num_coeffs_by_uniq_symbol = {}
    for symbol in uniq_symbols:
        n_coeffs = 0
        for l_tmp in range(lmax[symbol] + 1):
            for n_tmp in range(nmax[(symbol, l_tmp)]):
                n_coeffs += 2 * l_tmp + 1
        num_coeffs_by_uniq_symbol[symbol] = n_coeffs

    num_coeffs_by_symbol = np.array(
        [num_coeffs_by_uniq_symbol[symbol] for symbol in symbols]
    )
    assert np.sum(num_coeffs_by_symbol) == tot_coeffs

    # Split the flat array by atomic index
    split_by_atom = np.split(coeffs, np.cumsum(num_coeffs_by_symbol), axis=0)[:-1]
    assert len(split_by_atom) == tot_atoms
    assert np.sum([len(x) for x in split_by_atom]) == tot_coeffs

    num_coeffs_by_l = {
        symbol: np.array(
            [
                (2 * l_tmp + 1) * nmax[(symbol, l_tmp)]
                for l_tmp in range(lmax[symbol] + 1)
            ]
        )
        for symbol in uniq_symbols
    }
    for symbol in uniq_symbols:
        assert np.sum(num_coeffs_by_l[symbol]) == num_coeffs_by_uniq_symbol[symbol]

    # Split each atom array by angular momentum channel
    new_split_by_atom = []
    for symbol, atom_arr in zip(symbols, split_by_atom):
        split_by_l = np.split(atom_arr, np.cumsum(num_coeffs_by_l[symbol]), axis=0)[:-1]
        assert len(split_by_l) == lmax[symbol] + 1

        new_split_by_l = []
        for l_tmp, l_arr in enumerate(split_by_l):
            assert len(l_arr) == nmax[(symbol, l_tmp)] * (2 * l_tmp + 1)

            # Reshape to have components and properties on the 2nd and 3rd axes.
            # IMPORTANT: Fortran order!
            l_arr = l_arr.reshape((1, 2 * l_tmp + 1, nmax[(symbol, l_tmp)]), order="F")
            new_split_by_l.append(l_arr)

        new_split_by_atom.append(new_split_by_l)

    # Create a dict to store the arrays by l and species
    results_dict = {}
    for symbol in uniq_symbols:
        for l_tmp in range(lmax[symbol] + 1):
            results_dict[(l_tmp, symbol)] = {}

    for i_tmp, (symbol, atom_arr) in enumerate(zip(symbols, new_split_by_atom)):
        for l_tmp, l_array in zip(range(lmax[symbol] + 1), atom_arr):
            results_dict[(l_tmp, symbol)][i_tmp] = l_array

    # Build the TensorMap keys
    keys = Labels(
        names=["spherical_harmonics_l", "species_center"],
        values=np.array([[l, SYM_TO_NUM[symbol]] for l, symbol in results_dict.keys()]),
    )

    # Build the TensorMap blocks
    blocks = []
    for l, species_center in keys:
        symbol = NUM_TO_SYM[species_center]
        raw_block = results_dict[(l, symbol)]
        atom_idxs = np.sort(list(raw_block.keys()))
        block = TensorBlock(
            samples=Labels(
                names=["structure", "center"],
                values=np.array([[structure_idx, i] for i in atom_idxs]),
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
            values=np.ascontiguousarray(
                np.concatenate([raw_block[i] for i in atom_idxs], axis=0)
            ),
        )
        assert block.values.shape == (len(atom_idxs), 2 * l + 1, nmax[(symbol, l)])
        blocks.append(block)

    # Construct TensorMap
    tensor = TensorMap(keys=keys, blocks=blocks)

    # Check number of elements
    assert utils.num_elements_tensormap(tensor) == tot_coeffs

    # Check values of 1000 test coefficients
    for _ in range(tests):
        if not test_coeff_vector_conversion(
            frame, structure_idx, lmax, nmax, coeffs, tensor
        ):
            raise ValueError("Conversion test failed.")

    return tensor


def overlap_matrix_to_tensormap(
    frame: ase.Atoms,
    overlaps: np.ndarray,
    structure_idx: int,
    lmax: dict,
    nmax: dict,
    tests: int = 0,
) -> TensorMap:
    """ """
    # Define some useful variables
    symbols = frame.get_chemical_symbols()
    uniq_symbols = np.unique(symbols)
    tot_atoms = len(symbols)
    tot_coeffs = overlaps.shape[0]
    tot_overlaps = np.prod(overlaps.shape)

    # First, confirm the length of each side of square matrix array is as expected
    num_coeffs_by_uniq_symbol = {}
    for symbol in uniq_symbols:
        n_coeffs = 0
        for l_tmp in range(lmax[symbol] + 1):
            for n_tmp in range(nmax[(symbol, l_tmp)]):
                n_coeffs += 2 * l_tmp + 1
        num_coeffs_by_uniq_symbol[symbol] = n_coeffs

    num_coeffs_by_symbol = np.array(
        [num_coeffs_by_uniq_symbol[symbol] for symbol in symbols]
    )
    assert np.sum(num_coeffs_by_symbol) == tot_coeffs
    assert overlaps.shape == (tot_coeffs, tot_coeffs)

    # Define the number of coefficients for each respective l value (0,1,...)
    # for each unique atomic symbol
    num_coeffs_by_l = {
        symbol: np.array(
            [
                (2 * l_tmp + 1) * nmax[(symbol, l_tmp)]
                for l_tmp in range(lmax[symbol] + 1)
            ]
        )
        for symbol in uniq_symbols
    }
    for symbol in uniq_symbols:
        assert np.sum(num_coeffs_by_l[symbol]) == num_coeffs_by_uniq_symbol[symbol]

    # Split the overlaps into a list of matrices along axis 0, one for each atom
    split_by_i1 = np.split(overlaps, np.cumsum(num_coeffs_by_symbol), axis=0)[:-1]
    assert len(split_by_i1) == tot_atoms

    tmp_i1 = []
    for i1, i1_matrix in enumerate(split_by_i1):
        # Check the shape
        assert i1_matrix.shape == (
            num_coeffs_by_uniq_symbol[symbols[i1]],
            tot_coeffs,
        )
        # Split the overlaps into a list of matrices along axis 1, one for each atom
        split_by_i2 = np.split(i1_matrix, np.cumsum(num_coeffs_by_symbol), axis=1)[:-1]
        assert len(split_by_i2) == tot_atoms

        tmp_i2 = []
        for i2, i2_matrix in enumerate(split_by_i2):
            # Check that the matrix for this atom is correct shape
            assert i2_matrix.shape == (
                num_coeffs_by_uniq_symbol[symbols[i1]],
                num_coeffs_by_uniq_symbol[symbols[i2]],
            )
            # Split by angular channel l1, along axis 0
            split_by_l1 = np.split(
                i2_matrix, np.cumsum(num_coeffs_by_l[symbols[i1]]), axis=0
            )[:-1]
            assert len(split_by_l1) == lmax[symbols[i1]] + 1

            tmp_l1 = []
            for l1, l1_matrix in enumerate(split_by_l1):
                # Check the shape
                assert l1_matrix.shape == (
                    num_coeffs_by_l[symbols[i1]][l1],
                    num_coeffs_by_uniq_symbol[symbols[i2]],
                )
                # Split now by angular channel l2, along axis 1
                split_by_l2 = np.split(
                    l1_matrix, np.cumsum(num_coeffs_by_l[symbols[i2]]), axis=1
                )[:-1]
                assert len(split_by_l2) == lmax[symbols[i2]] + 1

                # Now reshape the matrices such that the m-components are expanded
                tmp_l2 = []
                for l2, l2_matrix in enumerate(split_by_l2):
                    assert l2_matrix.shape == (
                        num_coeffs_by_l[symbols[i1]][l1],
                        num_coeffs_by_l[symbols[i2]][l2],
                    )

                    l2_matrix = np.reshape(
                        l2_matrix,
                        (
                            1,  # as this is a single atomic sample
                            2 * l1 + 1,
                            nmax[(symbols[i1], l1)],
                            num_coeffs_by_l[symbols[i2]][l2],
                        ),
                        order="F",
                    )

                    l2_matrix = np.reshape(
                        l2_matrix,
                        (
                            1,
                            2 * l1 + 1,
                            nmax[(symbols[i1], l1)],
                            2 * l2 + 1,
                            nmax[(symbols[i2], l2)],
                        ),
                        order="F",
                    )
                    # Reorder the axes such that m-components are in the
                    # intermediate dimensions
                    l2_matrix = np.swapaxes(l2_matrix, 2, 3)

                    # Reshape matrix such that both n1 and n2 are along the final
                    # axes
                    l2_matrix = np.reshape(
                        l2_matrix,
                        (
                            1,
                            2 * l1 + 1,
                            2 * l2 + 1,
                            nmax[(symbols[i1], l1)] * nmax[(symbols[i2], l2)],
                        ),
                        order="C",
                    )

                    assert l2_matrix.shape == (
                        1,
                        2 * l1 + 1,
                        2 * l2 + 1,
                        nmax[(symbols[i1], l1)] * nmax[(symbols[i2], l2)],
                    )
                    tmp_l2.append(l2_matrix)
                tmp_l1.append(tmp_l2)
            tmp_i2.append(tmp_l1)
        tmp_i1.append(tmp_i2)
    split_by_i1 = tmp_i1

    # Initialize dict of the form {(l, symbol): array}
    results_dict = {}
    for symbol_1, symbol_2 in itertools.product(np.unique(symbols), repeat=2):
        for l1 in range(lmax[symbol_1] + 1):
            for l2 in range(lmax[symbol_2] + 1):
                results_dict[(l1, l2, symbol_1, symbol_2)] = {}

    # Store the arrays by l1, l2, species
    for i1, (symbol_1, i1_matrix) in enumerate(zip(symbols, split_by_i1)):
        for i2, (symbol_2, i2_matrix) in enumerate(zip(symbols, i1_matrix)):
            for l1, l1_matrix in zip(range(lmax[symbol_1] + 1), i2_matrix):
                for l2, l2_matrix in zip(range(lmax[symbol_2] + 1), l1_matrix):
                    results_dict[(l1, l2, symbol_1, symbol_2)][i1, i2] = l2_matrix

    # Contruct the TensorMap keys
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
    # Contruct the TensorMap blocks
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
        values = np.ascontiguousarray(
            np.concatenate([raw_block[i1, i2] for i1, i2 in atom_idxs], axis=0)
        )
        block = TensorBlock(
            samples=Labels(
                names=["structure", "center_1", "center_2"],
                values=np.array([[structure_idx, i1, i2] for (i1, i2) in atom_idxs]),
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

    for _ in range(tests):
        assert test_overlap_matrix_conversion(
            frame, structure_idx, lmax, nmax, overlaps, tensor
        )

    return tensor


def test_coeff_vector_conversion(
    frame,
    structure_idx: int,
    lmax: dict,
    nmax: dict,
    coeffs_flat: np.ndarray,
    coeffs_tm: TensorMap,
    print_level: int = 0,
) -> bool:
    """
    Tests that the TensorMap has been constructed correctly from the raw overlap
    matrix.
    """
    # Define some useful variables
    n_atoms = len(frame.get_positions())
    species_symbols = np.array(frame.get_chemical_symbols())

    # Pick a random atom index and find its chemical symbol
    rng = np.random.default_rng()
    i = rng.integers(n_atoms)
    symbol = species_symbols[i]

    # Pick a random l, n, and m
    l = rng.integers(lmax[symbol] + 1)
    n = rng.integers(nmax[(symbol, l)])
    m = rng.integers(-l, l + 1)
    if print_level > 0:
        print("Atom:", i, symbol, "l:", l, "n:", n, "m:", m)

    # Get the flat index + value of this basis function coefficient in the flat array
    flat_index = get_flat_index(species_symbols, lmax, nmax, i, l, n, m)
    raw_elem = coeffs_flat[flat_index]
    if print_level > 0:
        print("Raw array: idx", flat_index, "coeff", raw_elem)

    # Pick out this element from the TensorMap
    tm_block = coeffs_tm.block(
        spherical_harmonics_l=l,
        species_center=SYM_TO_NUM[symbol],
    )
    s_idx = tm_block.samples.position((structure_idx, i))
    c_idx = tm_block.components[0].position((m,))
    p_idx = tm_block.properties.position((n,))

    tm_elem = tm_block.values[s_idx][c_idx][p_idx]
    if print_level > 0:
        print(f"TensorMap: idx", (s_idx, c_idx, p_idx), "coeff", tm_elem)

    return np.isclose(raw_elem, tm_elem)


def test_overlap_matrix_conversion(
    frame,
    structure_idx: int,
    lmax: dict,
    nmax: dict,
    overlaps_matrix: np.ndarray,
    overlaps_tm: TensorMap,
    off_diags_dropped: bool = False,
    print_level: int = 0,
) -> bool:
    """
    Tests that the TensorMap has been constructed correctly from the raw overlap
    matrix.
    """
    # Define some useful variables
    n_atoms = len(frame.get_positions())
    species_symbols = np.array(frame.get_chemical_symbols())

    # Pick 2 random atom indices and find their chemical symbols
    # and define their species symbols
    rng = np.random.default_rng()
    if off_diags_dropped:
        # a1 <= a2 and i1 <= i2; choose `a` values first, then `i` values
        atomic_nums = [SYM_TO_NUM[sym] for sym in species_symbols]

        a1 = rng.choice(atomic_nums, replace=True)
        a2 = rng.choice(atomic_nums, replace=True)
        if a1 > a2:
            a1, a2 = a2, a1

        i2_idxs_gt_i1 = []
        while len(i2_idxs_gt_i1) == 0:  # keep selecting until i1 <= i2 satisfied
            i1 = rng.choice(np.where(atomic_nums == a1)[0], replace=True)
            i2_idxs = np.where(atomic_nums == a2)[0]
            i2_idxs_gt_i1 = [i for i in i2_idxs if i1 <= i]

        i2 = rng.choice(np.array(i2_idxs_gt_i1), replace=True)
        assert a1 <= a2 and i1 <= i2

        symbol1, symbol2 = NUM_TO_SYM[a1], NUM_TO_SYM[a2]

    else:
        # i and a values can be drawn independently
        i1, i2 = rng.integers(n_atoms), rng.integers(n_atoms)
        symbol1, symbol2 = species_symbols[i1], species_symbols[i2]

    # Pick pairs of random l
    l1 = rng.integers(lmax[symbol1] + 1)
    if off_diags_dropped:  # ensure l1 <= l2
        l2 = rng.integers(l1, lmax[symbol2] + 1)
    else:
        l2 = rng.integers(lmax[symbol2] + 1)

    # Pick random pairs of n and m based on the l values
    n1, n2 = rng.integers(nmax[(symbol1, l1)]), rng.integers(nmax[(symbol2, l2)])
    m1, m2 = rng.integers(-l1, l1 + 1), rng.integers(-l2, l2 + 1)

    if print_level > 0:
        print("Atom 1:", i1, symbol1, "l1:", l1, "n1:", n1, "m1:", m1)
        print("Atom 2:", i2, symbol2, "l2:", l2, "n2:", n2, "m2:", m2)

    # Get the flat row and column indices for this matrix element
    row_idx = get_flat_index(species_symbols, lmax, nmax, i1, l1, n1, m1)
    col_idx = get_flat_index(species_symbols, lmax, nmax, i2, l2, n2, m2)
    raw_elem = overlaps_matrix[row_idx][col_idx]
    assert raw_elem == overlaps_matrix[col_idx][row_idx]
    if print_level > 0:
        print("Raw matrix: idx", (row_idx, col_idx), "coeff", raw_elem)

    # Pick out this matrix element from the TensorMap
    # Start by extracting the block.
    tm_block = overlaps_tm.block(
        spherical_harmonics_l1=l1,
        spherical_harmonics_l2=l2,
        species_center_1=SYM_TO_NUM[symbol1],
        species_center_2=SYM_TO_NUM[symbol2],
    )

    # Define the samples, components, and properties indices
    s_idx = tm_block.samples.position((structure_idx, i1, i2))
    c_idx_1 = tm_block.components[0].position((m1,))
    c_idx_2 = tm_block.components[1].position((m2,))
    p_idx = tm_block.properties.position((n1, n2))

    tm_elem = tm_block.values[s_idx][c_idx_1][c_idx_2][p_idx]
    if print_level > 0:
        print(f"TensorMap: idx", (s_idx, c_idx_1, c_idx_2, p_idx), "coeff", tm_elem)

    return np.isclose(raw_elem, tm_elem)


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
    where off-diagonal blocks are dropped, such that the new TensorMap has keys
    with l1 <= l2 and a1 <= a2.
    """
    keys = tensor.keys

    # Define a filter for the keys *TO DROP*
    keys_to_drop_filter = []
    for key in keys:
        l1, l2, a1, a2 = key

        # Keep keys where l1 <= l2 and a1 <= a2
        if a1 < a2:  # keep
            keys_to_drop_filter.append(False)
        elif a1 == a2 and l1 <= l2:  # keep
            keys_to_drop_filter.append(False)
        else:  # drop
            keys_to_drop_filter.append(True)

    # Drop these blocks
    new_tensor = equistore.drop_blocks(tensor, keys=keys[keys_to_drop_filter])

    # Check the number of output blocks is correct
    K_old = len(tensor.keys)
    K_new = len(new_tensor.keys)
    assert K_new == np.sqrt(K_old) / 2 * (np.sqrt(K_old) + 1)

    return new_tensor


def drop_redundant_samples_diagonal_blocks(tensor: TensorMap) -> TensorMap:
    """
    Given an input TensorMap with keys (l1, l2, a1, a2), returns a new TensorMap
    where diagonal blocks (i.e. l1 = l2, a1 = a2) have redundant off-diagonal
    atom pairs from the samples dropped. For instance, the samples must have the
    integer form (A, i1, i2), where i1 and i2 correspond to the atom centers 1
    and 2 repsectively. Only samples where i1 <= i2 are kept.
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

        # Create a samples filter for samples *TO KEEP*
        samples_filter = []
        for sample in block.samples:
            A, i1, i2 = sample
            if i1 <= i2:  # keep
                samples_filter.append(True)
            else:  # discard
                samples_filter.append(False)
        new_samples = block.samples[samples_filter]

        # Check the number of output blocks is correct
        S_old = len(block.samples)
        S_new = len(new_samples)
        assert S_new == np.sqrt(S_old) / 2 * (np.sqrt(S_old) + 1)

        # Create and store the new block
        blocks.append(
            TensorBlock(
                values=block.values[samples_filter],
                samples=new_samples,
                components=block.components,
                properties=block.properties,
            )
        )

    return TensorMap(keys=keys, blocks=blocks)
