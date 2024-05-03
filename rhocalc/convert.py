"""
Module for converting coefficient (or projection) vectors and overlap matrices
between numpy ndarrays and metatensor TensorMap formats.
"""

import itertools
from typing import Optional

import ase
import numpy as np
import torch

import metatensor
from metatensor import Labels, TensorBlock, TensorMap
import metatensor.torch

from rholearn import utils
from rhocalc import convention


def get_flat_index(
    symbol_list: list, lmax: dict, nmax: dict, i: int, l: int, n: int, m: int
) -> int:
    """
    Get the flat index of the coefficient pointed to by the basis function
    indices ``i``, ``l``, ``n``, ``m``.

    Given the basis set definition specified by ``lmax`` and ``nmax``, the
    assumed ordering of basis function coefficients follows the following
    hierarchy, which can be read as nested loops over the various indices. Be
    mindful that some indices range are from 0 to x (exclusive) and others from
    0 to x + 1 (exclusive). The ranges reported below are ordered.

    1. Loop over atoms (index ``i``, of chemical species ``a``) in the
       structure. ``i`` takes values 0 to N (** exclusive **), where N is the
       number of atoms in the structure.
    2. Loop over spherical harmonics channel (index ``l``) for each atom. ``l``
       takes values from 0 to ``lmax[a] + 1`` (** exclusive **), where ``a`` is
       the chemical species of atom ``i``, given by the chemical symbol at the
       ``i``th position of ``symbol_list``.
    3. Loop over radial channel (index ``n``) for each atom ``i`` and spherical
       harmonics channel ``l`` combination. ``n`` takes values from 0 to
       ``nmax[(a, l)]`` (** exclusive **).
    4. Loop over spherical harmonics component (index ``m``) for each atom.
       ``m`` takes values from ``-l`` to ``l`` (** inclusive **).

    Note that basis function coefficient vectors, projection vectors, and
    overlap matrices outputted by quantum chemistry packages such as PySCF and
    AIMS may follow different conventions. ``rholearn`` provides parsing
    functions to standardize these outputs to the convention described above.
    Once standardized, the functions in this module can be used to convert to
    metatensor format.

    :param lmax : dict containing the maximum spherical harmonics (l) value for
        each atom type.
    :param nmax: dict containing the maximum radial channel (n) value for each
        combination of atom type and l.
    :param i: int, the atom index.
    :param l: int, the spherical harmonics index.
    :param n: int, the radial channel index.
    :param m: int, the spherical harmonics component index.

    :return int: the flat index of the coefficient pointed to by the input
        indices.
    """
    # Check atom index is valid
    if i not in np.arange(0, len(symbol_list)):
        raise ValueError(
            f"invalid atom index, i={i} is not in the range [0, {len(symbol_list)})."
            f"Passed symbol list: {symbol_list}"
        )
    # Check l value is valid
    if l not in np.arange(0, lmax[symbol_list[i]] + 1):
        raise ValueError(
            f"invalid spherical harmonics index, l={l} is not in the range "
            f"of valid values for species {symbol_list[i]}: [0, {lmax[symbol_list[i]]}] (inclusive)."
        )
    # Check n value is valid
    if n not in np.arange(0, nmax[(symbol_list[i], l)]):
        raise ValueError(
            f"invalid radial index, n={n} is not in the range of valid values for species "
            f"{symbol_list[i]}, l={l}: [0, {nmax[(symbol_list[i], l)]}) (exclusive)."
        )
    # Check m value is valid
    if m not in np.arange(-l, l + 1):
        raise ValueError(
            f"invalid azimuthal index, m={m} is not in the l range [-l, l] = [{-l}, +{l}] (inclusive)."
        )
    # Define the atom offset
    atom_offset = 0
    for atom_index in range(i):
        symbol = symbol_list[atom_index]
        for l_tmp in np.arange(lmax[symbol] + 1):
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


# ===== convert numpy to metatensor format =====


def coeff_vector_ndarray_to_tensormap(
    frame: ase.Atoms,
    coeffs: np.ndarray,
    lmax: dict,
    nmax: dict,
    structure_idx: Optional[int] = None,
    tests: Optional[int] = 0,
) -> TensorMap:
    """
    Convert a vector of basis function coefficients (or projections) to
    metatensor TensorMap format.

    :param frame: ase.Atoms object containing the atomic structure for which the
        coefficients (or projections) were calculated.
    :param coeffs: np.ndarray of shape (N,), where N is the number of basis
        functions the electron density is expanded onto.
    :param lmax: dict containing the maximum spherical harmonics (l) value for
        each atomic species.
    :param nmax: dict containing the maximum radial channel (n) value for each
        combination of atomic species and l value.
    :param structure_idx: int, the index of the structure in the overall
        dataset. If None (default), the samples metadata of each block in the
        output TensorMap will not contain an index for the structure, i.e. the
        sample names will just be ["atom"]. If an integer, the sample names
        will be ["system", "atom"] and the index for "system" will be
        ``structure_idx``.
    :param tests: int, the number of coefficients to randomly compare between
        the raw input array and processed TensorMap to check for correct
        conversion.

    :return TensorMap: the TensorMap containing the coefficients data and
        metadata.
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
        names=convention.COEFF_VECTOR.keys.names,
        values=np.array([[l, SYM_TO_NUM[symbol]] for l, symbol in results_dict.keys()]),
    )

    # Define the sample names, with or without the structure index
    if structure_idx is None:  # don't include structure idx in the metadata
        sample_names = convention.COEFF_VECTOR.sample_names[1:]
    else:  # include
        sample_names = convention.COEFF_VECTOR.sample_names

    # Build the TensorMap blocks
    blocks = []
    for l, center_type in keys:
        symbol = NUM_TO_SYM[center_type]
        raw_block = results_dict[(l, symbol)]
        atom_idxs = np.sort(list(raw_block.keys()))
        # Define the sample values, with or without the structure index
        if structure_idx is None:  # don't include structure idx in the metadata
            sample_values = np.array([[i] for i in atom_idxs])
        else:  # include
            sample_values = np.array([[structure_idx, i] for i in atom_idxs])
        block = TensorBlock(
            samples=Labels(names=sample_names, values=sample_values),
            components=[
                Labels(
                    names=convention.COEFF_VECTOR.block(0).components[0].names,
                    values=np.arange(-l, +l + 1).reshape(-1, 1),
                ),
            ],
            properties=Labels(
                names=convention.COEFF_VECTOR.property_names,
                values=np.arange(nmax[(symbol, l)]).reshape(-1, 1),
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

    # Check values of the coefficients, repeating the test `tests` number of times.
    for _ in range(tests):
        if not test_coeff_vector_conversion(
            frame,
            lmax,
            nmax,
            coeffs,
            tensor,
            structure_idx=structure_idx,
        ):
            raise ValueError("Conversion test failed.")

    return tensor


def overlap_matrix_ndarray_to_tensormap(
    frame: ase.Atoms,
    overlap: np.ndarray,
    lmax: dict,
    nmax: dict,
    structure_idx: Optional[int] = None,
    tests: int = 0,
) -> TensorMap:
    """
    Converts a 2D numpy array corresponding to the overlap matrix into metatensor
    TensorMap format.

    :param frame: the ASE Atoms object corresponding to the structure for which
        the overlap matrix was computed.
    :param overlap: the overlap matrix, as a 2D numpy array of shape (N, N),
        where N is the number of basis functions the electron density is
        expanded on.
    :param lmax: dict containing the maximum spherical harmonics (l) value for
        each atomic species.
    :param nmax: dict containing the maximum radial channel (n) value for each
        combination of atomic species and l value.
    :param structure_idx: int, the index of the structure in the overall
        dataset. If None (default), the samples metadata of each block in the
        output TensorMap will not contain an index for the structure, i.e. the
        sample names will just be ["atom_1", "atom_2"]. If an integer, the
        sample names will be ["system", "atom_1", "atom_2"] and the index
        for "system" will be ``structure_idx``.
    :param tests: int, the number of coefficients to randomly compare between
        the raw input array ``overlap`` and processed TensorMap to check for
        correct conversion.

    :return TensorMap: the TensorMap containing the overlap matrix data and
        metadata.
    """
    # Define some useful variables
    symbols = frame.get_chemical_symbols()
    uniq_symbols = np.unique(symbols)
    tot_atoms = len(symbols)
    tot_coeffs = overlap.shape[0]
    tot_overlaps = np.prod(overlap.shape)

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
    assert overlap.shape == (tot_coeffs, tot_coeffs)

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

    # Split the overlap into a list of matrices along axis 0, one for each atom
    split_by_i1 = np.split(overlap, np.cumsum(num_coeffs_by_symbol), axis=0)[:-1]
    assert len(split_by_i1) == tot_atoms

    tmp_i1 = []
    for i1, i1_matrix in enumerate(split_by_i1):
        # Check the shape
        assert i1_matrix.shape == (
            num_coeffs_by_uniq_symbol[symbols[i1]],
            tot_coeffs,
        )
        # Split the overlap into a list of matrices along axis 1, one for each atom
        split_by_i2 = np.split(i1_matrix, np.cumsum(num_coeffs_by_symbol), axis=1)[:-1]
        assert len(split_by_i2) == tot_atoms

        tmp_i2 = []
        for i2, i2_matrix in enumerate(split_by_i2):
            # Check that the matrix for this atom is correct shape
            assert i2_matrix.shape == (
                num_coeffs_by_uniq_symbol[symbols[i1]],
                num_coeffs_by_uniq_symbol[symbols[i2]],
            )
            # Split by angular channel l_1, along axis 0
            split_by_l1 = np.split(
                i2_matrix, np.cumsum(num_coeffs_by_l[symbols[i1]]), axis=0
            )[:-1]
            assert len(split_by_l1) == lmax[symbols[i1]] + 1

            tmp_l1 = []
            for l_1, l1_matrix in enumerate(split_by_l1):
                # Check the shape
                assert l1_matrix.shape == (
                    num_coeffs_by_l[symbols[i1]][l_1],
                    num_coeffs_by_uniq_symbol[symbols[i2]],
                )
                # Split now by angular channel l_2, along axis 1
                split_by_l2 = np.split(
                    l1_matrix, np.cumsum(num_coeffs_by_l[symbols[i2]]), axis=1
                )[:-1]
                assert len(split_by_l2) == lmax[symbols[i2]] + 1

                # Now reshape the matrices such that the m-components are expanded
                tmp_l2 = []
                for l_2, l2_matrix in enumerate(split_by_l2):
                    assert l2_matrix.shape == (
                        num_coeffs_by_l[symbols[i1]][l_1],
                        num_coeffs_by_l[symbols[i2]][l_2],
                    )

                    l2_matrix = np.reshape(
                        l2_matrix,
                        (
                            1,  # as this is a single atomic sample
                            2 * l_1 + 1,
                            nmax[(symbols[i1], l_1)],
                            num_coeffs_by_l[symbols[i2]][l_2],
                        ),
                        order="F",
                    )

                    l2_matrix = np.reshape(
                        l2_matrix,
                        (
                            1,
                            2 * l_1 + 1,
                            nmax[(symbols[i1], l_1)],
                            2 * l_2 + 1,
                            nmax[(symbols[i2], l_2)],
                        ),
                        order="F",
                    )
                    # Reorder the axes such that m-components are in the
                    # intermediate dimensions
                    l2_matrix = np.swapaxes(l2_matrix, 2, 3)

                    # Reshape matrix such that both n_1 and n_2 are along the final
                    # axes
                    l2_matrix = np.reshape(
                        l2_matrix,
                        (
                            1,
                            2 * l_1 + 1,
                            2 * l_2 + 1,
                            nmax[(symbols[i1], l_1)] * nmax[(symbols[i2], l_2)],
                        ),
                        order="C",
                    )

                    assert l2_matrix.shape == (
                        1,
                        2 * l_1 + 1,
                        2 * l_2 + 1,
                        nmax[(symbols[i1], l_1)] * nmax[(symbols[i2], l_2)],
                    )
                    tmp_l2.append(l2_matrix)
                tmp_l1.append(tmp_l2)
            tmp_i2.append(tmp_l1)
        tmp_i1.append(tmp_i2)
    split_by_i1 = tmp_i1

    # Initialize dict of the form {(l, symbol): array}
    results_dict = {}
    for symbol_1, symbol_2 in itertools.product(np.unique(symbols), repeat=2):
        for l_1 in range(lmax[symbol_1] + 1):
            for l_2 in range(lmax[symbol_2] + 1):
                results_dict[(l_1, l_2, symbol_1, symbol_2)] = {}

    # Store the arrays by l_1, l_2, species
    for i1, (symbol_1, i1_matrix) in enumerate(zip(symbols, split_by_i1)):
        for i2, (symbol_2, i2_matrix) in enumerate(zip(symbols, i1_matrix)):
            for l_1, l1_matrix in zip(range(lmax[symbol_1] + 1), i2_matrix):
                for l_2, l2_matrix in zip(range(lmax[symbol_2] + 1), l1_matrix):
                    results_dict[(l_1, l_2, symbol_1, symbol_2)][i1, i2] = l2_matrix

    # Contruct the TensorMap keys
    keys = Labels(
        names=convention.OVERLAP_MATRIX.keys.names,
        values=np.array(
            [
                [l_1, l_2, SYM_TO_NUM[symbol_1], SYM_TO_NUM[symbol_2]]
                for l_1, l_2, symbol_1, symbol_2 in results_dict.keys()
            ]
        ),
    )

    # Define the sample names, with or without the structure index
    if structure_idx is None:  # don't include structure idx in the metadata
        sample_names = convention.OVERLAP_MATRIX.sample_names[1:]
    else:  # include
        sample_names = convention.OVERLAP_MATRIX.sample_names

    # Contruct the TensorMap blocks
    blocks = []
    for l_1, l_2, center_1_type, center_2_type in keys:
        # Get the chemical symbols for the corresponding atomic numbers
        symbol_1 = NUM_TO_SYM[center_1_type]
        symbol_2 = NUM_TO_SYM[center_2_type]

        # Retrieve the raw block of data for the l and symbols
        raw_block = results_dict[(l_1, l_2, symbol_1, symbol_2)]

        # Get the atomic indices
        atom_idxs = np.array(list(raw_block.keys()))

        # Define the sample values, with or without the structure index
        if structure_idx is None:  # don't include structure idx in the metadata
            sample_values = np.array([[i1, i2] for (i1, i2) in atom_idxs])
        else:  # include
            sample_values = np.array(
                [[structure_idx, i1, i2] for (i1, i2) in atom_idxs]
            )

        # Build the TensorBlock
        values = np.ascontiguousarray(
            np.concatenate([raw_block[i1, i2] for i1, i2 in atom_idxs], axis=0)
        )
        block = TensorBlock(
            samples=Labels(
                names=sample_names,
                values=sample_values,
            ),
            components=[
                Labels(
                    names=convention.OVERLAP_MATRIX.block(0).components[0].names,
                    values=np.arange(-l_1, +l_1 + 1).reshape(-1, 1),
                ),
                Labels(
                    names=convention.OVERLAP_MATRIX.block(0).components[1].names,
                    values=np.arange(-l_2, +l_2 + 1).reshape(-1, 1),
                ),
            ],
            properties=Labels(
                names=convention.OVERLAP_MATRIX.property_names,
                values=np.array(
                    [
                        [n_1, n_2]
                        for n_1, n_2 in itertools.product(
                            np.arange(nmax[(symbol_1, l_1)]),
                            np.arange(nmax[(symbol_2, l_2)]),
                        )
                    ]
                ),
            ),
            values=values,
        )
        blocks.append(block)

    # Build TensorMap and check num elements same as input
    tensor = TensorMap(keys=keys, blocks=blocks)
    assert utils.num_elements_tensormap(tensor) == np.prod(overlap.shape)

    for _ in range(tests):
        assert test_overlap_matrix_conversion(
            frame,
            lmax,
            nmax,
            overlap,
            tensor,
            structure_idx=structure_idx,
        )

    return tensor


# ===== Functions to sparsify the symmetric overlap matrix =====


def overlap_is_symmetric(tensor: TensorMap) -> bool:
    """
    Returns true if the overlap matrices stored in TensorMap form are symmetric,
    false otherwise.

    This class assumes that the overlap-type matrices are stored in metatensor
    TensorMap format and has the data structure as given in the example
    TensorMap found in `rhoparse.convention.OVERLAP_MATRIX`.

    :param tensor: the overlap matrix in TensorMap format. Will be checked for
        symmetry.
    """
    # Get the key names and check they have the assumed form
    keys = tensor.keys
    assert np.all(keys.names == convention.OVERLAP_MATRIX.keys.names)

    # Iterate over the blocks
    checked = set()  # for storing the keys that have been checked
    for key in keys:
        l_1, l_2, a1, a2 = key
        if (l_1, l_2, a1, a2) in checked or (l_2, l_1, a2, a1) in checked:
            continue

        # Get 2 blocks for comparison, permuting l and a. If this is a diagonal
        # block, block 1 and block 2 will be the same object
        block1 = tensor.block(
            o3_lambda_1=l_1,
            o3_lambda_2=l_2,
            center_1_type=a1,
            center_2_type=a2,
        )
        block2 = tensor.block(
            o3_lambda_1=l_2,
            o3_lambda_2=l_1,
            center_1_type=a2,
            center_2_type=a1,
        )
        if not overlap_is_symmetric_block(block1, block2):
            return False
        # Only track checking of off-diagonal blocks
        if not (l_1 == l_2 and a1 == a2):
            checked.update({(l_1, l_2, a1, a2)})

    return True


def overlap_is_symmetric_block(block1: TensorBlock, block2: TensorBlock) -> bool:
    """
    Returns true if the passed blocks are symmetric with respect to eachother,
    false otherwise. The relevant data of one of the blocks is permuted and
    checked for exact equivalence with the other block.

    Assumes both blocks have the data structure shown in the only block of the
    example TensorMap found in `rhoparse.convention.OVERLAP_MATRIX`.

    If symmetric, the data tensor of ``block2`` should be exactly equivalent to
    that of ``block2`` after permuting:

        - the atom-center pairs in the samples, i.e. 'atom_1' with 'atom_2'.
        - the spherical harmonics components *axes* (as these exist on separate
          component axes, not in the same one as with samples/properties), i.e.
          'o3_mu_1' with 'o3_mu_2'.
        - the radial channels in the properties, i.e. 'n_1' with 'n_2'.
    """
    # Check the samples names and determine whether or not the structure index
    # is included
    if np.all(
        block1.samples.names == convention.OVERLAP_MATRIX.sample_names
    ) and np.all(block2.samples.names == convention.OVERLAP_MATRIX.sample_names):
        structure_idx_present = True
    elif np.all(
        block1.samples.names == convention.OVERLAP_MATRIX.sample_names[1:]
    ) and np.all(block2.samples.names == convention.OVERLAP_MATRIX.sample_names[1:]):
        structure_idx_present = False
    else:
        raise ValueError(
            f"the sample names of both blocks must be either "
            f"{convention.OVERLAP_MATRIX.sample_names}"
            f" or {convention.OVERLAP_MATRIX.sample_names[1:]},"
            f" but got: {block1.samples.names} and {block2.samples.names}"
        )
    # Check the component names
    c_names = [c.names for c in convention.OVERLAP_MATRIX.block(0).components]
    for block in [block1, block2]:
        if not (
            np.all(block.components[0].names == c_names[0])
            and np.all(block.components[1].names == c_names[1])
        ):
            raise ValueError(
                f"the component names of both blocks must be {c_names},"
                f" but got: {block.components[0].names} and {block.components[1].names}"
            )
    # Check the property names
    if not (
        np.all(block1.properties.names == convention.OVERLAP_MATRIX.property_names)
        and np.all(block2.properties.names == convention.OVERLAP_MATRIX.property_names)
    ):
        raise ValueError(
            f"the property names of both blocks must be "
            f"{convention.OVERLAP_MATRIX.property_names}, but got:"
            f" {block1.properties.names} and {block2.properties.names}"
        )

    # Create a samples filter for how the samples map to eachother between
    # blocks
    if structure_idx_present:
        samples_filter = [
            block2.samples.position((A, i2, i1)) for [A, i1, i2] in block1.samples
        ]
    else:
        samples_filter = [
            block2.samples.position((i2, i1)) for [i1, i2] in block1.samples
        ]
    # Create a properties filter for how the properties map to eachother between
    # blocks
    properties_filter = [
        block2.properties.position((n_2, n_1)) for [n_1, n_2] in block1.properties
    ]
    # Broadcast the values array using the filters, and swap the components axes
    reordered_block2 = np.swapaxes(
        block2.values[samples_filter][..., properties_filter], 1, 2
    )

    return np.allclose(block1.values, reordered_block2)


def overlap_drop_redundant_off_diagonal_blocks(tensor: TensorMap) -> TensorMap:
    """
    Takes an input TensorMap ``tensor`` that corresponds to the overlap matrix
    for a given structure. Assumes blocks have keys with names
    ("o3_lambda_1", "o3_lambda_2", "center_1_type",
    "center_2_type",) corresponding to indices of the form (l_1, l_2, a1, a2).

    The returned TensorMap only has off-diagonal blocks with keys where:
        - l_1 < l_2, or
        - a1 <= a2 if l_1 == l_2

    ensuring that if an off-diagonal block (l_1, l_2, a1, a2) exists (i.e. where
    l_1 != l_2 or a1 != a2), then the exactly symmetric (and redundant) block (l_2,
    l_1, a2, a1) has been dropped.

    :param tensor: the input TensorMap corresponding to the overlap-type matrix
        in metatensor format. Must have keys with names
        ("o3_lambda_1", "o3_lambda_2", "center_1_type",
        "center_2_type",)

    :return: the TensorMap with redundant off-diagonal blocks dropped.
    """
    # Get the key names and check they have the assumed form
    keys = tensor.keys
    assert np.all(keys.names == convention.OVERLAP_MATRIX.keys.names)

    # Define a filter for the keys *TO DROP*
    keys_to_drop_filter = []
    for l_1, l_2, a1, a2 in keys:
        # Keep keys where l_1 < l_2, or a1 <= a2 if l_1 == l_2
        if l_1 < l_2:  # keep
            keys_to_drop_filter.append(False)
        elif l_1 == l_2:
            if a1 <= a2:  # keep
                keys_to_drop_filter.append(False)
            else:  # drop
                keys_to_drop_filter.append(True)
        else:  # l_1 > l_2: drop
            keys_to_drop_filter.append(True)

    # Drop these blocks
    new_tensor = metatensor.drop_blocks(
        tensor,
        keys=Labels(names=keys.names, values=keys.values[keys_to_drop_filter]),
    )

    # Check the number of output blocks is correct
    K_old = len(tensor.keys)
    K_new = len(new_tensor.keys)
    assert K_new == np.sqrt(K_old) / 2 * (np.sqrt(K_old) + 1)

    return new_tensor


# ===== convert metatensor to numpy format =====


def coeff_vector_tensormap_to_ndarray(
    frame: ase.Atoms,
    tensor: TensorMap,
    lmax: dict,
    nmax: dict,
    tests: Optional[int] = 0,
) -> np.ndarray:
    """
    Convert a metatensor TensorMap of basis function coefficients (or
    projections) to numpy ndarray format.

    :param frame: ase.Atoms object containing the atomic structure for which the
        coefficients (or projections) were calculated.
    :param tensor: the TensorMap containing the basis function coefficients data
        and metadata.
    :param lmax: dict containing the maximum spherical harmonics (l) value for
        each atomic species.
    :param nmax: dict containing the maximum radial channel (n) value for each
        combination of atomic species and l value.
    :param tests: int, the number of coefficients to randomly compare between
        the raw input array and processed TensorMap to check for correct
        conversion.

    :return np.ndarray: vector of coefficients converted from TensorMap format,
        of shape (N,), where N is the number of basis functions the electron
    density is expanded onto
    """
    # Convert to core/numpy if a torch/torchscript tensormap
    if isinstance(tensor, torch.ScriptObject):
        tensor = utils.mts_tensormap_torch_to_core(
            metatensor.torch.requires_grad(tensor, False)
        ).to(arrays="numpy")

    # Check the samples names and determine whether or not the structure index
    # is included
    if tensor.sample_names == convention.COEFF_VECTOR.sample_names:
        structure_idx_present = True
        structure_idxs = metatensor.unique_metadata(tensor, "samples", "system")
        assert len(structure_idxs) == 1
        structure_idx = structure_idxs[0].values[0]
    elif tensor.sample_names == convention.COEFF_VECTOR.sample_names[1:]:
        structure_idx_present = False
        structure_idx = None
    else:
        raise ValueError(
            "the sample names of the input tensor must be either "
            f"{convention.COEFF_VECTOR.sample_names} or "
            f"{convention.COEFF_VECTOR.sample_names[1:]}, but got:"
            f" {tensor.sample_names}"
        )

    # Define some useful variables
    symbols = frame.get_chemical_symbols()
    uniq_symbols = np.unique(symbols)
    tot_atoms = len(symbols)
    tot_coeffs = utils.num_elements_tensormap(tensor)

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
    if np.sum(num_coeffs_by_symbol) != tot_coeffs:
        raise ValueError(
            f"the number of coefficients in the input tensor ({tot_coeffs})"
            " does not match the expected number of coefficients according"
            " to the basis set definition and input frame"
            f" ({np.sum(num_coeffs_by_symbol)})"
        )

    # Initialize a dict to store the block values
    results_dict = {}

    # Loop over the blocks and split up the values tensors
    for key, block in tensor.items():
        l, a = key
        symbol = NUM_TO_SYM[a]
        tmp_dict = {}

        # Store the block values in a dict by atom index
        for atom_idx in np.unique(block.samples["atom"]):
            atom_idx_mask = block.samples["atom"] == atom_idx
            # Get the array of values for this atom, of species `symbol` and `l`` value
            # The shape of this array is (1, 2*l+1, nmax[(symbol, l)]
            atom_arr = block.values[atom_idx_mask]
            assert atom_arr.shape == (1, 2 * l + 1, nmax[(symbol, l)])
            # Reshape to a flatten array and store. IMPORTANT: Fortran order
            atom_arr = np.reshape(atom_arr, (-1,), order="F")
            tmp_dict[atom_idx] = atom_arr
        results_dict[(l, symbol)] = tmp_dict

    # Combine the individual arrays into a single flat vector
    # Loop over the atomic species in the order given in `frame`
    coeffs = np.array([])
    for atom_i, symbol in enumerate(symbols):
        for l in range(lmax[symbol] + 1):
            coeffs = np.append(coeffs, results_dict[(l, symbol)][atom_i])

    # Check number of elements
    assert len(coeffs) == tot_coeffs

    # Check values of the coefficients, repeating the test `tests` number of times.
    for _ in range(tests):
        if not test_coeff_vector_conversion(
            frame,
            lmax,
            nmax,
            coeffs,
            tensor,
            structure_idx=structure_idx,
        ):
            raise ValueError("Conversion test failed.")

    # Loop over the blocks and store the values
    return coeffs


def overlap_matrix_tensormap_to_ndarray(
    frame: ase.Atoms,
    overlap: TensorMap,
    lmax: dict,
    nmax: dict,
    structure_idx: Optional[int] = None,
    tests: int = 0,
) -> TensorMap:
    """ """
    raise NotImplementedError


# ===== Functions to test conversions etc. =====


def test_coeff_vector_conversion(
    frame,
    lmax: dict,
    nmax: dict,
    coeffs_flat: np.ndarray,
    coeffs_tm: TensorMap,
    structure_idx: Optional[int] = None,
    print_level: int = 0,
) -> bool:
    """
    Tests that the TensorMap has been constructed correctly from the raw
    coefficients vector.
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
        o3_lambda=l,
        center_type=ase.Atoms(symbol).get_atomic_numbers()[0],
    )
    if structure_idx is None:
        # s_idx = tm_block.samples.position((i,))
        s_idx = tm_block.samples.position(
            Labels(names=["atom"], values=np.array([[i]]))[0]
        )
    else:
        # s_idx = tm_block.samples.position((structure_idx, i))
        s_idx = tm_block.samples.position(
            Labels(names=["system", "atom"], values=np.array([[structure_idx, i]]))[0]
        )

    # c_idx = tm_block.components[0].position((m,))
    c_idx = tm_block.components[0].position(
        Labels(names=["o3_mu"], values=np.array([[m]]))[0]
    )
    # p_idx = tm_block.properties.position((n,))
    p_idx = tm_block.properties.position(Labels(names=["n"], values=np.array([[n]]))[0])

    tm_elem = tm_block.values[s_idx][c_idx][p_idx]
    if print_level > 0:
        print(f"TensorMap: idx", (s_idx, c_idx, p_idx), "coeff", tm_elem)

    return np.isclose(raw_elem, tm_elem)


def test_overlap_matrix_conversion(
    frame,
    lmax: dict,
    nmax: dict,
    overlaps_matrix: np.ndarray,
    overlaps_tm: TensorMap,
    structure_idx: Optional[int] = None,
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
    l_1 = rng.integers(lmax[symbol1] + 1)
    if off_diags_dropped:  # ensure l_1 <= l_2
        l_2 = rng.integers(l_1, lmax[symbol2] + 1)
    else:
        l_2 = rng.integers(lmax[symbol2] + 1)

    # Pick random pairs of n and m based on the l values
    n_1, n_2 = rng.integers(nmax[(symbol1, l_1)]), rng.integers(nmax[(symbol2, l_2)])
    m1, m2 = rng.integers(-l_1, l_1 + 1), rng.integers(-l_2, l_2 + 1)

    if print_level > 0:
        print("Atom 1:", i1, symbol1, "l_1:", l_1, "n_1:", n_1, "m1:", m1)
        print("Atom 2:", i2, symbol2, "l_2:", l_2, "n_2:", n_2, "m2:", m2)

    # Get the flat row and column indices for this matrix element
    row_idx = get_flat_index(species_symbols, lmax, nmax, i1, l_1, n_1, m1)
    col_idx = get_flat_index(species_symbols, lmax, nmax, i2, l_2, n_2, m2)
    raw_elem = overlaps_matrix[row_idx][col_idx]

    # Check that the matrix element is symmetric
    assert np.isclose(raw_elem, overlaps_matrix[col_idx][row_idx])

    if print_level > 0:
        print("Raw matrix: idx", (row_idx, col_idx), "coeff", raw_elem)

    # Pick out this matrix element from the TensorMap
    # Start by extracting the block.
    tm_block = overlaps_tm.block(
        o3_lambda_1=l_1,
        o3_lambda_2=l_2,
        center_1_type=SYM_TO_NUM[symbol1],
        center_2_type=SYM_TO_NUM[symbol2],
    )

    # Define the samples, components, and properties indices
    if structure_idx is None:
        s_idx = tm_block.samples.position((i1, i2))
    else:
        s_idx = tm_block.samples.position((structure_idx, i1, i2))
    c_idx_1 = tm_block.components[0].position((m1,))
    c_idx_2 = tm_block.components[1].position((m2,))
    p_idx = tm_block.properties.position((n_1, n_2))

    tm_elem = tm_block.values[s_idx][c_idx_1][c_idx_2][p_idx]
    if print_level > 0:
        print(f"TensorMap: idx", (s_idx, c_idx_1, c_idx_2, p_idx), "coeff", tm_elem)

    return np.isclose(raw_elem, tm_elem)


SYM_TO_NUM = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
}
NUM_TO_SYM = {num: sym for sym, num in SYM_TO_NUM.items()}
