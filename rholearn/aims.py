"""
A module for generating FHI-AIMS input files and parsing output files, tailored
to calculations of RI electron density coefficients, projections, and overlap
matrices.
"""
import itertools
import os
from typing import List, Tuple, Optional

import ase
import ase.io
import numpy as np

import equistore
from equistore import Labels, TensorBlock, TensorMap

from rholearn import utils


SYM_TO_NUM = {"H": 1, "C": 6, "O": 8, "N": 7}
NUM_TO_SYM = {1: "H", 6: "C", 8: "O", 7: "N"}


# ===== AIMS input file generation =====


def generate_aims_input_geometry_files(frames: List[ase.Atoms], save_dir: str):
    """
    Takes a list of ASE Atoms objects (i.e. ``frames``) for a set of structures
    and generates input geometry files in the AIMS format.

    For a set of N structures in ``frames``, N new directories in the parent
    directory ``save_dir`` are created, with relative paths
    f"{save_dir}/{A}/geometry.in", where A is a numeric structure index running
    from 0 -> (N - 1) (inclusive), and corresponding to the order of structures
    in ``frames``.

    :param frames: a :py:class:`list` of :py:class:`ase.Atoms` objects
        corresponding to the structures in the dataset to generate AIMS input
        files for.
    :param save_dir: a `str` of the absolute path to the directory where the
        AIMS input geometry files should be saved.
    """

    # Create the save directory if it doesn't already exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for A in range(len(frames)):  # Iterate over structures
        # Create a directory named simply by the structure index
        structure_dir = os.path.join(save_dir, f"{A}")
        if not os.path.exists(structure_dir):
            os.mkdir(structure_dir)

        # Write the AIMS input file. By using the ".in" suffix/extension in the
        # filename, ASE will automatically produce an input file that follows
        # AIMS formatting.
        ase.io.write(os.path.join(structure_dir, "geometry.in"), frames[A])


# ===== AIMS output file parsing =====


def extract_aims_basis_set_info(
    frame: ase.Atoms, aims_output_dir: str
) -> Tuple[dict, dict]:
    """
    Takes an AIMS basis info file and converts it into a dictionary of the lmax
    and nmax values for each atom type in the structure.

    :param frame: an :py:class:`ase.Atoms` object corresponding to the structure
        for which the AIMS basis set info should be extracted.
    :param aims_output_dir: a `str` of the absolute path to the directory
        containing AIMS output files. In particular, this directory must contain
        a file called "basis_info.out". This contains the information of the
        constructed RI basis set for the structure passed in ``frame``.

    :return lmax: a `dict` of the maximum angular momentum for each chemical
        species in ``frame``.
    :return nmax: a `dict` of the maximum radial channel index for each chemical
        species and angular channel in ``frame``.
    """
    # Check the directory containing AIMS output files exists
    if not os.path.exists(aims_output_dir):
        raise ValueError(f"`aims_output_dir` {aims_output_dir} does not exist.")

    # Check the basis info file exists
    basis_info_file = os.path.join(aims_output_dir, "basis_info.out")
    if not os.path.exists(basis_info_file):
        raise FileNotFoundError(
            f"{basis_info_file} does not exist. Check it is in the directory"
            f" {aims_output_dir} and it has not been renamed"
        )

    # Read the basis set information
    with open(basis_info_file, "r") as f:
        lines = f.readlines()

    # Get the species symbols for the atoms in the frame
    symbols = frame.get_chemical_symbols()

    # Parse the file to extract the line number intervals for each atom block
    intervals = []
    for line_i, line in enumerate(lines):
        line_split = line.split()
        if len(line_split) == 0:
            continue

        if line_split[0] == "atom":
            block_start = line_i
            continue
        elif line_split[:2] == ["For", "atom"]:
            block_end = line_i + 1
            intervals.append((block_start, block_end))
            continue

    # Group the lines of the file into atom blocks
    atom_blocks = [lines[start:end] for start, end in intervals]

    # Parse the lmax and nmax values for each chemical species
    # This assumes that the basis set parameters is the same for every atom of
    # the same chemical species
    lmax, nmax = {}, {}
    for block in atom_blocks:
        # Get the atom index (zero indexed)
        atom_idx = int(block[0].split()[1]) - 1

        # Skip if we already have an entry for this chemical species
        symbol = symbols[atom_idx]
        if symbol in lmax.keys():
            continue

        # Get the max l value and store
        assert int(block[-1].split()[2]) - 1 == atom_idx
        species_lmax = int(block[-1].split()[6])
        lmax[symbol] = species_lmax

        # Parse the nmax values and store. There are (lmax + 1) angular channels
        for l in range(species_lmax + 1):
            line = block[l + 1]
            assert l == int(line.split()[3])
            species_nmax = int(line.split()[6])
            nmax[(symbol, l)] = species_nmax

    return lmax, nmax


def process_aims_aux_basis_func_data(
    aims_output_dir: str,
    save_dir: Optional[str] = None,
    delete_original_files: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes and returns the coefficients and projections vectors, and overlap
    matirx, from the AIMS output file directory at absolute path
    `aims_output_dir`. If the absolute path `save_dir` is specified, the data is
    saved to the directory under filenames `coefficients.npy`,
    `projections.npy`, and `overlap.npy`.

    Assumes that in the directory `aims_output_dir` there exists the following
    files, named according to AIMS calculation output using version at release
    221103 with keyword "ri_full_output" set to true. Assumes that the files
    have not been modified in any way.

        - "ri_restart_coeffs.out" contains a single column of the RI fitted
          auxiliary basis function (ABF) coefficients. There should be N
          entries, where N is the number of ABFs.
        - "ri_projections.out" contains a single column of the RI fitted ABF
          projections. There should be N entries.
        - "ri_ovlp.out" contains a single column of the elements of the overlap
          matrix between all pairs of ABFs. There should be N^2 entries.

        - "idx_prodbas_details.out" contains 5 columns of information about the
          auxiliary basis functions. The columns correspond to, respectively:
              -  ABF numeric index, running from 1 to N (inclusive). Note that
                 AIMS outputs numeric indices using 1- (not 0-) indexing.
              - atom index, running from 1 to N_atoms (inc). The index of the
                atom that the ABF is centered on.
              - angular momentum l value, running from 0 to l_max (inc). Note
                again that this is an inclusive range, but this time starting at
                0 as this is an l *value*, not a numeric index. As such, there
                are (l_max + 1) possible values of l.
              - radial channel index. This is a cumulative index across all l
                values. For instance, the radial channels for l = 0 are indexed
                from 1 to n_{max}(l=0) inclusive, the radial channels for l = 1
                are indexed from (n_{max}(l=0) + 1) to n_{max}(l=1) inclusive,
                and so on. Note again that this is a numeric index, so is
                1-indexed.
              - angular momentum component m value, running from -l to l (inc)
                for the given l value.
        - "prodbas_condon_shotley_list.out" contains a single column of the
          numeric indices of the ABFs to which the Condon-Shortley (CS)
          convention should be applied. The important thing to note is that in
          AIMS, the CS convention is *NOT* applied to ABFs with m > 0, hence it
          needs to be applied here.

    This function performs the following processing of the data.

    First, the 1-indexing of the numeric indices in "idx_prodbas_details.out"
    (i.e. ABF, atom, and radial channel indices) and
    "prodbas_condon_shotley_list.out" (i.e. the ABF indices) is converted to
    0-indexing. Second, the CS convention is applied to coefficients,
    projections, and overlap matrix elements for the ABFs with m > 0. Third, the
    order of the coefficients and projections are modified to match the numeric
    indices of the ABFs in "idx_prodbas_details.out".

    :param aims_output_dir: str for the absolute path to the directory
        containing the AIMS output files from the RI calculation on a single
        structure using keyword "ri_full_output" set to true.
    :param save_dir: optional ``str`` to the absolute path to the directory in
        which to save the processed coefficients, projections, and overlap
        matrix.
    :param delete_original_files: optional ``bool`` to indicate whether or not
        to delete the original AIMS output files corresponding to the
        coefficients, projections, and overlap matrix, i.e.
        "ri_restart_coeffs.out", "ri_projections.out", and "ri_ovlp.out".

    :return: A tuple of the coefficients, projections, and overlap matrix, all
        as numpy arrays. The coefficients and projections are 1D arrays with
        shape (N,), where N is the number of ABFs. The overlap matrix is a 2D
        array with shape (N, N).
    """
    # Load coefficients, projections, and overlap matrix
    if not os.path.exists(aims_output_dir):
        raise ValueError(f"`aims_output_dir` {aims_output_dir} does not exist.")
    if save_dir is not None:
        if not os.path.exists(save_dir):
            raise ValueError(f"`save_dir` {save_dir} does not exist.")

    coeffs = np.loadtxt(os.path.join(aims_output_dir, "ri_restart_coeffs.out"))
    projs = np.loadtxt(os.path.join(aims_output_dir, "ri_projections.out"))
    overlap = np.loadtxt(os.path.join(aims_output_dir, "ri_ovlp.out"))

    # Check shapes
    assert coeffs.shape == projs.shape
    assert overlap.shape == (coeffs.shape[0] ** 2,)

    # Reshape overlap into a square matrix and check symmetry. As the overlap
    # matrix should be symmetric, reshaping in either C or F order should give
    # the same result. Here we just use the default C order.
    overlap = overlap.reshape(coeffs.shape[0], coeffs.shape[0])
    assert np.allclose(overlap, overlap.T)

    # Load the auxiliary basis function (ABF) information. This is 2D array where
    # columns correspond to, respectively, the auxiliary basis function index, atom
    # index, angular momentum l value, radial channel index, and the angular
    # momentum component m value.
    abf_info = np.loadtxt(
        os.path.join(aims_output_dir, "idx_prodbas_details.out"),
        dtype=int,
    )
    # Convert to zero indexing for the columns that correspond to numeric indices.
    # The l and m values need not be modified here.
    abf_info[:, 0] -= 1  # ABF index
    abf_info[:, 1] -= 1  # atom index
    abf_info[:, 3] -= 1  # radial channel index

    # Load the indices of the auxiliary basis functions to which the Condon-Shortley
    # convention should be applied
    cs_abf_idxs = np.loadtxt(
        os.path.join(aims_output_dir, "prodbas_condon_shotley_list.out"),
        dtype=int,
    )
    cs_abf_idxs -= 1  # Convert to zero indexing

    # Check that all the ABFs in `cs_abf_idxs` have an positive odd value of m,
    # and those not present have an even, or negative, value of m. This is just
    # a convention of AIMS, that the Condon-Shortley convention is *NOT* applied
    # to m > 0. In AIMS version 221103, in file /src/cartesian_ylm.f90, this is
    # explained.
    for abf in abf_info:
        abf_idx = abf[0]
        abf_m_value = abf[4]
        if abf_idx in cs_abf_idxs:  # assert m odd and positive
            assert abf_m_value % 2 == 1 and abf_m_value > 0
        else:  # assert m even, or negative
            assert abf_m_value % 2 == 0 or abf_m_value < 0

    # Apply the Condon-Shortley convention to the coefficients, projections, and
    # overlap
    for cs_abf_idx in cs_abf_idxs:
        coeffs[cs_abf_idx] *= -1
        projs[cs_abf_idx] *= -1
        overlap[cs_abf_idx, :] *= -1
        overlap[:, cs_abf_idx] *= -1

    # Broadcast the coefficients, projections, and overlap such that they are
    # ordered according to the ABF indices: 0, 1, 2, ...
    coeffs = coeffs[abf_info[:, 0]]
    projs = projs[abf_info[:, 0]]
    overlap = overlap[abf_info[:, 0], :]
    overlap = overlap[:, abf_info[:, 0]]

    # Save files if requested
    if save_dir is not None:
        np.save(os.path.join(save_dir, "coeffs.npy"), coeffs)
        np.save(os.path.join(save_dir, "projs.npy"), projs)
        np.save(os.path.join(save_dir, "overlap.npy"), overlap)

    # Delete original files if requested
    if delete_original_files:
        os.remove(os.path.join(aims_output_dir, "ri_restart_coeffs.out"))
        os.remove(os.path.join(aims_output_dir, "ri_projections.out"))
        os.remove(os.path.join(aims_output_dir, "ri_ovlp.out"))

    return coeffs, projs, overlap


def density_fitting_error(aims_output_dir: str) -> float:
    """
    Calculates the error in the RI fitted electron density relative to the SCF
    converged electron density. The files required for this calculation, that
    must be present in `aims_output_dir` are as follows:
    
        - rho_scf.out: SCF converged electron density.
        - rho_rebuilt_ri.out: RI fitted electron density.
        - partition_tab.out: the tabulated partition function.

    :param aims_output_dir: str for the absolute path to the directory
        containing the AIMS output files from the RI calculation on a single
        structure using keyword "ri_full_output" set to true.

    :return float: the error in the RI fitted density relative to the SCF
        converged density.
    """
    # Check various directories and paths exist
    if not os.path.isdir(aims_output_dir):
        raise NotADirectoryError(
            f"The directory {aims_output_dir} does not exist."
        )
    if not os.path.exists(os.path.join(aims_output_dir, "rho_scf.out")):
        raise FileNotFoundError(
            f"The file rho_scf.out does not exist in {aims_output_dir}."
        )
    if not os.path.exists(os.path.join(aims_output_dir, "rho_rebuilt_ri.out")):
        raise FileNotFoundError(
            f"The file rho_rebuilt_ri.out does not exist in {aims_output_dir}."
        )
    if not os.path.exists(os.path.join(aims_output_dir, "partition_tab.out")):
        raise FileNotFoundError(
            f"The file partition_tab.out does not exist in {aims_output_dir}."
        )
        
    # Load the real-space density data. Each row corresponds to x, y, z coordinates
    # followed by the value of the density in the 3rd column. We are only
    # interested
    # 1) SCF converged electron density
    scf_rho = np.loadtxt(os.path.join(aims_output_dir, "rho_scf.out"))
    # 2) RI fitted electron density
    ri_rho = np.loadtxt(os.path.join(aims_output_dir, "rho_rebuilt_ri.out"))
    # 3) Partitioning table
    partition = np.loadtxt(os.path.join(aims_output_dir, "partition_tab.out"))

    # Check the coordinates on each row are exactly equivalent
    assert np.all(scf_rho[:, :3] == ri_rho[:, :3]) and np.all(
        scf_rho[:, :3] == partition[:, :3]
    )

    # Now just slice to keep only the final column of data from each file, as
    # this is the only bit we're interested in
    scf_rho = scf_rho[:, 3]
    ri_rho = ri_rho[:, 3]
    partition = partition[:, 3]

    # Get the absolute residual error between the SCF and fitted densities
    error = np.abs(ri_rho - scf_rho)

    # Calculate and return the relative error
    return np.dot(error, partition) / np.dot(scf_rho, partition)


# ===== Converting to equistore format =====


def get_flat_index(
    symbol_list: list, lmax: dict, nmax: dict, i: int, l: int, n: int, m: int
) -> int:
    """
    Get the flat index of the coefficient pointed to by the basis function
    indices i, l, n, m, in AIMS format.

    :param lmax : dict containing the maximum spherical harmonics (l)
        value for each atom type.
    :param nmax: dict containing the maximum radial channel (n) value
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


def coeff_vector_to_tensormap(
    frame: ase.Atoms,
    coeffs: np.ndarray,
    lmax: dict,
    nmax: dict,
    structure_idx: Optional[int] = None,
    tests: Optional[int] = 0,
) -> TensorMap:
    """
    Convert a vector of basis function coefficients (or projections) to
    equistore TensorMap format.

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
        sample names will just be ["center"]. If an integer, the sample names
        will be ["structure", "center"] and the index for "structure" will be
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
        names=["spherical_harmonics_l", "species_center"],
        values=np.array([[l, SYM_TO_NUM[symbol]] for l, symbol in results_dict.keys()]),
    )

    # Define the sample names, with or without the structure index
    if structure_idx is None:  # don't include structure idx in the metadata
        sample_names = ["center"]
    else:  # include
        sample_names = ["structure", "center"]

    # Build the TensorMap blocks
    blocks = []
    for l, species_center in keys:
        symbol = NUM_TO_SYM[species_center]
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

    # Check values of the coefficients, repeating the test `tests` number of times.
    for _ in range(tests):
        if not test_coeff_vector_conversion(
            frame, structure_idx, lmax, nmax, coeffs, tensor
        ):
            raise ValueError("Conversion test failed.")

    return tensor


def overlap_matrix_to_tensormap(
    frame: ase.Atoms,
    overlap: np.ndarray,
    lmax: dict,
    nmax: dict,
    structure_idx: Optional[int] = None,
    tests: int = 0,
) -> TensorMap:
    """
    Converts a 2D numpy array corresponding to the overlap matrix into equistore
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
        sample names will just be ["center_1", "center_2"]. If an integer, the
        sample names will be ["structure", "center_1", "center_2"] and the index
        for "structure" will be ``structure_idx``.
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

    # Define the sample names, with or without the structure index
    if structure_idx is None:  # don't include structure idx in the metadata
        sample_names = ["center_1", "center_2"]
    else:  # include
        sample_names = ["structure", "center_1", "center_2"]

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
    assert utils.num_elements_tensormap(tensor) == np.prod(overlap.shape)

    for _ in range(tests):
        assert test_overlap_matrix_conversion(
            frame, structure_idx, lmax, nmax, overlap, tensor
        )

    return tensor


# ===== Functions to sparsify the symmetric overlap matrix =====


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
    symmetric, false otherwise.
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
    Takes an input TensorMap ``tensor`` that corresponds to the overlap matrix
    for a given structure. Assumes blocks have keys of the form (l1, l2, a1,
    a2), and returns a new TensorMap where off-diagonal blocks are dropped, such
    that the new TensorMap has keys with l1 <= l2 and a1 <= a2.
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
        spherical_harmonics_l=l,
        species_center=SYM_TO_NUM[symbol],
    )
    if structure_idx is None:
        s_idx = tm_block.samples.position((i,))
    else:
        s_idx = tm_block.samples.position((structure_idx, i))
    c_idx = tm_block.components[0].position((m,))
    p_idx = tm_block.properties.position((n,))

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
    if structure_idx is None:
        s_idx = tm_block.samples.position((i1, i2))
    else:
        s_idx = tm_block.samples.position((structure_idx, i1, i2))
    c_idx_1 = tm_block.components[0].position((m1,))
    c_idx_2 = tm_block.components[1].position((m2,))
    p_idx = tm_block.properties.position((n1, n2))

    tm_elem = tm_block.values[s_idx][c_idx_1][c_idx_2][p_idx]
    if print_level > 0:
        print(f"TensorMap: idx", (s_idx, c_idx_1, c_idx_2, p_idx), "coeff", tm_elem)

    return np.isclose(raw_elem, tm_elem)
