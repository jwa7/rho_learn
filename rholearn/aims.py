"""
A module for generating FHI-AIMS input files and parsing output files, tailored
to calculations of RI electron density coefficients, projections, and overlap
matrices.
"""
import os
from typing import List, Tuple, Optional

import ase
import ase.io
import numpy as np


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
    # overlaps
    for cs_abf_idx in cs_abf_idxs:
        coeffs[cs_abf_idx] *= -1
        projs[cs_abf_idx] *= -1
        overlap[cs_abf_idx, :] *= -1
        overlap[:, cs_abf_idx] *= -1

    # Broadcast the coefficients, projections, and overlaps such that they are
    # ordered according to the ABF indices: 0, 1, 2, ...
    coeffs = coeffs[abf_info[:, 0]]
    projs = projs[abf_info[:, 0]]
    overlap = overlap[abf_info[:, 0], :]
    overlap = overlap[:, abf_info[:, 0]]

    if save_dir is not None:
        np.save(os.path.join(save_dir, "coeffs.npy"), coeffs)
        np.save(os.path.join(save_dir, "projs.npy"), projs)
        np.save(os.path.join(save_dir, "overlap.npy"), overlap)

    if delete_original_files:
        os.remove(os.path.join(aims_output_dir, "ri_restart_coeffs.out"))
        os.remove(os.path.join(aims_output_dir, "ri_projections.out"))
        os.remove(os.path.join(aims_output_dir, "ri_ovlp.out"))

    return coeffs, projs, overlap
