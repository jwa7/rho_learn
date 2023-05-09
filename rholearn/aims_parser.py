"""
A module for generating FHI-AIMS input files and parsing output files, tailored
to calculations of RI electron density coefficients, projections, and overlap
matrices.

Note that these parsers have been written to work with the AIMS version 221103
and may not necessarily be compatible with older or newer versions.
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


# ===== AIMS input file generation =====


def generate_input_geometry_files(frames: List[ase.Atoms], save_dir: str):
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


def extract_calculation_info(aims_output_dir: str) -> dict:
    """
    Extracts relevant information from the main AIMS output file "aims.out",
    stored in the directory at absolute path ``aims_output_dir``.

    Relevent information is returned in a dictionary, and may contain, but not
    limited to:
        - the max cpu time and wall clock time for the calculation
        - the number of k-points requested in control.in, and those actually
          used
        - the number of auxiliary basis functions used in the RI fitting
        - the number of SCF cycles run

    :param aims_output_dir: a `str` of the absolute path to the directory
        containing AIMS output files. In particular, this directory must contain
        a file called "aims.out".

    :returns: a `dict` of the relevant information extracted from the AIMS
        output.
    """
    # Initialize a dict to store the extracted information
    calc_info = {
        "scf": {
            "num_cycles": 0,
            "converged": False,
            "charge_density": [],
            "tot_energy_eV": [],
        }
    }
    # Open aims.out file
    with open(os.path.join(aims_output_dir, "aims.out"), "r") as f:
        # Read lines
        lines = f.readlines()

        # Parse each line for relevant information
        for line_i, line in enumerate(lines):
            split = line.split()

            # Net and non-zero number of real-space integration points
            # Example: "| Net number of integration points:    49038"
            if split[:6] == "| Net number of integration points:".split():
                calc_info["num_int_points"] = {
                    "net": int(split[6]),
                    "non-zero": int(lines[line_i + 1].split()[7]),  # on next line
                }

            # Requested and actually used number of k points
            # Example: "| k-points reduced from:        8 to        8"
            if split[:4] == "| k-points reduced from:".split():
                calc_info["k_pts"] = {
                    "requested": int(split[4]),
                    "actual": int(split[6]),
                }

            # Number of auxiliary basis functions after RI fitting.
            # Example: "| Shrink_full_auxil_basis : there are totally 1001
            # partial auxiliary wave functions."
            if split[:6] == "| Shrink_full_auxil_basis : there are totally".split():
                calc_info["num_abfs"] = int(split[6])

            # SCF convergence criteria
            # Example:
            # Self-consistency convergence accuracy:
            # | Change of charge density      :  0.9587E-07
            # | Change of unmixed KS density  :  0.2906E-06
            # | Change of sum of eigenvalues  : -0.2053E-05 eV
            # | Change of total energy        : -0.1160E-11 eV
            if split[:6] == "| Change of charge density :".split():
                calc_info["scf"]["charge_density"].append(float(split[6]))
            if split[:6] == "| Change of total energy :".split():
                calc_info["scf"]["tot_energy_eV"].append(float(split[6]))

            # SCF converged or not, and number of SCF cycles run
            if "Self-consistency cycle converged." in line:
                calc_info["scf"]["converged"] = True

            # Number of SCF cycles run
            # Example:
            # Computational steps:
            # | Number of self-consistency cycles          :           21
            # | Number of SCF (re)initializations          :            1
            if split[:6] == "| Number of self-consistency cycles          :".split():
                calc_info["scf"]["num_cycles"] = int(split[6])

            # Extract the total time for the calculation
            # Example:
            # Detailed time accounting                     :  max(cpu_time)    wall_clock(cpu1)
            # | Total time                                  :       28.746 s          28.901 s
            # | Preparation time                            :        0.090 s           0.173 s
            # | Boundary condition initalization            :        0.031 s           0.031 s
            if split[:4] == "| Total time :".split():
                calc_info["time"] = {
                    "max(cpu_time)": float(split[4]),
                    "wall_clock(cpu1)": float(split[6]),
                }

    return calc_info


def extract_basis_set_info(frame: ase.Atoms, aims_output_dir: str) -> Tuple[dict, dict]:
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


def process_aux_basis_func_data(
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


def calc_density_fitting_error(aims_output_dir: str) -> float:
    """
    Calculates the error in the RI fitted electron density relative to the SCF
    converged electron density. A returned value of 1 corresponds to an error of
    100%. The files required for this calculation, that must be present in
    `aims_output_dir` are as follows:

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
        raise NotADirectoryError(f"The directory {aims_output_dir} does not exist.")
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
    # 3) Tabulated partition function
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


def calc_num_electrons_error() -> float:
    """
    Integrates the real space density and the reconstructed fitted density to
    get a relative error in the number of electrons.
    """
    return
