"""
A module for generating FHI-AIMS input files and parsing output files, tailored
to calculations of RI electron density coefficients, projections, and overlap
matrices.

Note that these parsers have been written to work with the AIMS version 221103
and may not necessarily be compatible with older or newer versions.
"""
import itertools
import os
from typing import Sequence, Tuple, Optional

import ase
import ase.io
import numpy as np

import equistore
from equistore import Labels, TensorBlock, TensorMap

from rholearn import io

# TODO:
#   - remove redundant processing of the overlap matrix for each MO.
#   - fix file paths here with ri_calc_idx, line 443.


# ===== AIMS input file generation =====


def generate_input_geometry_files(
    frames: Sequence[ase.Atoms],
    save_dir: str,
    structure_idxs: Optional[Sequence[int]] = None,
):
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
    :param structure_idxs: an optional :py:class:`list` of :py:class:`int` of
        the indices of the structures in ``frames`` to generate AIMS input files
        for. If ``None``, then "geometry.in" files are saved in directories
        indexed by 0, 1, ..., N-1, where N is the number of structures in
        ``frames``. If not ``None``, then the explicit indices passed in
        `structure_idxs` are used to index the directories, mapping one-to-one
        to the structures in ``frames``.
    """
    # Create the save directory if it doesn't already exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Define the structure indices used to name the sub-directories
    if structure_idxs is None:
        structure_idxs = range(len(frames))  # 0, 1, ..., N-1
    else:
        if len(frames) != len(structure_idxs):
            raise ValueError(
                f"The number of structures in `frames` ({len(frames)}) must match "
                f"the number of indices in `structure_idxs` ({len(structure_idxs)})"
            )

    for A, frame in zip(structure_idxs, frames):  # Iterate over structures
        # Create a directory named simply by the structure index
        structure_dir = os.path.join(save_dir, f"{A}")
        if not os.path.exists(structure_dir):
            os.mkdir(structure_dir)

        # Write the AIMS input file. By using the ".in" suffix/extension in the
        # filename, ASE will automatically produce an input file that follows
        # AIMS formatting.
        ase.io.write(os.path.join(structure_dir, "geometry.in"), frame)


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
        "aims": {
            "run_dir": aims_output_dir,
        },
        "scf": {
            "num_cycles": 0,
            "converged": False,
            "d_charge_density": [],
            "d_tot_energy_eV": [],
        },
        "ks_states": {},
    }
    # Open aims.out file
    with open(os.path.join(aims_output_dir, "aims.out"), "r") as f:
        # Read lines
        lines = f.readlines()

        # Parse each line for relevant information
        for line_i, line in enumerate(lines):
            split = line.split()

            # AIMS unique identifier for the run
            if split[:2] == "aims_uuid :".split():
                calc_info["aims"]["run_id"] = split[2]

            # AIMS version
            if split[:3] == "FHI-aims version      :".split():
                calc_info["aims"]["version"] = split[3]

            # AIMS commit version
            if split[:3] == "Commit number         :".split():
                calc_info["aims"]["commit"] = split[3]

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
            # Every time a new SCF loop is encountered, the values are
            # overwritten such that only the values from the final SCF loop are
            # returned
            # Example:
            # Self-consistency convergence accuracy:
            # | Change of charge density      :  0.9587E-07
            # | Change of unmixed KS density  :  0.2906E-06
            # | Change of sum of eigenvalues  : -0.2053E-05 eV
            # | Change of total energy        : -0.1160E-11 eV
            if split[:6] == "| Change of charge density :".split():
                calc_info["scf"]["d_charge_density"] = float(split[6])
            if split[:6] == "| Change of total energy :".split():
                calc_info["scf"]["d_tot_energy_eV"] = float(split[6])

            # Number of Kohn-Sham states
            # Example: " Number of Kohn-Sham states (occupied + empty):       11"
            if split[:8] == "| Number of Kohn-Sham states (occupied + empty):".split():
                calc_info["num_ks_states"] = int(split[8])

            # Highest occupied state
            # Example:
            # "Highest occupied state (VBM) at     -9.04639836 eV"
            if split[:5] == "Highest occupied state (VBM) at".split():
                calc_info["homo_eV"] = float(split[5])

            # Lowest unoccupied state
            # Example:
            # "Lowest unoccupied state (CBM) at    -0.05213986 eV"
            if split[:5] == "Lowest unoccupied state (CBM) at".split():
                calc_info["lumo_eV"] = float(split[5])

            # HOMO-LUMO gap
            # Example:
            # "Overall HOMO-LUMO gap:      8.99425850 eV."
            if split[:4] == "Overall HOMO-LUMO gap:".split():
                calc_info["homo_lumo_gap_eV"] = float(split[4])

            # SCF converged?
            if "Self-consistency cycle converged." in line:
                calc_info["scf"]["converged"] = True

            # Number of SCF cycles run
            # Example:
            # Computational steps:
            # | Number of self-consistency cycles          :           21
            # | Number of SCF (re)initializations          :            1
            if split[:6] == "| Number of self-consistency cycles          :".split():
                calc_info["scf"]["num_cycles"] = int(split[6])

            # Kohn-Sham states, occupations, and eigenvalues (eV)
            # Example:
            # State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]
            # 1       2.00000         -19.211485         -522.77110
            # 2       2.00000          -1.038600          -28.26175
            # 3       2.00000          -0.545802          -14.85203
            # ...
            if (
                split[:6]
                == "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]".split()
            ):
                for state_i in range(1, calc_info["num_ks_states"] + 1):
                    state, occ, eig_ha, eig_eV = lines[line_i + state_i].split()
                    assert int(state) == state_i
                    calc_info["ks_states"][int(state)] = {
                        "occ": float(occ),
                        "eig_eV": float(eig_eV),
                    }

            # Final total energy
            # Example:
            # | Total energy of the DFT / Hartree-Fock s.c.f. calculation : -2078.592149198 eV
            if (
                split[:11]
                == "| Total energy of the DFT / Hartree-Fock s.c.f. calculation :".split()
            ):
                calc_info["tot_energy_eV"] = float(split[11])

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
    ri_calc_idx: Optional[int] = None,
    process_overlap_matrix: bool = True,
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
              - ABF numeric index, running from 1 to N (inclusive). Note that
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

    This function performs the following processing of the data. First, the
    1-indexing of the numeric indices in "idx_prodbas_details.out" (i.e. ABF,
    atom, and radial channel indices) and "prodbas_condon_shotley_list.out"
    (i.e. the ABF indices) is converted to 0-indexing. Second, the CS convention
    is applied to coefficients, projections, and overlap matrix elements for the
    ABFs with m > 0. Third, the order of the coefficients and projections are
    modified to match the numeric indices of the ABFs in
    "idx_prodbas_details.out".

    :param aims_output_dir: str for the absolute path to the directory
        containing the AIMS output files from the RI calculation on a single
        structure using keyword "ri_full_output" set to true.
    :param save_dir: optional ``str`` to the absolute path to the directory in
        which to save the processed coefficients, projections, and overlap
        matrix.
    :param ri_calc_idx: optional ``int`` to indicate the index of the AIMS RI
        calculation. This may track, for instance, the index of the MO for which
        the RI calculation was performed.
    :param process_overlap_matrix: optional ``bool`` to indicate whether or not
        to load and process the overlap matrix. If there exists multiple RI
        fittings in `aims_output_dir` for a fixed basis set definition, then the
        overlap matrix for all of these fittings will be the same and as such
        the overlap matrix only needs to be processed once. If this is the case,
        then this argument should be set to ``False`` for all but one of the
        processing steps.
    :param delete_original_files: optional ``bool`` to indicate whether or not
        to delete the original AIMS output files corresponding to the
        coefficients, projections, and overlap matrix, i.e.
        "ri_restart_coeffs.out", "ri_projections.out", and "ri_ovlp.out".

    :return: A tuple of the coefficients, projections, and overlap matrix, all
        as numpy arrays. The coefficients and projections are 1D arrays with
        shape (N,), where N is the number of ABFs. The overlap matrix is a 2D
        array with shape (N, N).
    """
    # Check that the AIMS output directory exists
    if not os.path.exists(aims_output_dir):
        raise ValueError(f"`aims_output_dir` {aims_output_dir} does not exist.")

    # If a run index is passed (i.e. for different MOs), use this to suffix the
    # coefficients and projections filenames. The overlap matrix only depends on
    # the fixed basis set definition, so does not need to be suffixed.
    ri_calc_suffix = "" if ri_calc_idx is None else f"_{int(ri_calc_idx):04d}"

    # Load coefficients, projections, and overlap matrix
    coeffs = np.loadtxt(
        os.path.join(aims_output_dir, f"ri_restart_coeffs{ri_calc_suffix}.out")
    )
    projs = np.loadtxt(
        os.path.join(aims_output_dir, f"ri_projections{ri_calc_suffix}.out")
    )
    if process_overlap_matrix:
        overlap = np.loadtxt(os.path.join(aims_output_dir, "ri_ovlp.out"))
    else:
        overlap = None

    # Check shapes
    assert coeffs.shape == projs.shape
    if process_overlap_matrix:
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
    # to m > 0. In AIMS version > 221103, in file /src/cartesian_ylm.f90, this is
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
        if process_overlap_matrix:
            overlap[cs_abf_idx, :] *= -1
            overlap[:, cs_abf_idx] *= -1

    # Broadcast the coefficients, projections, and overlap such that they are
    # ordered according to the ABF indices: 0, 1, 2, ...
    coeffs = coeffs[abf_info[:, 0]]
    projs = projs[abf_info[:, 0]]
    if process_overlap_matrix:
        overlap = overlap[abf_info[:, 0], :]
        overlap = overlap[:, abf_info[:, 0]]

    # Save files if requested
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # TODO: fix file paths here with ri_calc_idx
        np.save(os.path.join(save_dir, "coeffs.npy"), coeffs)
        np.save(os.path.join(save_dir, "projs.npy"), projs)
        if process_overlap_matrix:
            np.save(os.path.join(save_dir, "overlap.npy"), overlap)

    # Delete original files if requested
    if delete_original_files:
        os.remove(
            os.path.join(aims_output_dir, f"ri_restart_coeffs{ri_calc_suffix}.out")
        )
        os.remove(os.path.join(aims_output_dir, f"ri_projections{ri_calc_suffix}.out"))
        if process_overlap_matrix:
            os.remove(os.path.join(aims_output_dir, "ri_ovlp.out"))

    return coeffs, projs, overlap  # overlap is None if process_overlap_matrix = False


def calc_density_fitting_error(
    aims_output_dir: str, ri_calc_idx: Optional[int] = None
) -> float:
    """
    Calculates the error in the RI fitted electron density relative to the SCF
    converged electron density. A returned value of 1 corresponds to an error of
    100%.

    The files required for this calculation, that must be present in
    `aims_output_dir`, are as follows:

        - rho_scf.out: SCF converged electron density.
        - rho_rebuilt_ri.out: RI fitted electron density.
        - partition_tab.out: the tabulated partition function.

    Alternatively, the SCF converged density and rebuilt density may be saved
    under filenames 'rho_scf_xxx.out' and 'rho_rebuilt_ri_xxxx.out'
    respectively, corresponding to, for example, the mod squared 'densities' of
    a single molecular orbital. In this case, the keyword argument `ri_calc_idx`
    should be passed, specifying the integer 'xxxx' in the filenames.

    :param aims_output_dir: str for the absolute path to the directory
        containing the AIMS output files from the RI calculation on a single
        structure using keyword "ri_full_output" set to true.
    :param ri_calc_idx: optional ``int`` to indicate the index of the AIMS RI
        calculation within a given AIMS output directory. This may track, for
        instance, the index of the MO for which the RI calculation was
        performed.

    :return float: the error in the RI fitted density relative to the SCF
        converged density.
    """
    # Check output directory exists
    if not os.path.isdir(aims_output_dir):
        raise NotADirectoryError(f"The directory {aims_output_dir} does not exist.")

    # If a run index is passed (i.e. for different MOs), use this to suffix the
    # coefficients and projections filenames. The overlap matrix only depends on
    # the fixed basis set definition, so does not need to be suffixed.
    ri_calc_suffix = "" if ri_calc_idx is None else f"_{int(ri_calc_idx):04d}"

    # Check required files exist
    req_files = [
        f"rho_scf{ri_calc_suffix}.out",
        f"rho_rebuilt_ri{ri_calc_suffix}.out",
        "partition_tab.out",
    ]
    for req_file in req_files:
        if not os.path.exists(os.path.join(aims_output_dir, req_file)):
            raise FileNotFoundError(
                f"The file {req_file} does not exist in {aims_output_dir}."
            )

    # Load the real-space data. Each row corresponds to x, y, z coordinates
    # followed by the value. The files loaded, respectively, are 1) SCF
    # converged electron density, 2) RI fitted (rebuilt) electron density, 3)
    # Tabulated partition function.
    scf_rho, ri_rho, partition = [
        np.loadtxt(os.path.join(aims_output_dir, req_file)) for req_file in req_files
    ]

    # Check the xyz coordinates on each row are exactly equivalent
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

    # Calculate and return the relative error (normalized by the number of electrons)
    return np.dot(error, partition) / np.dot(scf_rho, partition)


def calc_density_fitting_error_by_mo_sum(
    aims_output_dir: str,
    mo_idxs: Sequence[int],
    occupations: Sequence[float],
    ref_total_density: str,
    mo_prob_densities: str,
) -> float:
    """
    First constructs a total electron density from an occupation-weighted sum of
    the real-space MO probability densities, and then calculates the error in
    this density relative to a reference total electron density. A returned
    value of 1 corresponds to an error of 100%.

    Requires input of the Kohn-Sham MO orbital indices (running from 1 to
    n_states inclusive) in the `ri_calc_idxs` argument. Also requires input of
    the occupation number of each orbital in the `occupations` argument.

    The reference total electron density can be either the SCF converged total
    density stored in the file "rho_scf.out", or the RI fitted total density in
    file "rho_rebuilt_ri.out". These options are controlled by setting
    `ref_total_density` to "SCF" or "RI" respectively.

    The molecular orbital probabilty densities used to construct the total
    density can be either the SCF converged probability densities stored in
    files "rho_scf_xxxx.out", or the RI fitted probability densities in files
    "rho_rebuilt_ri_xxxx.out". These options are also controlled by setting
    `mo_prob_densities` to "SCF" or "RI" respectively. "xxxx" points to a string
    suffix corresponding to the each of the Kohn-Sham MO indices passed in
    `mo_idxs`.

    Also required is that the file "partition_tab.out" is present in
    `aims_output_dir`. This contains the integration weights for each grid point
    in real space.

    :param aims_output_dir: str for the absolute path to the directory
        containing the AIMS output files from the RI calculation on a single
        structure using keyword "ri_full_output" set to true.
    :param mo_idxs: list of ``int`` to indicate the Kohn-Sham molecular orbital
        indices for which MO probability densities exist.
    :param occupations: list of ``float`` to indicate the occupation number of
        each MO in `mo_idxs`.
    :param ref_total_density: ``str`` to indicate the reference total electron
        density to which the error in the other density is calculated. Must be
        either "SCF" or "RI".
    :param mo_prob_densities: ``str`` to indicate the type of MO probability
        densities to use to construct the total density and compare to the
        reference total density. Must be either "SCF" or "RI".

    :return float: the error in the RI fitted density relative to the SCF
        converged density.
    """
    # Check output directory exists
    if not os.path.isdir(aims_output_dir):
        raise NotADirectoryError(f"The directory {aims_output_dir} does not exist.")

    for arg in [ref_total_density, mo_prob_densities]:
        if arg not in ["SCF", "RI"]:
            raise ValueError(f"Invalid argument {arg} passed. Should be 'SCF' or 'RI'.")

    # Load the reference total density
    if ref_total_density == "SCF":
        ref_rho = np.loadtxt(os.path.join(aims_output_dir, "rho_scf.out"))
    else:
        assert ref_total_density == "RI"
        ref_rho = np.loadtxt(os.path.join(aims_output_dir, "rho_rebuilt_ri.out"))

    # Load the integration weights
    partition = np.loadtxt(os.path.join(aims_output_dir, "partition_tab.out"))
    assert np.all(partition[:, :3] == ref_rho[:, :3])

    # Loop over MO indices and load the MO probability densities
    mo_summed_density = []
    for mo_idx, occ in zip(mo_idxs, occupations):
        if mo_prob_densities == "SCF":
            c_a = np.loadtxt(
                os.path.join(aims_output_dir, f"rho_scf_{int(mo_idx):04d}.out")
            )
        else:
            assert mo_prob_densities == "RI"
            c_a = np.loadtxt(
                os.path.join(
                    aims_output_dir, f"rho_rebuilt_ri_{int(mo_idx):04d}.out"
                )
            )
        # Check that the grid point coords are the same
        assert np.all(c_a[:, :3] == ref_rho[:, :3])

        # Calculate and store the MO density (i.e. probability density * occupation)
        mo_summed_density.append(c_a[:, 3] * occ)

    # Sum the MO densities at each grid point
    mo_summed_density = np.sum(mo_summed_density, axis=0)

    # Now it's confirmed that the grid point coordinates are consistent, throw
    # away the grid points for the ref total density and integration weights
    ref_rho = ref_rho[:, 3]
    partition = partition[:, 3]

    # Get the absolute residual error between the ref and mo-built densities
    error = np.abs(mo_summed_density - ref_rho)

    # Calculate and return the relative error (normalized by the number of electrons)
    return np.dot(error, partition) / np.dot(ref_rho, partition)


def process_aims_ri_results(
    frame: ase.Atoms,
    aims_output_dir: str,
    process_total_density: bool = True,
    ri_calc_idxs: Optional[Sequence[int]] = [],
) -> None:
    """
    Calls a series of functions to process the results of an AIMS RI calculation
    and saves them in the relative directory `aims_output_dir/processed/`.

    First, the calculation info is extracted from aims.out and stored in a
    dictionary. To this dict are added the RI basis set definition, then the
    dict is pickled to file "calc.pickle".

    Then, the RI basis set coefficients, projections, and overlap matrix are
    processed and saved as numpy arrays "c.npy", "w.npy", and "s.npy"
    respectively.

    If `ri_calc_idxs` is passed, indicating the indices of the RI calculation
    within the main AIMS calculation, the processed coefficients and projections
    are indexed by these values, e.g. "c_0000.npy", "w_0000.npy". Note that the
    overlap matrix "s.npy" isn't, as this is fixed for all MOs.

    This function assumes that within the directory `aims_output_dir` there is a
    single basis set definition, such that the overlap matrix is fixed
    regardless of the number of RI fittings that have taken place.

    :param frame: the ASE.atoms frame for which the AIMS RI calculation was
        performed.
    :param aims_output_dir: the path to the directory containing the AIMS output
        files.
    :param ri_calc_idx: optional ``int`` to indicate the index of the AIMS RI
        calculation within a given AIMS output directory. This may track, for
        instance, the index of the MO for which the RI calculation was
        performed.
    """
    # Create a directory for the processed data
    processed_dir = os.path.join(aims_output_dir, "processed")
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    # Parse calc info
    calc = extract_calculation_info(aims_output_dir)
    if not calc["scf"]["converged"]:
        io.pickle_dict(os.path.join(processed_dir, "calc.pickle"), calc)
        raise ValueError("SCF did not converge")

    # Parse basis set info
    lmax, nmax = extract_basis_set_info(frame, aims_output_dir)
    calc["lmax"] = lmax
    calc["nmax"] = nmax
    calc["df_error"] = {}

    # Now perform processing that is dependent on the RI calculation index
    if isinstance(ri_calc_idxs, int):
        ri_calc_idxs = [ri_calc_idxs]
    elif isinstance(ri_calc_idxs, np.ndarray):
        ri_calc_idxs = ri_calc_idxs.tolist()

    # Include None in the list of indices to process if the total
    # density should be processed too
    if process_total_density:
        ri_calc_idxs = [None] + ri_calc_idxs

    process_overlap_matrix = True
    for ri_calc_idx in ri_calc_idxs:
        # Calculate the density fitting error
        calc["df_error"][
            "total" if ri_calc_idx is None else ri_calc_idx
        ] = calc_density_fitting_error(
            aims_output_dir,
            ri_calc_idx=ri_calc_idx,
        )

        # Convert coeffs, projs, overlaps to numpy arrays
        c, w, s = process_aux_basis_func_data(
            aims_output_dir,
            ri_calc_idx=ri_calc_idx,
            process_overlap_matrix=process_overlap_matrix,
        )

        # If a run index is passed (i.e. for different MOs), use this to suffix the
        # coefficients and projections filenames. The overlap matrix only depends on
        # the fixed basis set definition, so does not need to be suffixed.
        ri_calc_suffix = "" if ri_calc_idx is None else f"_{int(ri_calc_idx):04d}"

        # Save to file
        np.save(os.path.join(processed_dir, f"c{ri_calc_suffix}.npy"), c)
        np.save(os.path.join(processed_dir, f"w{ri_calc_suffix}.npy"), w)
        np.save(os.path.join(processed_dir, "s.npy"), s)

        # Clear from memory
        del c, w, s

        # Only process the overlap matrix once
        process_overlap_matrix = False

    # Pickle calc info
    io.pickle_dict(os.path.join(processed_dir, "calc.pickle"), calc)

    return


# ===== Convert numpy to AIMS format =====


def coeff_vector_ndarray_to_aims_coeffs(
    aims_output_dir: str, coeffs: np.ndarray, save_dir: Optional[str] = None
) -> np.ndarray:
    """
    Takes a vector of RI coefficients in the standard order convention and
    converts it to the AIMS format.

    This involves reversing the order of the  coefficients contained in
    idx_prodbas_details.out, and undoing the application of the Condon-Shortley
    convention. Essentially, this performs the reverse conversion of the
    :py:func:`process_aux_basis_func_data` function in this
    :py:mod:`aims_parser` module, but only applied to the coefficients vector.

    This function requires that the files idx_prodbas_details.out and
    prodbas_condon_shotley_list.out exist in the directory `aims_output_dir`.

    If `save_dir` is not None, the converted coefficients are saved to this
    directory under the filename "ri_restart_coeffs.out", in the AIMS output
    file format for this data type - i.e. one value per line. The coefficients
    saved under this filename allow the RI fitting procedure to be restarted
    from them.

    :param aims_output_dir: str, absolute path to the directory containing the
        AIMS calculation output files.
    :param coeffs: np.ndarray, vector of RI coefficients in the standard order
        convention.
    :param save_dir: optional str, the absolute path to the directory to save
        the coefficients to. If specified, they are saved as one coefficient per
        line under the filename "ri_restart_coeffs.out".

    :return np.ndarray: vector of RI coefficients in the AIMS format.
    """
    # Check various directories and paths exist
    if not os.path.isdir(aims_output_dir):
        raise NotADirectoryError(f"The directory {aims_output_dir} does not exist.")
    if not os.path.exists(os.path.join(aims_output_dir, "idx_prodbas_details.out")):
        raise FileNotFoundError(
            f"The file idx_prodbas_details.out does not exist in {aims_output_dir}."
        )
    if not os.path.exists(
        os.path.join(aims_output_dir, "prodbas_condon_shotley_list.out")
    ):
        raise FileNotFoundError(
            f"The file prodbas_condon_shotley_list.out does not exist in {aims_output_dir}."
        )
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # Load the auxiliary basis function (ABF) information. This is 2D array where
    # columns correspond to, respectively, the auxiliary basis function index, atom
    # index, angular momentum l value, radial channel index, and the angular
    # momentum component m value.
    abf_idxs = np.loadtxt(
        os.path.join(aims_output_dir, "idx_prodbas_details.out"),
        dtype=int,
    )[:, 0]
    abf_idxs -= 1  # Convert to zero indexing

    # Load the indices of the auxiliary basis functions to which the Condon-Shortley
    # convention should be applied
    cs_abf_idxs = np.loadtxt(
        os.path.join(aims_output_dir, "prodbas_condon_shotley_list.out"),
        dtype=int,
    )
    cs_abf_idxs -= 1  # Convert to zero indexing

    # First, re-broadcast the coefficients back to the original AIMS ordering
    reverse_abf_idxs = [np.where(abf_idxs == i)[0][0] for i in range(len(abf_idxs))]
    aims_coeffs = coeffs[reverse_abf_idxs]

    # Second, undo the Condon-Shortley convention for the coefficients of the ABFs
    # in `cs_abf_idxs`
    for cs_abf_idx in cs_abf_idxs:
        aims_coeffs[cs_abf_idx] *= -1

    # Save the coefficient to file "ri_restart_coeffs.out" if `save_dir`
    # specified.
    if save_dir is not None:
        np.savetxt(os.path.join(save_dir, "ri_restart_coeffs.out"), aims_coeffs)

    return aims_coeffs
