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
        "prodbas_acc": {},
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

            # Number of atoms
            # Example:
            # "| Number of atoms                   :       64"
            if split[:5] == "| Number of atoms                   :".split():
                calc_info["num_atoms"] = int(split[5])

            # Net and non-zero number of real-space integration points
            # Example: "| Net number of integration points:    49038"
            if split[:6] == "| Net number of integration points:".split():
                calc_info["num_int_points"] = {
                    "net": int(split[6]),
                    "non-zero": int(lines[line_i + 1].split()[7]),  # on next line
                }
            
            # Number of spin states
            # Example:
            # "| Number of spin channels           :        1"
            if split[:6] == "| Number of spin channels           :".split():
                calc_info["num_spin_states"] = int(split[6])

            # Requested and actually used number of k points
            # Example: "| k-points reduced from:        8 to        8"
            if split[:4] == "| k-points reduced from:".split():
                calc_info["num_k_points"] = {
                    "requested": int(split[4]),
                    "actual": int(split[6]),
                }

            # Number of auxiliary basis functions after RI fitting.
            # Example: "| Shrink_full_auxil_basis : there are totally 1001
            # partial auxiliary wave functions."
            if split[:6] == "| Shrink_full_auxil_basis : there are totally".split():
                calc_info["num_abfs"] = int(split[6])

            # For the following quantities, every time a new SCF loop is
            # encountered in aims.out, the values are overwritten such that only
            # the values from the final SCF loop are returned.

            # SCF convergence criteria
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

            # Fermi level / chemical potential
            # Example:
            # "| Chemical potential (Fermi level):    -9.07068018 eV"
            if split[:5] == "| Chemical potential (Fermi level):".split():
                calc_info["fermi_eV"] = float(split[5])

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

            # Extratc the default prodbas accuracy
            # Example:
            # "Species H: Using default value for prodbas_acc =   1.000000E-04."
            if split[2:8] == "Using default value for prodbas_acc =".split():
                calc_info["prodbas_acc"][split[1][:-1]] = float(split[8][:-1])

            # Cutoff radius for evaluating overlap matrix
            # Example:
            # "ri_fit: Found cutoff radius for calculating ovlp matrix:   2.00000"
            if split[:8] == "ri_fit: Found cutoff radius for calculating ovlp matrix:".split():
                calc_info["ri_fit_cutoff_radius"] = float(split[8])

            # Extract ri_fit info
            # Example:
            # ri_fit: Finished.
            if split[:2] == "ri_fit: Finished.".split():
                calc_info["ri_fit_finished"] = True

    return calc_info


def extract_basis_set_info(frame: ase.Atoms, aims_output_dir: str) -> Tuple[dict, dict]:
    """
    Takes an AIMS basis info file and converts it into a dictionary of the lmax
    and nmax values for each atom type in the structure.

    :param frame: an :py:class:`ase.Atoms` object corresponding to the structure
        for which the AIMS basis set info should be extracted.
    :param aims_output_dir: a `str` of the absolute path to the directory
        containing AIMS output files. In particular, this directory must contain
        a file called "product_basis_definition.out". This contains the
        information of the constructed RI basis set for the structure passed in
        ``frame``.

    :return lmax: a `dict` of the maximum angular momentum for each chemical
        species in ``frame``.
    :return nmax: a `dict` of the maximum radial channel index for each chemical
        species and angular channel in ``frame``.
    """
    # Check the directory containing AIMS output files exists
    if not os.path.exists(aims_output_dir):
        raise ValueError(f"`aims_output_dir` {aims_output_dir} does not exist.")

    # Check the basis info file exists
    basis_info_file = os.path.join(aims_output_dir, "product_basis_definition.out")
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
    ri_calc_idx: Optional[int] = None,
    process_what: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes and returns the coefficients and projections vectors, and overlap
    matirx, from the AIMS output file directory at absolute path
    `aims_output_dir`.

    Assumes that in the directory `aims_output_dir` there exists the following
    files, or a subset of them, output by AIMS using the "ri_fit_*" set of
    keywords:

    - "ri_coeffs.out" contains a single column of the RI fitted auxiliary
        basis function (ABF) coefficients. There should be N entries, where N is
        the number of ABFs. Only processed if "coeffs" is in `process_what`.

    - "ri_projs.out" contains a single column of the RI fitted ABF
        projections. There should be N entries. Only processed if "projs" is in
        `process_what`.

    - "ri_ovlp.out" contains a single column of the elements of the overlap
        matrix between all pairs of ABFs. There should be N^2 entries. Only
        processed if "ovlp" is in `process_what`.

    - "product_basis_idxs.out" contains 5 columns of information about the
        auxiliary basis functions. The columns correspond to, respectively:

        - ABF numeric index, running from 1 to N (inclusive). Note that
            AIMS outputs numeric indices using 1- (not 0-) indexing.
        - atom index, running from 1 to N_atoms (inc). The index of the
            atom that the ABF is centered on.
        - angular momentum l value, running from 0 to l_max (inc). Note
            again that this is an inclusive range, but this time starting at 0
            as this is an l *value*, not a numeric index. As such, there are
            (l_max + 1) possible values of l.
        - radial channel index. This is a cumulative index across all l
            values. For instance, the radial channels for l = 0 are indexed from
            1 to n_{max}(l=0) inclusive, the radial channels for l = 1 are
            indexed from (n_{max}(l=0) + 1) to n_{max}(l=1) inclusive, and so
            on. Note again that this is a numeric index, so is 1-indexed.
        - angular momentum component m value, running from -l to l (inc)
            for the given l value.

    This function performs the following processing of the data. 
    
    First, the 1-indexing of the numeric indices in "product_basis_idxs.out"
    (i.e. ABF, atom, and radial channel indices) is converted to 0-indexing. 
    
    Second, the CS convention is applied to coefficients, projections, and
    overlap matrix elements for the ABFs with m odd and > 0. 
    
    Third, the order of the coefficients, projections, and overlap matrix
    elements are modified to match the numeric indices of the ABFs in
    "product_basis_idxs.out".

    :param aims_output_dir: str for the absolute path to the directory
        containing the AIMS output files from the RI calculation on a single
        structure using keyword "ri_full_output" set to true.
    :param ri_calc_idx: optional ``int`` to indicate the index of the AIMS RI
        calculation. This may track, for instance, the index of the MO for which
        the RI calculation was performed.
    :param process_what: optional list of strings indicating which data to
        process. If None, all data is processed. If a subset of ["coeffs",
        "projs", "ovlp"], only the corresponding data is processed.

    :return: A tuple of the coefficients, projections, and overlap matrix, all
        as numpy arrays. The coefficients and projections are 1D arrays with
        shape (N,) where N is the number of ABFs. The overlap matrix is a 2D
        array with shape (N, N).
    """
    # Check that the AIMS output directory exists
    if not os.path.exists(aims_output_dir):
        raise ValueError(f"`aims_output_dir` {aims_output_dir} does not exist.")

    # If a run index is passed (i.e. for different MOs), use this to suffix the
    # coefficients and projections filenames. The overlap matrix only depends on
    # the fixed basis set definition, so does not need to be suffixed.
    ri_calc_suffix = "" if ri_calc_idx is None else f"_{int(ri_calc_idx):04d}"

    # Choose which data to process
    if process_what is None:
        process_what = ["coeffs", "projs", "ovlp"]
    if not np.all([i in ["coeffs", "projs", "ovlp"] for i in process_what]):
        raise ValueError(
            f"`process_what` {process_what} must be a subset of ['coeffs', "
            f"'projs', 'ovlp']"
        )

    # Load coefficients, projections, and overlap matrix
    coeffs, projs, ovlp = None, None, None
    if "coeffs" in process_what:
        coeffs = np.loadtxt(
            os.path.join(aims_output_dir, f"ri_coeffs{ri_calc_suffix}.out")
        )
    if "projs" in process_what:
        projs = np.loadtxt(
            os.path.join(aims_output_dir, f"ri_projs{ri_calc_suffix}.out")
        )
    if "ovlp" in process_what:
        ovlp = np.loadtxt(os.path.join(aims_output_dir, "ri_ovlp.out"))
        ovlp_dim = int(np.sqrt(ovlp.shape[0]))

    # Check shapes
    if "coeffs" in process_what and "projs" in process_what:
        assert coeffs.shape == projs.shape
    if "coeffs" in process_what and "ovlp" in process_what:
        assert ovlp.shape == (coeffs.shape[0] ** 2,)

    # Reshape overlap into a square matrix and check symmetry. As the overlap
    # matrix should be symmetric, reshaping in either C or F order should give
    # the same result. Here we just use the default C order.
    if "ovlp" in process_what:
        ovlp = ovlp.reshape(ovlp_dim, ovlp_dim)
        assert np.allclose(ovlp, ovlp.T)

    # Load the auxiliary basis function (ABF) information. This is 2D array where
    # columns correspond to, respectively, the auxiliary basis function index, atom
    # index, angular momentum l value, radial channel index, and the angular
    # momentum component m value.
    abf_info = np.loadtxt(
        os.path.join(aims_output_dir, "product_basis_idxs.out"),
        dtype=int,
    )
    # Convert to zero indexing for the columns that correspond to numeric indices.
    # The l and m values need not be modified here.
    abf_info[:, 0] -= 1  # ABF index
    abf_info[:, 1] -= 1  # atom index
    abf_info[:, 3] -= 1  # radial channel index

    # Apply the Condon-Shortley convention to the coefficients, projections, and
    # overlap, for basis functions that correspond to m odd and positive.
    for abf in abf_info:
        abf_idx = abf[0]
        abf_m_value = abf[4]
        if abf_m_value % 2 == 1 and abf_m_value > 0:
            if "coeffs" in process_what:
                coeffs[abf_idx] *= -1
            if "projs" in process_what:
                projs[abf_idx] *= -1
            if "ovlp" in process_what:
                ovlp[abf_idx, :] *= -1
                ovlp[:, abf_idx] *= -1

    # Broadcast the coefficients, projections, and ovlp such that they are
    # ordered according to the ABF indices: 0, 1, 2, ...
    if "coeffs" in process_what:
        coeffs = coeffs[abf_info[:, 0]]
    if "projs" in process_what:
        projs = projs[abf_info[:, 0]]
    if "ovlp" in process_what:
        ovlp = ovlp[abf_info[:, 0], :]
        ovlp = ovlp[:, abf_info[:, 0]]

    return coeffs, projs, ovlp


def calc_density_fitting_error(
    aims_output_dir: str, ri_calc_idx: Optional[int] = None
) -> float:
    """
    Calculates the error in the RI fitted electron density relative to the SCF
    converged electron density. A returned value of 1 corresponds to an error of
    100%.

    The files required for this calculation, that must be present in
    `aims_output_dir`, are as follows:

        - rho_ref.out: SCF converged electron density on real-space grid
        - rho_rebuilt.out: electron density rebuilt from RI coefficients, on
          real-space grid.
        - partition_tab.out: the tabulated partition function - i.e. integration
          weights for the grid points on which the real-space fields are
          evaluated.

    Alternatively, the SCF converged density and rebuilt density may be saved
    under filenames 'rho_ref_xxxx.out' and 'rho_rebuilt_xxxx.out'
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
        f"rho_ref{ri_calc_suffix}.out",
        f"rho_rebuilt{ri_calc_suffix}.out",
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
    rho_ref, rho_ri, partition = [
        np.loadtxt(os.path.join(aims_output_dir, req_file)) for req_file in req_files
    ]

    # Check the xyz coordinates on each row are exactly equivalent
    assert np.all(rho_ref[:, :3] == rho_ri[:, :3]) and np.all(
        rho_ref[:, :3] == partition[:, :3]
    )

    # Now just slice to keep only the final column of data from each file, as
    # this is the only bit we're interested in
    rho_ref = rho_ref[:, 3]
    rho_ri = rho_ri[:, 3]
    partition = partition[:, 3]

    # Get the absolute residual error between the SCF and fitted densities
    error = np.abs(rho_ri - rho_ref)

    # Calculate and return the relative error (normalized by the number of electrons)
    return np.dot(error, partition) / np.dot(rho_ref, partition)


def get_mo_states_info(aims_output_dir: str) -> dict:
    """
    Parses the details of the MO states from the AIMS output file
    `aims_output_dir`/"mo_states_info.out". Each row of this file corresponds
    to, respectively, the numeric MO index (1, ..., num_states, inclusive), the
    KS state index (1 .. num_ks_states, inc), the spin state index (1 ..
    num_spin_states, inc), the k-point index (1 .. num_k_points, inc), the
    k_point weight, the KS occupation number, and the eigenvalue in eV.

    Assumes that the order of the eigenvalues follows the nested loop hierarchy;
    loop over KS states, loop over spin states, loop over k-points. The number
    of KS states, spin states, and k-points are passed in the arguments
    `num_ks_states`, `num_spin_states`, and `num_k_points` respectively.
    `ks_occupations` is a list of the occupation numbers of each KS state.
    """
    # Load the eigenvalues
    state_info = np.loadtxt(os.path.join(aims_output_dir, "mo_states_info.out"))

    mo_state_dict = {}
    for row in state_info:
        # Unpack the row
        mo_idx, ks_state, spin_state, k_point, k_weight, occ, eigval = row[:7]

        # Store the info
        mo_state_dict[int(mo_idx)] = {
            "ks_state": int(ks_state),
            "spin_state": int(spin_state),
            "k_point": int(k_point),
            "k_weight": k_weight,
            "occ": occ, 
            "eig_eV": eigval,
        }

    return mo_state_dict


def calc_density_fitting_error_by_mo_sum(
    aims_output_dir: str,
    mo_idxs: Sequence[int],
    occupations: Sequence[float],
    ref_total_density: str,
    mo_prob_densities: str,
    mo_weights: Optional[Sequence[float]] = None,
) -> float:
    """
    First constructs a total electron density from an occupation-weighted sum of
    the real-space MO probability densities, and then calculates the error in
    this density relative to a reference total electron density. A returned
    value of 1 corresponds to an error of 100%.

    Requires input of the MO orbital indices (running from 1 to n_states
    inclusive) in the `mo_idxs` argument. Also requires input of the occupation
    numbers of each orbital in the `occupations` argument. The MO orbitals may
    be any set of orbitals that sum to the total density, for instance the KS-MO
    orbitals decomposed (or not) by spin state and k-point.

    The reference total electron density can be either the SCF converged total
    density stored in the file "rho_ref.out", or the RI fitted total density in
    file "rho_rebuilt.out". These options are controlled by setting
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
    :param mo_weights: optional ``list`` of ``float`` to indicate the weighting
        for each MO. This is typically used to correct the occupation numberfor
        different k-points. This should be a list of length .

    :return float: the error in the RI fitted density relative to the SCF
        converged density.
    """
    # Check output directory exists
    if not os.path.isdir(aims_output_dir):
        raise NotADirectoryError(f"The directory {aims_output_dir} does not exist.")

    for arg in [ref_total_density, mo_prob_densities]:
        if arg not in ["SCF", "RI"]:
            raise ValueError(f"Invalid argument {arg} passed. Should be 'SCF' or 'RI'.")

    if mo_weights is None:
        mo_weights = [1.0] * len(mo_idxs)

    # Load the reference total density
    if ref_total_density == "SCF":
        ref_rho = np.loadtxt(os.path.join(aims_output_dir, "rho_ref.out"))
    else:
        assert ref_total_density == "RI"
        ref_rho = np.loadtxt(os.path.join(aims_output_dir, "rho_rebuilt.out"))

    # Load the integration weights
    partition = np.loadtxt(os.path.join(aims_output_dir, "partition_tab.out"))
    assert np.all(partition[:, :3] == ref_rho[:, :3])

    # Loop over MO indices and load the MO probability densities
    mo_summed_density = []
    for mo_idx, occ, weight in zip(mo_idxs, occupations, mo_weights):
        if mo_prob_densities == "SCF":
            mo_a = np.loadtxt(
                os.path.join(aims_output_dir, f"rho_scf_{int(mo_idx):04d}.out")
            )
        else:
            assert mo_prob_densities == "RI"
            mo_a = np.loadtxt(
                os.path.join(
                    aims_output_dir, f"rho_rebuilt_ri_{int(mo_idx):04d}.out"
                )
            )
        # Check that the grid point coords are the same
        assert np.all(mo_a[:, :3] == ref_rho[:, :3])

        # Calculate and store the MO density (i.e. probability density *
        # occupation)
        mo_summed_density.append(weight * occ * mo_a[:, 3])

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
    process_what: Sequence[str] = None,
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
    # if not calc["scf"]["converged"]:
    #     io.pickle_dict(os.path.join(processed_dir, "calc.pickle"), calc)
    #     raise ValueError("SCF did not converge")

    # Parse basis set info
    lmax, nmax = extract_basis_set_info(frame, aims_output_dir)
    calc["lmax"] = lmax
    calc["nmax"] = nmax
    calc["df_error"] = {}

    # Parse eigenvalues if printed
    if os.path.exists(os.path.join(aims_output_dir, "mo_states_info.out")):
        calc["mo_states"] = get_mo_states_info(aims_output_dir)
        calc["num_mo_states"] = len(calc["mo_states"].keys())
        assert calc["num_mo_states"] == (
            calc["num_ks_states"] * calc["num_spin_states"] * calc["num_k_points"]["actual"]
        )

    # Now perform processing that is dependent on the RI calculation index
    if isinstance(ri_calc_idxs, int):
        ri_calc_idxs = [ri_calc_idxs]
    elif isinstance(ri_calc_idxs, np.ndarray):
        ri_calc_idxs = ri_calc_idxs.tolist()

    # Include None in the list of indices to process if the total
    # density should be processed too
    if process_total_density:
        ri_calc_idxs = [None] + ri_calc_idxs

    # Choose what to process
    if process_what is None:
        process_what = ["coeffs", "projs", "ovlp"]

    process_overlap_matrix = True
    for iteration_i, ri_calc_idx in enumerate(ri_calc_idxs):

        # Calculate the density fitting error
        calc["df_error"][
            "total" if ri_calc_idx is None else ri_calc_idx
        ] = calc_density_fitting_error(
            aims_output_dir,
            ri_calc_idx=ri_calc_idx,
        )

        # Only process the overlap matrix once
        if "ovlp" in process_what and iteration_i > 0:
            process_what.remove("ovlp")

        # Convert coeffs, projs, overlaps to numpy arrays
        coeffs, projs, ovlp = process_aux_basis_func_data(
            aims_output_dir,
            ri_calc_idx=ri_calc_idx,
            process_what=process_what,
        )

        # If a run index is passed (i.e. for different MOs), use this to suffix the
        # coefficients and projections filenames. The overlap matrix only depends on
        # the fixed basis set definition, so does not need to be suffixed.
        ri_calc_suffix = "" if ri_calc_idx is None else f"_{int(ri_calc_idx):04d}"

        # Save to file
        if "coeffs" in process_what:
            np.save(os.path.join(processed_dir, f"ri_coeffs{ri_calc_suffix}.npy"), coeffs)
        else:
            assert coeffs is None
        if "projs" in process_what:
            np.save(os.path.join(processed_dir, f"ri_projs{ri_calc_suffix}.npy"), projs)
        else:
            assert projs is None
        if "ovlp" in process_what:
            np.save(os.path.join(processed_dir, "ri_ovlp.npy"), ovlp)
        else:
            assert ovlp is None

        # Clear from memory
        del coeffs, projs, ovlp

    # Pickle calc info
    io.pickle_dict(os.path.join(processed_dir, "calc_info.pickle"), calc)

    return


# ===== Convert numpy to AIMS format =====


def coeff_vector_ndarray_to_aims_coeffs(
    aims_output_dir: str, coeffs: np.ndarray, save_dir: Optional[str] = None
) -> np.ndarray:
    """
    Takes a vector of RI coefficients in the standard order convention and
    converts it to the AIMS format.

    This involves reversing the order of the  coefficients contained in
    product_basis_idxs.out, and undoing the application of the Condon-Shortley
    convention. Essentially, this performs the reverse conversion of the
    :py:func:`process_aux_basis_func_data` function in this
    :py:mod:`aims_parser` module, but only applied to the coefficients vector.

    This function requires that the file product_basis_idxs.out exists in the
    directory `aims_output_dir`.

    If `save_dir` is not None, the converted coefficients are saved to this
    directory under the filename "ri_coeffs.in", in the AIMS output file format
    for this data type - i.e. one value per line. The coefficients saved under
    this filename allow the RI fitting procedure to be restarted from them using
    the AIMS keyword "ri_fit_rebuild_from_coeffs".

    :param aims_output_dir: str, absolute path to the directory containing the
        AIMS calculation output files.
    :param coeffs: np.ndarray, vector of RI coefficients in the standard order
        convention.
    :param save_dir: optional str, the absolute path to the directory to save
        the coefficients to. If specified, they are saved as one coefficient per
        line under the filename "ri_coeffs.in".

    :return np.ndarray: vector of RI coefficients in the AIMS format.
    """
    # Check various directories and paths exist
    if not os.path.isdir(aims_output_dir):
        raise NotADirectoryError(f"The directory {aims_output_dir} does not exist.")
    if not os.path.exists(os.path.join(aims_output_dir, "product_basis_idxs.out")):
        raise FileNotFoundError(
            f"The file product_basis_idxs.out does not exist in {aims_output_dir}."
        )
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # Load the auxiliary/product basis function (ABF) information. This is 2D
    # array where columns correspond to, respectively, the auxiliary basis
    # function index, atom index, angular momentum l value, radial channel
    # index, and the angular momentum component m value.
    abf_info = np.loadtxt(
        os.path.join(aims_output_dir, "product_basis_idxs.out"),
        dtype=int,
    )
    abf_idxs = abf_info[:, 0]
    abf_idxs -= 1  # Convert to zero indexing

    # First, re-broadcast the coefficients back to the original AIMS ordering
    reverse_abf_idxs = [np.where(abf_idxs == i)[0][0] for i in range(len(abf_idxs))]
    aims_coeffs = coeffs[reverse_abf_idxs]

    # Second, undo the Condon-Shortley convention for the coefficients of ABFs
    # where m is odd and positive
    for abf in abf_info:
        abf_idx = abf[0]
        abf_m_value = abf[4]
        if abf_m_value % 2 == 1 and abf_m_value > 0:
            aims_coeffs[abf_idx] *= -1

    # Save the coefficient to file "ri_restart_coeffs.out" if `save_dir`
    # specified.
    if save_dir is not None:
        np.savetxt(os.path.join(save_dir, "ri_coeffs.in"), aims_coeffs)

    return aims_coeffs
