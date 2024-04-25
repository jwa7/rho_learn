"""
A module for generating FHI-AIMS input files and parsing output files, tailored
to calculations of RI electron density coefficients, projections, and overlap
matrices.

Note that these parsers have been written to work with the AIMS version 221103
and may not necessarily be compatible with older or newer versions.
"""
import os
from typing import Tuple, Optional, Union, List

import ase
import ase.io
import numpy as np

import metatensor
from metatensor import Labels, TensorBlock, TensorMap

from rhocalc import convert
from rholearn import io


# ===== AIMS output file parsing =====


def parse_aims_out(aims_output_dir: str) -> dict:
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
        "prodbas_radial_fn_radii_ang": {},
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

            # Kohn-Sham states, occs, and eigenvalues (eV)
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

            # Extract the default prodbas accuracy
            # Example:
            # "Species H: Using default value for prodbas_acc =   1.000000E-04."
            if split[2:8] == "Using default value for prodbas_acc =".split():
                calc_info["prodbas_acc"][split[1][:-1]] = float(split[8][:-1])

            # Cutoff radius for evaluating overlap matrix
            # Example:
            # "ri_fit: Found cutoff radius for calculating ovlp matrix:   2.00000"
            if (
                split[:8]
                == "ri_fit: Found cutoff radius for calculating ovlp matrix:".split()
            ):
                calc_info["ri_fit_cutoff_radius"] = float(split[8])

            # Extract the charge radii of the product basis radial functions
            if split[:2] == "Product basis:".split():
                assert lines[line_i + 1].split()[:3] == "| charge radius:".split()
                assert lines[line_i + 2].split()[:3] == "| field radius:".split()
                assert lines[line_i + 3].split()[:9] == "| Species   l  charge radius    field radius  multipol momen".split()

                tmp_line_i = line_i + 4
                keep_reading = True
                while keep_reading:
                    tmp_split = lines[tmp_line_i].split()

                    # Break if not valid lines
                    if len(tmp_split) == 0:
                        keep_reading = False
                        break
                    if tmp_split[-1] != "a.u.":
                        keep_reading = False
                        break
                    
                    # This is one of the charge radius lines we want to read from
                    if calc_info["prodbas_radial_fn_radii_ang"].get(tmp_split[1]) is None:
                        calc_info["prodbas_radial_fn_radii_ang"][tmp_split[1]] = [float(tmp_split[3])]
                    else:
                        calc_info["prodbas_radial_fn_radii_ang"][tmp_split[1]].append(float(tmp_split[3]))
                    
                    tmp_line_i += 1

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
    process_what: Optional[List[str]] = None,
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

    return coeffs, projs, ovlp


def get_ks_orbital_info(ks_orb_info: str, as_array: bool = True) -> dict:
    """
    Parses the AIMS output file "ks_orbital_inof.out" produced by setting the
    keyword `ri_fit_write_orbital_info` to True in the AIMS control.in file.

    The number of rows of this file is equal to the number of KS-orbitals, i.e.
    the product of the number of KS states, spin states, and k-points.

    Each column corresponds to, respectively: KS-orbital index, KS state index,
    spin state, k-point index, k-point weight, occupation, and energy
    eigenvalue.

    If `as_array` is True, the data is returned as a structured numpy array.
    Otherwise, it is returned as a dict.
    """
    # Load file
    file = np.loadtxt(ks_orb_info)
    # Check number of columns
    if len(file[0]) == 8:
        with_weights = True
    elif len(file[0]) == 7:
        with_weights = False
    else:
        raise ValueError(
            "expected 7 (without KSO weights) or 8 (with weights)"
            f" columns. Got: {len(file[0])}"
        )
    if as_array:
        if with_weights:
            w_col = [
                ("kso_weight", float),
            ]
        else:
            w_col = []
        info = np.array(
            [tuple(row) for row in file],
            dtype=[
                ("kso_i", int),
                ("state_i", int),
                ("spin_i", int),
                ("kpt_i", int),
                ("k_weight", float),
                ("occ", float),
                ("energy_eV", float),
            ]
            + w_col,
        )

    else:
        info = {}
        for row in file:
            if with_weights:
                i_kso, i_state, i_spin, i_kpt, k_weight, occ, energy, weight = row
            else:
                i_kso, i_state, i_spin, i_kpt, k_weight, occ, energy = row
            info[i_kso] = {
                "i_kso": i_kso,
                "i_state": i_state,
                "i_spin": i_spin,
                "i_kpt": i_kpt,
                "k_weight": k_weight,
                "occ": occ,
                "energy": energy,
            }
            if with_weights:
                info[i_kso].update({"kso_weight": weight})
    return info


def process_aims_ri_results(
    frame: ase.Atoms,
    aims_output_dir: str,
    process_total_density: bool = True,
    process_what: List[str] = None,
    ri_calc_idxs: Optional[List[int]] = [],
    structure_idx: Optional[int] = None,
    save_numpy: bool = False,
) -> None:
    """
    Calls a series of functions to process the results of an AIMS RI calculation
    and saves them in the relative directory `aims_output_dir/processed/`.

    First, the calculation info is extracted from aims.out and stored in a
    dictionary. To this dict are added the RI basis set definition, then the
    dict is pickled to file "calc.pickle".

    Then, the RI basis set coefficients, projections, and overlap matrix are
    processed and saved as numpy arrays if `save_numpy` is true, under filenames
    "ri_coeffs.npy", "ri_projs.npy", and "ri_ovlp.npy" respectively.

    If `ri_calc_idxs` is passed, indicating the indices of the RI calculation
    within the main AIMS calculation, (1-indexed) the processed coefficients and
    projections are indexed by these values, e.g. "c_0001.npy", "w_0001.npy".
    Note that the overlap matrix "s.npy" isn't, as this is fixed for all MOs.

    This function assumes that within the directory `aims_output_dir` there is a
    single basis set definition, such that the overlap matrix is fixed
    regardless of the number of RI fittings that have taken place.

    If passed, `structure_idx` is used as metadata in the constructed TensorMap.

    :param frame: the ASE.atoms frame for which the AIMS RI calculation was
        performed.
    :param aims_output_dir: the path to the directory containing the AIMS output
        files.
    :param process_total_density: optional ``bool`` indicating whether the total
        density should be processed. If True, files with no ri_file_suffix will
        be processed.
    :param process_what: optional list of strings indicating which data to
        process, of "coeffs", "projs", and "ovlp". If None, all data is
        processed.
    :param ri_calc_idxs: optional list of int to indicate the index of the AIMS
        RI calculation within a given AIMS output directory. This may track, for
        instance, the index of the MO for which the RI calculation was
        performed. These calculations are assumed to be run on the same system
        and with the same settings - i.e. for a fixed basis set definition and
        overlap matrix.
    :param structure_idx: optional ``int`` to indicate the index of the
        structure in the dataset. This is used as metadata in the constructed
        TensorMap.
    :param save_numpy: optional ``bool`` indicating whether the processed data
        should be saved as numpy arrays (the intermediate format) as well as
        TensorMaps. If False, the data is saved only in TensorMap format.
    """
    # Create a directory for the processed data
    processed_dir = os.path.join(aims_output_dir, "processed")
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    # Parse aims.out
    aims_out = parse_aims_out(aims_output_dir)

    # Parse basis set info
    lmax, nmax = extract_basis_set_info(frame, aims_output_dir)
    aims_out.update({"basis_set": {"lmax": lmax, "nmax": nmax}})

    # # Parse basis set idx ordering
    # idx_ordering = np.loadtxt(
    #     os.path.join(aims_output_dir, "product_basis_idxs.out"), dtype=int
    # )
    # aims_out.update(
    #     {
    #         "basis_set": {
    #             "def": {"lmax": lmax, "nmax": nmax},
    #             "idxs": idx_ordering,
    #         }
    #     }
    # )

    # Parse KS orbital information
    if os.path.exists(os.path.join(aims_output_dir, "ks_orbital_info.out")):
        aims_out.update(
            {
                "ks_orbitals": get_ks_orbital_info(
                    os.path.join(aims_output_dir, "ks_orbital_info.out"), as_array=True
                )
            }
        )
        aims_out.update({"num_ks_orbitals": aims_out["ks_orbitals"].shape[0]})
        if aims_out.get("num_k_points") is not None:
            n_kpts = aims_out["num_k_points"]["actual"]
        else:
            n_kpts = 1
        assert aims_out["num_ks_orbitals"] == (
            aims_out["num_ks_states"] * aims_out["num_spin_states"] * n_kpts
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

    aims_out["df_error_percent"] = {}
    grid = np.loadtxt(os.path.join(aims_output_dir, "partition_tab.out"))
    for iteration_i, ri_calc_idx in enumerate(ri_calc_idxs):
        # If a run index is passed (i.e. for different KSOs), use this to suffix the
        # coefficients and projections filenames. The overlap matrix only depends on
        # the fixed basis set definition, so does not need to be suffixed.
        ri_calc_suffix = "" if ri_calc_idx is None else f"_{int(ri_calc_idx):04d}"

        # Calculat edensity fitting error
        aims_out["df_error_percent"][
            "total" if ri_calc_idx is None else ri_calc_idx
        ] = get_percent_mae_between_fields(
            input=np.loadtxt(
                os.path.join(aims_output_dir, f"rho_ri{ri_calc_suffix}.out")
            ),
            target=np.loadtxt(
                os.path.join(aims_output_dir, f"rho_ref{ri_calc_suffix}.out")
            ),
            grid=grid,
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

        # Save to file
        if "coeffs" in process_what:
            if save_numpy:
                np.save(
                    os.path.join(processed_dir, f"ri_coeffs{ri_calc_suffix}.npy"),
                    coeffs,
                )
            # Convert to metatensor and save
            coeffs = convert.coeff_vector_ndarray_to_tensormap(
                frame, coeffs, lmax, nmax, structure_idx=structure_idx
            )
            metatensor.save(
                os.path.join(processed_dir, f"ri_coeffs{ri_calc_suffix}.npz"), coeffs
            )
        else:
            assert coeffs is None
        if "projs" in process_what:
            if save_numpy:
                np.save(
                    os.path.join(processed_dir, f"ri_projs{ri_calc_suffix}.npy"), projs
                )
            # Convert to metatensor and save
            projs = convert.coeff_vector_ndarray_to_tensormap(
                frame, projs, lmax, nmax, structure_idx=structure_idx
            )
            metatensor.save(
                os.path.join(processed_dir, f"ri_projs{ri_calc_suffix}.npz"), projs
            )
        else:
            assert projs is None
        if "ovlp" in process_what:
            if save_numpy:
                np.save(os.path.join(processed_dir, "ri_ovlp.npy"), ovlp)
            # Convert to metatensor, drop redundant blocks, and save
            ovlp = convert.overlap_matrix_ndarray_to_tensormap(
                frame, ovlp, lmax, nmax, structure_idx=structure_idx
            )
            ovlp = convert.overlap_drop_redundant_off_diagonal_blocks(ovlp)
            metatensor.save(os.path.join(processed_dir, "ri_ovlp.npz"), ovlp)

        else:
            assert ovlp is None

        # Clear from memory
        del coeffs, projs, ovlp

    # Pickle calc info
    utils.pickle_dict(os.path.join(processed_dir, "calc_info.pickle"), aims_out)

    return

