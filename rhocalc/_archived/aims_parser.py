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


def df_error_by_ks_orb_prob_dens_sum(
    aims_output_dir: str,
    ks_orb_idxs: Sequence[int],
    occs: Sequence[float],
    ref_total_density: str,
    ks_orb_prob_dens: str,
    ks_orb_weights: Optional[Sequence[float]] = None,
) -> float:
    """
    First constructs a total electron density from an occupation-weighted sum of
    the real-space KS-orbital probability densities, and then calculates the
    error in this density relative to a reference total electron density. A
    returned value of 1 corresponds to an error of 100%.

    Requires input of the KS-orbital indices (running from 1 to n_states
    inclusive) in the `ks_orb_idxs` argument. Also requires input of the
    occupation numbers of each orbital in the `occs` argument. The KS-orbitals
    may be any set of orbitals that sum to the total density, for instance the
    KS-orbitals decomposed (or not) by spin state and k-point.

    The reference total electron density can be either the SCF converged total
    density stored in the file "rho_ref.out", or the RI fitted total density in
    file "rho_rebuilt.out". These options are controlled by setting
    `ref_total_density` to "SCF" or "RI" respectively.

    The molecular orbital probabilty densities used to construct the total
    density can be either the SCF converged probability densities stored in
    files "rho_ref_xxxx.out", or the RI fitted probability densities in files
    "rho_rebuilt_xxxx.out". These options are also controlled by setting
    `ks_orb_prob_dens` to "SCF" or "RI" respectively. "xxxx" points to a string
    suffix corresponding to the each of the KS-orbital indices passed in
    `ks_orb_idxs`.

    Also required is that the file "partition_tab.out" is present in
    `aims_output_dir`. This contains the integration weights for each grid point
    in real space.

    :param aims_output_dir: str for the absolute path to the directory
        containing the AIMS output files.
    :param ks_orb_idxs: list of ``int`` to indicate the KS-orbital indices for
        which probability densities exist.
    :param occs: list of ``float`` to indicate the occupation number of each
        KS-orbital in `ks_orb_idxs`.
    :param ref_total_density: ``str`` to indicate the reference total electron
        density to which the error in the other density is calculated. Must be
        either "SCF" or "RI".
    :param ks_orb_prob_dens: ``str`` to indicate the type of KS-orbital
        probability densities to use to construct the total density and compare
        to the reference total density. Must be either "SCF" or "RI".
    :param ks_orb_weights: optional ``list`` of ``float`` to indicate the
        weighting for each KS-orbital. This is typically used to correct the
        occupation number for different k-points.

    :return float: the error in the RI fitted density relative to the SCF
        converged density. A value of 1 corresponds to a 100% error.
    """
    # Check output directory exists
    if not os.path.isdir(aims_output_dir):
        raise NotADirectoryError(f"The directory {aims_output_dir} does not exist.")

    for arg in [ref_total_density, ks_orb_prob_dens]:
        if arg not in ["SCF", "RI"]:
            raise ValueError(f"Invalid argument {arg} passed. Should be 'SCF' or 'RI'.")

    if ks_orb_weights is None:
        ks_orb_weights = [1.0] * len(ks_orb_idxs)

    # Load the reference total density
    if ref_total_density == "SCF":
        rho_ref = np.loadtxt(os.path.join(aims_output_dir, "rho_ref.out"))
    else:
        assert ref_total_density == "RI"
        rho_ref = np.loadtxt(os.path.join(aims_output_dir, "rho_rebuilt.out"))

    # Load the integration weights
    partition = np.loadtxt(os.path.join(aims_output_dir, "partition_tab.out"))
    assert np.all(partition[:, :3] == rho_ref[:, :3])

    # Loop over MO indices and load the KS-orbital probability densities
    ks_orb_prob_dens_summed = []
    for ks_orb_idx, occ, weight in zip(ks_orb_idxs, occs, ks_orb_weights):
        if ks_orb_prob_dens == "SCF":
            kso_a = np.loadtxt(
                os.path.join(aims_output_dir, f"rho_ref_{int(ks_orb_idx):04d}.out")
            )
        else:
            assert ks_orb_prob_dens == "RI"
            kso_a = np.loadtxt(
                os.path.join(
                    aims_output_dir, f"rho_rebuilt_{int(ks_orb_idx):04d}.out"
                )
            )
        # Check that the grid point coords are the same
        assert np.all(kso_a[:, :3] == rho_ref[:, :3])

        # Calculate and store the MO density (i.e. probability density *
        # occupation)
        ks_orb_prob_dens_summed.append(weight * occ * kso_a[:, 3])

    # Sum the MO densities at each grid point
    ks_orb_prob_dens_summed = np.sum(ks_orb_prob_dens_summed, axis=0)

    # Now it's confirmed that the grid point coordinates are consistent, throw
    # away the grid points for the ref total density and integration weights
    rho_ref = rho_ref[:, 3]
    partition = partition[:, 3]

    # Get the absolute residual error between the ref and mo-built densities
    error = np.abs(ks_orb_prob_dens_summed - rho_ref)

    # Calculate and return the relative error (normalized by the number of electrons)
    return np.dot(error, partition) / np.dot(rho_ref, partition)


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


def coeff_vector_ndarray_to_aims_coeffs(
    coeffs: np.ndarray, basis_set_idxs: np.ndarray, save_dir: Optional[str] = None
) -> np.ndarray:
    """
    Takes a vector of RI coefficients in the standard order convention and
    converts it to the AIMS format.

    This involves reversing the order of the basis functions contained in
    product_basis_idxs.out, and undoing the application of the Condon-Shortley
    convention. Essentially, this performs the reverse conversion of the
    :py:func:`process_aux_basis_func_data` function in this
    :py:mod:`aims_parser` module, but only applied to the coefficients vector.

    ``basis_set_idxs`` must be passed as a 2D numpy array containing the
    information read directly from file "product_basis_idxs.out". Columns
    correspond to, respectively: the auxiliary basis function index, atom index,
    angular momentum l value, radial channel index, and the angular momentum
    component m value.

    If `save_dir` is passed, the converted coefficients are saved to this
    directory under the filename "ri_coeffs.in", in the AIMS output file format
    for this data type - i.e. one value per line. The coefficients saved under
    this filename allow the RI fitting procedure to be restarted from them using
    the AIMS keyword "ri_fit_rebuild_from_coeffs".

    :param coeffs: np.ndarray, vector of RI coefficients in the standard order
        convention.
    :param basis_set_idxs: a 2D numpy array containing the basis set indices are
        their ordering, as read driectly from the file "product_basis_idxs.out"
        file output by AIMS using the "ri_fit_*" keywords.
    :param save_dir: optional str, the absolute path to the directory to save
        the coefficients to. If specified, they are saved as one coefficient per
        line under the filename "ri_coeffs.in".

    :return np.ndarray: vector of RI coefficients in the AIMS format.
    """
    abf_info = basis_set_idxs.copy()
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
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savetxt(os.path.join(save_dir, "ri_coeffs.in"), aims_coeffs)

    return aims_coeffs


# ===== Converting vectors FHI-aims to rho_learn

# # Load the auxiliary basis function (ABF) information. This is 2D array where
# # columns correspond to, respectively, the auxiliary basis function index, atom
# # index, angular momentum l value, radial channel index, and the angular
# # momentum component m value.
# abf_info = np.loadtxt(
#     os.path.join(aims_output_dir, "product_basis_idxs.out"),
#     dtype=int,
# )
# # Convert to zero indexing for the columns that correspond to numeric indices.
# # The l and m values need not be modified here.
# abf_info[:, 0] -= 1  # ABF index
# abf_info[:, 1] -= 1  # atom index
# abf_info[:, 3] -= 1  # radial channel index

# # Apply the Condon-Shortley convention to the coefficients, projections, and
# # overlap, for basis functions that correspond to m odd and positive.
# for abf in abf_info:
#     abf_idx = abf[0]
#     abf_m_value = abf[4]
#     if abf_m_value % 2 == 1 and abf_m_value > 0:
#         if "coeffs" in process_what:
#             coeffs[abf_idx] *= -1
#         if "projs" in process_what:
#             projs[abf_idx] *= -1
#         if "ovlp" in process_what:
#             ovlp[abf_idx, :] *= -1
#             ovlp[:, abf_idx] *= -1

# # Broadcast the coefficients, projections, and ovlp such that they are
# # ordered according to the ABF indices: 0, 1, 2, ...
# if "coeffs" in process_what:
#     coeffs = coeffs[abf_info[:, 0]]
# if "projs" in process_what:
#     projs = projs[abf_info[:, 0]]
# if "ovlp" in process_what:
#     ovlp = ovlp[abf_info[:, 0], :]
#     ovlp = ovlp[:, abf_info[:, 0]]