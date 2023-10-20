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