import os
import ase.io


def generate_aims_input_geometry_files(xyz_file: str, save_dir: str):
    """
    Takes an `xyz_file` (i.e. of .xyz format) that contains the coordinates for
    a set of structures and generates input geometry files in the AIMS format.

    For a set of N structures in `xyz_file`, N new directories in the parent
    directory `save_dir` are created, with relative paths
    f"{save_dir}/{A}/geometry.in", where A is a numeric structure index running
    from 0 -> (N - 1) (inclusive), and corresponding to the order of structures
    in `xyz_file`.

    :param xyz_file: a `str` of the absolute path to the xyz file containing the
        geometries.
    :param save_dir: a `str` of the absolute path to the directory where the
        AIMS input geometry files should be saved.
    """

    # Read the xyz file into frames using ASE
    frames = ase.io.read(xyz_file, ":")
    N = len(frames)

    # Create the save directory if it doesn't already exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for A in range(N):  # Iterate over structures
        # Create a directory named simply by the structure index
        structure_dir = os.path.join(save_dir, f"{A}")
        if not os.path.exists(structure_dir):
            os.mkdir(structure_dir)

        # Write the AIMS input file. By using the ".in" suffix/extension in the
        # filename, ASE will automatically produce an input file that follows
        # AIMS formatting.
        ase.io.write(os.path.join(structure_dir, "geometry.in"), frames[A])
