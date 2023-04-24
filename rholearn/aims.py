import os
from typing import List, Tuple, Optional

import ase
import ase.io


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


def convert_aims_basis_info_to_dict(frame: ase.Atoms, basis_info_file: str) -> Tuple[dict, dict]:
    """
    Takes an AIMS basis info file and converts it into a dictionary of the lmax
    and nmax values for each atom type in the structure.

    :param frame: an :py:class:`ase.Atoms` object corresponding to the structure
        for which the AIMS basis set info should be extracted.
    :param basis_info_file: a `str` of the absolute path to the AIMS basis info
        file for the structure passed in ``frame``.

    :return lmax: a `dict` of the maximum angular momentum for each chemical
        species in ``frame``.
    :return nmax: a `dict` of the maximum radial channel index for each chemical
        species and angular channel in ``frame``.
    """
    # Get the species symbols for the atoms in the frame
    symbols = frame.get_chemical_symbols()

    # Load the basis info file and read the lines
    with open(basis_info_file, "r") as f:
        lines = f.readlines()

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
