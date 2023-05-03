"""
Module for running PySCF calculations and parsing outputs.
"""
import os
from typing import Union, Tuple

import ase
import numpy as np
import pyscf
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

from rholearn import basis, translator


# ===== PySCF input file generation =====


def generate_input_geometry_files(xyz_path: str, save_dir: str):
    """
    Loads an xyz file into a list of ASE Atoms objects, then writes each
    geometry to separate directories at relative paths
    f"{save_dir}/{frame_i}/geometry.xyz", where frame_i is the index of the
    stucture in the list of frames.
    """
    # Make the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Load the xyz file into frames
    frames = ase.io.read(xyz_path, ":")

    # Write each geometry to a separate directory
    for A, frame in enumerate(frames):
        # Create the dir
        structure_dir = os.path.join(save_dir, f"{A}")
        if not os.path.exists(structure_dir):
            os.mkdir(structure_dir)
        # Write the geometry
        ase.io.write(os.path.join(structure_dir, "geometry.xyz"), frame)

    return


# ===== PySCF calculation setup and execution =====


def build_structure_from_ase(
    frame: ase.Atoms, pbc: bool, qm_basis_name: str, **kwargs
) -> Union[pyscf.pbc.gto.cell.Cell, pyscf.gto.mole.Mole]:
    """
    From an ASE Atoms object, builds either a PySCF Cell or Molecule object
    depending on whether ``pbc`` is true or false (respectively), initialized
    with the various parameters.

    If ``pbc`` is true, the ``make_kpoints`` kwarg must be specified.
    """
    if pbc:  # periodic
        # Check k-points have been specified
        if kwargs.get("make_kpoints") is None:
            raise ValueError(
                "Must specify the k-points if running a periodic calculation"
            )

        # Build the PySCF Cell object
        structure = pyscf.pbc.gto.Cell(
            atom=pyscf.pbc.tools.pyscf_ase.ase_atoms_to_pyscf(frame),
            a=frame.cell,
            basis=qm_basis_name,
            make_kpoints=kwargs.get("make_kpoints"),
            exp_to_discard=kwargs.get("exp_to_discard"),
        ).build()

    else:  # molecular
        # Build the PySCF Molecule object
        frame.set_pbc = False

        structure = pyscf.gto.M(
            atom=pyscf_ase.ase_atoms_to_pyscf(frame),
            basis=qm_basis_name,
        ).build()

    return structure


def calculate_density_matrix(
    atom_structure: Union[pyscf.pbc.gto.cell.Cell, pyscf.gto.mole.Mole],
    pbc: bool,
    functional: str,
) -> np.ndarray:
    """
    Runs a RKS and denisty fitting calculation using PySCF to generate a denisty
    matrix for the input ``frame``. If ``pbc`` is true, runs RKS in k-space.
    """
    # Run restricted Kohn-Sham DFT calculation
    if pbc:  # periodic
        import pyscf.pbc.dft as dft

        rks = dft.KRKS(atom_structure)

    else:  # molecular
        import pyscf.dft as dft

        rks = dft.RKS(atom_structure)

    # Set the functional
    rks.xc = functional

    # Run the calculation
    rks = rks.density_fit()
    rks.kernel()

    # Get the density matrix
    density_matrix = rks.make_rdm1()

    if pbc:
        assert density_matrix.shape == (1, atom_structure.nao, atom_structure.nao)
        density_matrix = density_matrix[0]
    else:
        assert density_matrix.shape == (atom_structure.nao, atom_structure.nao)

    return density_matrix


def fit_auxiliary_basis(
    frame: ase.Atoms,
    atom_structure: Union[pyscf.pbc.gto.cell.Cell, pyscf.gto.mole.Mole],
    aux_structure: Union[pyscf.pbc.gto.cell.Cell, pyscf.gto.mole.Mole],
    density_matrix: np.ndarray,
    pbc: bool,
    df_lmax: dict,
    df_nmax: dict,
    reorder_l1: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits an auxiliary basis to the density matrix of the ``atomic_structure``
    and returns the density fitting coefficients, the density projections on the
    auxiliary basis, and the overlap matrix of the auxiliary basis.

    IMPORTANT: there is a certain convention in the ordering of the auxiliary
    basis function coefficents, projections, and overlap matrices that is
    important to note. PySCF orders the irreducible spherical component vectors
    for the l = 1 basis functions as [+1, -1, 0], whilst for all other l values
    the order follows the usual [-l, ... +l] convention. This ordering can be
    made consistent by using the functions :py:func:`reorder_l1_coeffs_vector`
    and :py:func:`reorder_l1_overlap_matrix` in this module.

    :param frame: the ASE Atoms object of the structure for which to fit the
        electron density.
    :param atom_structure: the PySCF Cell or Molecule object of the structure in
        ``frame``.
    :param aux_structure: the PySCF Cell or Molecule object of the auxiliary
        structure used to fit the auxiliary basis.
    :param density_matrix: the density matrix of the ``atom_structure``
        calculated with PySCF.
    :param pbc: whether or not the structure in ``frame`` is periodic.
    :param df_lmax: the maximum angular momentum of the density-fitted auxiliary
        basis for each chemcial species in ``frame``.
    :param df_nmax: the maximum number of radial basis functions of the density-
        fitted auxiliary basis for each chemcial species and l value in
        ``frame``.

    :return tuple: tuple of np.ndarray objects containing the density fitting
        coefficients, the density projections on the auxiliary basis, and the
        overlap matrix of the auxiliary basis, respectively.
    """

    # Define the product structure
    prod_structure = atom_structure + aux_structure

    # Calculate overlap matrix for the auxiliary basis
    overlap = aux_structure.intor("int1e_ovlp_sph")

    # Calculate the 2-center 2-electron integral
    eri2c = aux_structure.intor("int2c2e_sph")

    # Calculate the 3-center 2-electron integral
    eri3c = prod_structure.intor(
        "int3c2e_sph",
        shls_slice=(
            0,
            atom_structure.nbas,
            0,
            atom_structure.nbas,
            atom_structure.nbas,
            atom_structure.nbas + aux_structure.nbas,
        ),
    ).reshape(atom_structure.nao_nr(), atom_structure.nao_nr(), -1)

    # Compute density fitted coefficients
    rho = np.einsum("ijp,ij->p", eri3c, density_matrix)
    coeff = np.linalg.solve(eri2c, rho)

    # Compute density projections on auxiliary functions
    proj = np.dot(overlap, coeff)

    return coeff, proj, overlap


# ===== PySCF output parsing =====


def reorder_l1_coeffs_vector(
    frame: ase.Atoms, coeffs: np.ndarray, lmax: dict, nmax: dict, inplace: bool = False
) -> np.ndarray:
    """
    PySCF outputs the irreducible spherical components of l = 1 basis functions
    in the order m = [+1, -1, 0], while the order for all other l is in the
    standard m = [-l, ..., +l] order. This function reorders the coefficients
    (or projections) for the l = 1 components to be in the standard order m =
    [-1, 0, +1], and returns the result in a numpy array.

    :param frame: :py:class:`ase.Atoms` object that the coefficients have been
        calculated for.
    :param coeffs: :py:class:`np.ndarray` of density expansion coefficients
        outputted by PySCF, where their m = [+1, -1, 0] order convention for l =
        1 basis functions is followed.
    :param lmax: :py:class:`dict` of maximum angular momenta channels for each
        species in the frame.
    :param nmax: :py:class:`dict` of the number of radial basis functions for
        each species and l value.
    """
    # Initialize the reordered coefficients array
    reordered_coeffs = coeffs if inplace else coeffs.copy()
    symbols = frame.get_chemical_symbols()

    # Loop over all atoms in the frame
    for i, a in enumerate(symbols):
        # Loop over all radial channels for each atom, where l = 1
        for n in range(nmax[(a, 1)]):
            # Find the flat index of where the m = -1 coefficient for this atom i and
            # radial channel n **should be**, as this defines the start index of
            # the irreducible spherical component (ISC) vector of length 3
            isc_idx = translator.get_flat_index(
                symbols, lmax, nmax, i=i, l=1, n=n, m=-1
            )

            # Roll the ISC vector to go from [+1, -1, 0] to [-1, 0, +1]
            reordered_coeffs[isc_idx : isc_idx + 3] = np.roll(
                reordered_coeffs[isc_idx : isc_idx + 3], shift=-1
            )

    return reordered_coeffs


def reorder_l1_overlap_matrix(
    frame: ase.Atoms, overlap: np.ndarray, lmax: dict, nmax: dict, inplace: bool = False
) -> np.ndarray:
    """
    PySCF outputs the irreducible spherical components of l = 1 basis functions
    in the order m = [+1, -1, 0], while the order for all other l is in the
    standard m = [-l, ..., +l] order. This function reorders the matrix
    coefficients for the l = 1 components to be in the standard order m = [-1,
    0, +1], and returns the result in a numpy array, with the same 2D shape of

    :param frame: :py:class:`ase.Atoms` object that the overlap matrix has been
        calculated for.
    :param overlap: 2D :py:class:`np.ndarray` of overlap matrix coefficients
        outputted by PySCF, where their m = [+1, -1, 0] order convention for l =
        1 basis functions is followed.
    :param lmax: :py:class:`dict` of maximum angular momenta channels for each
        species in the frame.
    :param nmax: :py:class:`dict` of the number of radial basis functions for
        each species and l value.
    """
    # Initialize the reordered coefficients array
    reordered_overlap = overlap if inplace else overlap.copy()
    symbols = frame.get_chemical_symbols()

    # Loop over all atoms in the frame
    for i, a in enumerate(symbols):
        # Loop over all radial channels for each atom, where l = 1
        for n in range(nmax[(a, 1)]):
            # Find the flat index of where the m = -1 coefficient for this atom i and
            # radial channel n **should be**, as this defines the start index of
            # the irreducible spherical component (ISC) vector of length 3
            isc_idx = translator.get_flat_index(
                symbols, lmax, nmax, i=i, l=1, n=n, m=-1
            )

            # Roll the ISC vector to go from [+1, -1, 0] to [-1, 0, +1]
            # First roll for every row
            reordered_overlap[:, isc_idx : isc_idx + 3] = np.roll(
                reordered_overlap[:, isc_idx : isc_idx + 3], shift=-1, axis=1
            )
            # Then roll for every column
            reordered_overlap[isc_idx : isc_idx + 3, :] = np.roll(
                reordered_overlap[isc_idx : isc_idx + 3, :], shift=-1, axis=0
            )

    return reordered_overlap
