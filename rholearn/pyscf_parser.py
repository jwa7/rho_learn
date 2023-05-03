"""
Module for running PySCF calculations and parsing outputs.
"""
import ase
import numpy as np

from rholearn import translator


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
