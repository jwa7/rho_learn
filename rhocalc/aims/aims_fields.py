"""
Module containing helper functions for constructing scalar fields derived from
Kohn-Sham orbitals.
"""
import os
import numpy as np

from rhocalc.aims import aims_parser


def get_kso_weight_vector_e_density(kso_info_path: str) -> np.ndarray:
    """
    Returns the KS-orbital weight vector for constructing the electron density.

    Each KSO weight is given by the product of the k-point weight and the electronic
    occupation for each KS-orbital.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    return np.array([kso["k_weight"] * kso["occ"] for kso in kso_info])


def get_kso_weight_vector_ldos(
    kso_info_path: str, gaussian_width: float, target_energy: float = None
) -> np.ndarray:
    """
    Returns the KS-orbital weight vector for constructing the Local Density of
    States (LDOS).

    Each KSO weight is given by the evaluation of a Gaussian function, of width
    `gaussian_width`, centered on the KS-orbital eigenvalue and evaluated at the
    `target_energy`.

    Where `target_energy` is not passed, the eigenvalue of the highest occupied
    KS-orbital is used.

    `gaussian_width`, and `target_energy`  must be passed in units of eV.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    if target_energy is None:  # Find the energy of the HOMO
        homo_idx = aims_parser.find_homo_kso_idxs(kso_info)[0]
        target_energy = kso_info[homo_idx - 1]["energy_eV"]  # 1-indexing!

    return np.array(
        [
            evaluate_gaussian(
                target=target_energy, center=kso_eig, width=gaussian_width
            )
            for kso_eig in kso_info["energy_eV"]
        ]
    )


def get_kso_weight_vector_ildos(
    kso_info_path: str,
    gaussian_width: float,
    biasing_voltage: float,
    energy_grid_points: int = 100,
    target_energy: float = None,
) -> np.ndarray:
    """
    Returns the KS-orbital weight vector for constructing the Local Density of
    States (LDOS).

    Each KSO weight is given by the sum of Gaussian functions, each of width
    `gaussian_width`, centered on the KS-orbital eigenvalue and evaluated on a
    discrete energy grid from the `target_energy` to `target_energy` +
    `biasing_voltage`.

    Where `target_energy` is not passed, the eigenvalue of the highest occupied
    KS-orbital is used.

    `biasing_voltage`, `gaussian_width`, and `target_energy`  must be passed in
    units of eV.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    if target_energy is None:  # Find the energy of the HOMO
        homo_idx = aims_parser.find_homo_kso_idxs(kso_info)[0]
        target_energy = kso_info[homo_idx - 1]["energy_eV"]  # 1-indexing!

    return np.array(
        [
            np.mean(
                [
                    evaluate_gaussian(
                        target=tmp_target_energy, center=kso_eig, width=gaussian_width
                    )
                    for tmp_target_energy in np.linspace(
                        target_energy, biasing_voltage, energy_grid_points
                    )
                ]
            )
            for kso_eig in kso_info["energy_eV"]
        ]
    )


def evaluate_gaussian(target: float, center: float, width: float) -> float:
    """
    Evaluates a Gaussian function with the specified parameters at the target
    value
    """
    return (1.0 / (width * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((target - center) / width) ** 2
    )


# def plot_dos(kso_info_path: str, gaussian_width: float):
#     """
#     Centers a Gaussian of width `gaussian_width` on each energy eigenvalue read
#     from file "ks_orbital_info.out" at path `kso_info_path` and plots the sum of
#     these as the Density of States (DOS).
#     """
    # import matplotlib.pyplot as plt
    # from rholearn import utils

    # width = 0.3
    # # e_grid = np.linspace(np.min(kso_info["energy_eV"]) - 20, np.max(kso_info["energy_eV"]) + 20, 10000)
    # e_grid = np.linspace(-20, 5, 1000)

    # fig, ax = plt.subplots()

    # dos = np.zeros(len(e_grid))
    # kso_weights = []
    # for e_center in kso_info["energy_eV"]:
    #     gauss = utils.evaluate_gaussian(target=e_grid, center=e_center, width=width)
    #     kso_weights.append(utils.evaluate_gaussian(target=e_target, center=e_center, width=width))
    #     dos += gauss
    #     ax.plot(gauss, e_grid)
    # ax.plot(dos + 5, e_grid, color='k')
    # ax.axline((0, e_target), (10, e_target))
    # ax.set_ylabel("Energy (eV)")
    # # ax.set_xlim([-150, 10])