"""
Module containing helper functions for constructing scalar fields derived from
Kohn-Sham orbitals.
"""
import os
from typing import Union
import numpy as np

from rhocalc.aims import aims_parser


def get_kso_weight_vector_for_named_field(
    field_name: str, kso_info_path: str, **kwargs
) -> np.ndarray:
    """
    For the given scalar field with name `field_name`, return the corresponding
    KS-orbital weight vector that constructs it.

    `field_name` must be one of: ["homo", "lumo", "edensity", "ldos", "ildos"].

    In the case of "ldos" or "ildos", extra kwargs must be passed to control the
    construction of the weight vector:

    - "ldos": `gaussian_width` and `target_energy`. See function
      `get_kso_weight_vector_ldos` for details.
    - "ildos": `gaussian_width`, `target_energy`, `biasing_voltage`,
      `energy_grid_points`. See function `get_kso_weight_vector_ildos` for
      details.
    """
    allowed_names = ["homo", "lumo", "edensity", "ldos", "ildos"]
        
    if field_name == "homo":
        weights = get_kso_weight_vector_homo(kso_info_path)
    elif field_name == "lumo":
        weights = get_kso_weight_vector_lumo(kso_info_path)
    elif field_name == "edensity":
        weights = get_kso_weight_vector_e_density(kso_info_path)
    elif field_name == "ldos":
        weights = get_kso_weight_vector_ldos(kso_info_path, **kwargs)
    elif field_name == "ildos":
        weights = get_kso_weight_vector_ildos(kso_info_path, **kwargs)
    else:
        raise ValueError(
            f"Named field must be one of: {allowed_names}, got: {field_name}"
        )

    return weights


# =======================
# ===== HOMO / LUMO =====
# =======================

def get_kso_weight_vector_homo(kso_info_path: str) -> np.ndarray:
    """
    Returns the KS-orbital weight vector for constructing the HOMO. This
    identifies the highest-occupied KS orbital, then gives all KS orbital of the
    corresponding KS *state* a weight of 1 multiplied by their k-weights. All
    other KS orbitals get a weight of zero. 
    
    Therefore if KSOs exist for multiple k-points, the weight vector will be
    non-zero for all k-points corresponding to the given highest occupied KSO.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    # Find the indices of the HOMO states
    homo_kso_idxs = find_homo_kso_idxs(kso_info)

    # Fill in weights
    weights = np.zeros(ks_info.shape[0])
    for kso_idx in homo_kso_idxs:  # these are 1-indexed
        weights[kso_idx - 1] = kso_info[kso_idx - 1]["k_weight"]

    return weights

def get_kso_weight_vector_lumo(kso_info_path: str) -> np.ndarray:
    """
    Returns the KS-orbital weight vector for constructing the LUMO. This
    identifies the lowest-unoccupied KS orbital, then gives all KS orbital of
    the corresponding KS *state* a weight of 1 multiplied by their k-weights.
    All other KS orbitals get a weight of zero. 
    
    Therefore if KSOs exist for multiple k-points, the weight vector will be
    non-zero for all k-points corresponding to the given lowest unoccupied KSO.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    # Find the indices of the HOMO states
    lumo_kso_idxs = find_lumo_kso_idxs(kso_info)

    # Fill in weights
    weights = np.zeros(ks_info.shape[0])
    for kso_idx in lumo_kso_idxs:  # these are 1-indexed
        weights[kso_idx - 1] = kso_info[kso_idx - 1]["k_weight"]

    return weights


def find_homo_kso_idxs(ks_orbital_info: Union[str, np.ndarray]) -> np.ndarray:
    """
    Returns the KSO indices that correspond to the HOMO states. These are all
    the orbitals that have the same KS *state* index as the highest occupied KS
    orbital. 
    
    Note that the returned indices are 1-indexed for consistency with the
    labelling in FHI-aims.

    For instance, if the KS orbital with (state, spin, kpt) indices as (3, 1,
    4), the indices of all KS orbitals with KS state == 3 are returned.
    """
    if isinstance(ks_orbital_info, str):
        ks_orbital_info = get_ks_orbital_info(ks_orbital_info, as_array=True)
    ks_orbital_info = np.sort(ks_orbital_info, order="energy_eV")

    # Find the HOMO orbital
    homo_kso_idx = np.where(ks_orbital_info["occ"] > 0)[0][-1]
    homo_state_idx = ks_orbital_info[homo_kso_idx]["state_i"]

    # Find all states that correspond to the KS state
    homo_kso_idxs = np.where(ks_orbital_info["state_i"] == homo_state_idx)[0]

    return [ks_orbital_info[i]["kso_i"] for i in homo_kso_idxs]


def find_lumo_kso_idxs(ks_orbital_info: Union[str, np.ndarray]) -> np.ndarray:
    """
    Returns the KSO indices that correspond to the LUMO states. These are all
    the orbitals that have the same KS *state* index as the lowest unoccupied KS
    orbital. 
    
    Note that the returned indices are 1-indexed for consistency with the
    labelling in FHI-aims.

    For instance, if the KS orbital with (state, spin, kpt) indices as (3, 1,
    4), the indices of all KS orbitals with KS state == 3 are returned.
    """
    if isinstance(ks_orbital_info, str):
        ks_orbital_info = get_ks_orbital_info(ks_orbital_info, as_array=True)
    ks_orbital_info = np.sort(ks_orbital_info, order="energy_eV")

    # Find the HOMO orbital
    lumo_kso_idx = np.where(ks_orbital_info["occ"] == 0)[0][0]
    lumo_state_idx = ks_orbital_info[lumo_kso_idx]["state_i"]

    # Find all states that correspond to the KS state
    lumo_kso_idxs = np.where(ks_orbital_info["state_i"] == lumo_state_idx)[0]

    return [ks_orbital_info[i]["kso_i"] for i in lumo_kso_idxs]


# =========================================
# ===== Electron density, LDOS, ILDOS =====
# =========================================

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
    Returns the KS-orbital weight vector for constructing the Local Density of States
    (LDOS).

    Each KSO weight is given by the evaluation of a Gaussian function, of width
    `gaussian_width`, centered on the KS-orbital eigenvalue and evaluated at the
    `target_energy`.

    Where `target_energy` is not passed, the eigenvalue of the highest occupied
    KS-orbital is used.

    `gaussian_width`, and `target_energy`  must be passed in units of eV.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    if target_energy is None:  # Find the energy of the HOMO
        homo_idx = find_homo_kso_idxs(kso_info)[0]
        target_energy = kso_info[homo_idx - 1]["energy_eV"]  # 1-indexing!

    return np.array(
        [
            kso["k_weight"]
            * evaluate_gaussian(
                target=target_energy, center=kso["energy_eV"], width=gaussian_width
            )
            for kso in kso_info
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
    Returns the KS-orbital weight vector for constructing the Local Density of States
    (LDOS).

    Each KSO weight is given by the sum of Gaussian functions, each of width
    `gaussian_width`, centered on the KS-orbital eigenvalue and evaluated on a discrete
    energy grid from the `target_energy` to `target_energy` + `biasing_voltage`.

    Where `target_energy` is not passed, the eigenvalue of the highest occupied
    KS-orbital is used.

    `biasing_voltage`, `gaussian_width`, and `target_energy`  must be passed in units of
    eV.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    if target_energy is None:  # Find the energy of the HOMO
        homo_idx = find_homo_kso_idxs(kso_info)[0]
        target_energy = kso_info[homo_idx - 1]["energy_eV"]  # 1-indexing!

    return np.array(
        [
            np.mean(
                [
                    kso["k_weight"]
                    * evaluate_gaussian(
                        target=tmp_target_energy,
                        center=kso["energy_eV"],
                        width=gaussian_width,
                    )
                    for tmp_target_energy in np.linspace(
                        target_energy, biasing_voltage, energy_grid_points
                    )
                ]
            )
            for kso in kso_info
        ]
    )


def evaluate_gaussian(target: float, center: float, width: float) -> float:
    """
    Evaluates a Gaussian function with the specified parameters at the target value
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
