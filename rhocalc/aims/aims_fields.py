"""
Module containing helper functions for constructing scalar fields derived from
Kohn-Sham orbitals.
"""

import os
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

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
    Returns the KS-orbital weight vector for constructing the HOMO. This identifies the
    highest-occupied KS orbital, and assigns it its k-weight. All other KS orbitals get
    a weight of zero.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    # Find the indices of the HOMO states
    homo_kso_idx = get_homo_kso_idx(kso_info, max_occ=2)

    # Fill in weights
    weights = np.zeros(ks_info.shape[0])
    homo_kso =  kso_info[homo_kso_idx - 1]
    assert homo_kso["kso_i"] == homo_kso_idx
    weights[homo_kso_idx - 1] = homo_kso["k_weight"]

    return weights


def get_kso_weight_vector_lumo(kso_info_path: str) -> np.ndarray:
    """
    Returns the KS-orbital weight vector for constructing the LUMO. This identifies the
    lowest-unoccupied KS orbital, and assigns it its k-weight. All other KS orbitals get
    a weight of zero.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    # Find the indices of the LUMO states
    lumo_kso_idx = get_lumo_kso_idx(kso_info, max_occ=2)

    # Fill in weights
    weights = np.zeros(ks_info.shape[0])
    lumo_kso =  kso_info[lumo_kso_idx - 1]
    assert lumo_kso["kso_i"] == lumo_kso_idx
    weights[lumo_kso_idx - 1] = lumo_kso["k_weight"]

    return weights


def get_homo_kso_idx(kso_info: Union[str, np.ndarray], max_occ: int = 2) -> np.ndarray:
    """
    Returns the KSO index that corresponds to the HOMO state. This is defined as the
    highest energy state with an occupation greater than `max_occ` / 2.

    Note that the returned index correpsonds to the FHI-aims KSO index, which is
    1-indexed.
    """
    if isinstance(kso_info, str):
        kso_info = get_ks_orbital_info(kso_info, as_array=True)
    kso_info = np.sort(kso_info, order="energy_eV")

    occ_states = kso_info[np.where(kso_info["occ"] > (max_occ / 2))[0]]
    return np.sort(occ_states, order="energy_eV")[-1]["kso_i"]


def get_lumo_kso_idx(kso_info: Union[str, np.ndarray], max_occ: int = 2) -> np.ndarray:
    """
    Returns the KSO index that corresponds to the LUMO state. This is defined as the
    lowest energy state with an occupation less than `max_occ` / 2.

    Note that the returned index correpsonds to the FHI-aims KSO index, which is
    1-indexed.
    """
    if isinstance(kso_info, str):
        kso_info = get_ks_orbital_info(kso_info, as_array=True)
    kso_info = np.sort(kso_info, order="energy_eV")

    unocc_states = kso_info[np.where(kso_info["occ"] < (max_occ / 2))[0]]
    return np.sort(unocc_states, order="energy_eV")[0]["kso_i"]


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
    kso_info_path: str, gaussian_width: float, target_energy: float
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
    W_vect = []
    for kso in kso_info:
        W_a = kso["k_weight"] * evaluate_gaussian(
            target=target_energy, center=kso["energy_eV"], width=gaussian_width
        )
        W_vect.append(W_a)

    return np.array(W_vect)


def get_kso_weight_vector_ildos(
    kso_info_path: str,
    gaussian_width: float,
    biasing_voltage: float,
    target_energy: float,
    energy_grid_points: int = 1000,
    method: str = "gaussian_analytical",
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

    `method` is the method of computing the integral, either assuming a Gaussian
    integrated analytically (``method="gaussian_analytical"``) or numerically
    (``method="gaussian_numerical"``), or a delta function (``method="delta"``) centered
    on each eigenvalue.
    """
    if method == "gaussian_analytical":
        return _get_kso_weight_vector_ildos_analytical(
            kso_info_path=kso_info_path,
            gaussian_width=gaussian_width,
            biasing_voltage=biasing_voltage,
            target_energy=target_energy,
        )
    elif method == "gaussian_numerical":
        return _get_kso_weight_vector_ildos_numerical(
            kso_info_path=kso_info_path,
            gaussian_width=gaussian_width,
            biasing_voltage=biasing_voltage,
            target_energy=target_energy,
        )
    elif method == "delta_analytical":
        return _get_kso_weight_vector_ildos_delta(
            kso_info_path=kso_info_path,
            biasing_voltage=biasing_voltage,
            target_energy=target_energy,
        )
    else:
        raise ValueError(
            "Invalid option for `method`. must be one of ['gaussian_analytical',"
            " 'gaussian_numerical', 'delta_analytical']"
        )


def _get_kso_weight_vector_ildos_analytical(
    kso_info_path: str,
    gaussian_width: float,
    biasing_voltage: float,
    target_energy: float,
) -> np.ndarray:
    """
    For each KS-orbital, the weight is given by the numerical integral of a Gaussian
    centered on the energy eigenvalue, evaluated at the `target_energy`:

    W(a) = 0.5 * k_weight(a) * (
        erf( (\epsilon_a - \epsilon) / (\sigma * \sqrt(2)) )
        - erf( (\epsilon_a - \epsilon - V) / (\sigma * \sqrt(2))
    )

    where V is the biasing voltage, n is the number of energy grid points, g is the
    Gaussian function of width \sigma, k_weight(a) is the weight of the k-point that
    accounts for symmetry in the Brillouin zone.
    """
    if not np.abs(biasing_voltage) > 0:
        raise ValueError("Biasing voltage must be non-zero")

    # Define the integration limits. If biasing voltage is negative, limits should be
    # switched
    if biasing_voltage >= 0:
        lim_lo, lim_hi = target_energy, target_energy + biasing_voltage
    else:
        lim_lo, lim_hi = target_energy + biasing_voltage, target_energy

    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    W_vect = []
    for kso in kso_info:
        W_a = 0.5 * (
            erf((kso["energy_eV"] - lim_lo) / (np.sqrt(2) * gaussian_width))
            - erf((kso["energy_eV"] - lim_hi) / (np.sqrt(2) * gaussian_width))
        )
        W_vect.append(W_a * kso["k_weight"])

    return np.array(W_vect)


def _get_kso_weight_vector_ildos_numerical(
    kso_info_path: str,
    gaussian_width: float,
    biasing_voltage: float,
    target_energy: float,
    energy_grid_points: int = 100,
) -> np.ndarray:
    """
    For each KS-orbital, the weight is given by the numerical integral of a Gaussian
    centered on the energy eigenvalue \epsilon_a, evaluated at the `target_energy`:

    W(a) = (V / n) * k_weight(a)
        * \sum_{\epsilon'=\epsilon}^{\epsilon + V} g(\epsilon' - \epsilon_a, \sigma)

    where \epsilon is the target energy, V is the biasing voltage, n is the number of
    energy grid points, g is the Gaussian function of width \sigma, k_weight(a) is the
    weight of the k-point that accounts for symmetry in the Brillouin zone.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    if target_energy is None:  # Find the energy of the HOMO
        homo_idx = find_homo_kso_idxs(kso_info)[0]
        target_energy = kso_info[homo_idx - 1]["energy_eV"]  # 1-indexing!

    W_vect = []
    for kso in kso_info:
        W_a = (biasing_voltage / energy_grid_points) * np.sum(
            [
                evaluate_gaussian(
                    target=tmp_target_energy,
                    center=kso["energy_eV"],
                    width=gaussian_width,
                )
                for tmp_target_energy in np.linspace(
                    target_energy, target_energy + biasing_voltage, energy_grid_points
                )
            ]
        )
        W_vect.append(W_a * kso["k_weight"])

    return np.array(W_vect)


def _get_kso_weight_vector_ildos_delta(
    kso_info_path: str,
    biasing_voltage: float,
    target_energy: float,
) -> np.ndarray:
    """
    For each KS-orbital, the weight is given by the numerical integral of a Gaussian
    centered on the energy eigenvalue, evaluated at the `target_energy`:

    W(a) = 0.5 * (
        erf( (\epsilon_a - \epsilon) / (\sigma * \sqrt(2)) )
        - erf( (\epsilon_a - \epsilon - V) / (\sigma * \sqrt(2))
    )  * k_weight(a)

    where V is the biasing voltage, n is the number of energy grid points, and g is the
    Gaussian function of width \sigma.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)
    W_vect = []
    for kso in kso_info:
        if target_energy <= kso["energy_eV"] <= target_energy + biasing_voltage:
            W_vect.append(kso["k_weight"])
        else:
            W_vect.append(0)

    return np.array(W_vect)


def evaluate_gaussian(target: float, center: float, width: float) -> float:
    """
    Evaluates a Gaussian function with the specified parameters at the target value
    """
    return (1.0 / (width * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((target - center) / width) ** 2
    )


def calc_dos(kso_info_path: str, gaussian_width: float, e_grid: np.ndarray = None):
    """
    Centers a Gaussian of width `gaussian_width` on each energy eigenvalue read from
    file "kso_info.out" at path `kso_info_path` and plots the sum of these as the
    Density of States (DOS).
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)
    if e_grid is None:
        e_grid = np.linspace(
            np.min(kso_info["energy_eV"]) - 20,
            np.max(kso_info["energy_eV"]) + 20,
            10000,
        )

    dos = np.zeros(len(e_grid))
    for e_center in kso_info["energy_eV"]:
        dos += evaluate_gaussian(target=e_grid, center=e_center, width=gaussian_width)

    return e_grid, dos


def sort_field_by_grid_points(field: Union[str, np.ndarray]) -> np.ndarray:
    """
    Loads the scalar field from file and returns a 2D array sorted by the norm of the
    grid points.

    Assumes the file at `field_path` has four columns, corresponding to the x, y, z,
    coordinates of the grid points and the scalar field value at that grid point.
    """
    if isinstance(field, str):  # load from file
        field = np.loadtxt(
            field,
            dtype=[
                ("x", np.float64),
                ("y", np.float64),
                ("z", np.float64),
                ("w", np.float64),
            ],
        )
    else:  # create a structured array
        field = field.ravel().view(
            dtype=[
                ("x", np.float64),
                ("y", np.float64),
                ("z", np.float64),
                ("w", np.float64),
            ]
        )
    field = np.sort(field, order=["x", "y", "z"])

    return np.array([[x, y, z, w] for x, y, z, w in field], dtype=np.float64)
