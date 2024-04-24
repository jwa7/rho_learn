"""
Module containing helper functions for constructing scalar fields derived from
Kohn-Sham orbitals.
"""

import os
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
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


def calculate_dos(
    kso_info_path: str,
    gaussian_width: float,
    e_grid: np.ndarray = None,
    orbital_occupancy: float = 2.0,
):
    """
    Centers a Gaussian of width `gaussian_width` on each energy eigenvalue read from
    file "kso_info.out" at path `kso_info_path` and plots the sum of these as the
    Density of States (DOS).

    The Gaussian centered on each eigenvalue is multiplied by its k-weight, and the
    final DOS is multiplied by the `orbital_occupancy` to account for spin-(un)paired
    orbitals.
    """
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)
    if e_grid is None:
        e_grid = np.linspace(
            np.min(kso_info["energy_eV"]) - 20,
            np.max(kso_info["energy_eV"]) + 20,
            10000,
        )

    dos = np.zeros(len(e_grid))
    for kso in kso_info:
        dos += (
            evaluate_gaussian(
                target=e_grid, center=kso["energy_eV"], width=gaussian_width
            )
            * kso["k_weight"]  # ensure k-weighted
        )

    dos *= orbital_occupancy  # acccount for orbital occupancy

    return e_grid, dos


def calculate_fermi_energy(
    kso_info_path: str,
    n_electrons: int,
    e_grid: Optional[np.ndarray] = None,
    orbital_occupancy: float = 2.0,
    gaussian_width: float = 0.3,
    interpolation_truncation: Optional[float] = 0.1,
) -> float:
    """
    Calculates the Fermi energy by integrating the cumulative density of states.

    If `e_grid` is specified, the DOS will be calculated on this grid. Otherwise, the
    `e_grid` is taken across the range of energies in the `kso_info_path` file.

    The number of electrons `n_electrons` should correspond to the number of expected
    electrons to be integrated over in the DOS for the given energy range.

    If the `e_grid` range is specified as a subset, i.e. excluding core states,
    `n_electrons` should beadjusted accordingly.

    The `orbital_occupancy` is the number of electrons that can occupy each orbital. By
    default, this is set to 2.0 for spin-paired electrons.
    """
    # Calc and plot the DOS
    kso_info = aims_parser.get_ks_orbital_info(kso_info_path)

    # Calculate the DOS, k-weighted and accounting for orbital occupancy
    e_grid, dos = calculate_dos(
        kso_info_path,
        gaussian_width=gaussian_width,
        e_grid=e_grid,
        orbital_occupancy=orbital_occupancy,
    )

    # Compute the cumulative DOS and interpolate
    cumulative_dos = cumulative_trapezoid(dos, e_grid, axis=0) - n_electrons
    interpolated = interp1d(
        e_grid[:-1], cumulative_dos, kind="cubic", copy=True, assume_sorted=True
    )
    fermi_energy = brentq(
        interpolated,
        e_grid[0] + interpolation_truncation,
        e_grid[-1] - interpolation_truncation,
    )

    return fermi_energy


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


def get_percent_mae_between_fields(
    input: np.ndarray, target: np.ndarray, grid: np.ndarray
) -> float:
    """
    Calculates the absolute error between the target and input scalar fields,
    integrates this over all space, then divides by the target field integrated
    over all space (i.e. the number of electrons). Multiplies by 100 and returns
    this as a % MAE.
    """
    if not np.all(input[:, :3] == grid[:, :3]):
        raise ValueError(
            "grid points not equivalent between input scalar field and integration weights"
        )
    if not np.all(target[:, :3] == grid[:, :3]):
        raise ValueError(
            "grid points not equivalent between target scalar field and integration weights"
        )

    
    return (
        100
        * np.dot(np.abs(input[:, 3] - target[:, 3]), grid[:, 3])
        / np.dot(target[:, 3], grid[:, 3])
    )


def calculate_electrostatic_potential(
    rho: np.ndarray,
    grid: np.ndarray,
    eval_coords: Optional[np.ndarray] = None,
    eval_coords_size: Optional[np.ndarray] = None, 
) -> Tuple[np.ndarray]:
    """
    Calculate the electrostatic potential for a given charge density `rho` and grid
    points on which it is evaluated `grid` (including integration weights).
    
    A vector of points `eval_coords` can be provided as the points at which the
    potential is evaluated.
    
    If `eval_coords` is None, the grid is constructed by uniformly discretizing over the
    min and max coordinates in `grid` along each axis, according to the
    `eval_coord_size` parameter.

    Returned is a tuple of the Z evaluation coordinates, and the electrostatic potential
    V_z at the Z coordinate, averaged over the evaluation points in that Z plane.
    """
    # Check grid points match
    assert np.all(rho[:, :3] == grid[:, :3])

    # Weight the charge density by the grid weights
    rho = rho[:, 3] * grid[:, 3]
    coords = grid[:, :3]

    if eval_coords is None:
        if eval_coords_size is None:
            raise ValueError("If `eval_coords` is None, `eval_coords_size` must be provided")
        min_x, max_x = grid[:, 0].min(), grid[:, 0].max()
        min_y, max_y = grid[:, 1].min(), grid[:, 1].max()
        min_z, max_z = grid[:, 2].min(), grid[:, 2].max()
        X, Y, Z = [
            np.linspace(min_, max_, size) for min_, max_, size in zip(
                [min_x, min_y, min_z], [max_x, max_y, max_z], eval_coords_size
            )
        ]
    else:
        if eval_coords.shape[1] != 3:
            raise ValueError("eval_coords must have shape (N_pts, 3)")
        Z = eval_coords[:, 2]

    V = []
    for z in Z:
        # Get evaluation coordinates in the current z plane
        eval_coords = np.array([np.array([x, y, z]) for x in X for y in Y])

        # Calculate |r - r'| for all r in the `eval_coords` and all r' in `coords`
        norm_length = np.linalg.norm(
            np.abs(
                coords.reshape(coords.shape[0], 1, coords.shape[1]).repeat(eval_coords.shape[0], axis=1) 
                - eval_coords
            ),
            axis=2
        )

        # Calculate the integrand for all r in `coords` and all r' in `eval_coords`
        integrand = (rho.reshape(-1, 1) / norm_length)
        
        # Calculate the integral, y summing over `coords` 
        integral = integrand.sum(axis=0)

        # Mean over points in the Z plane
        V_z = integral.mean()
        V.append(V_z)

    return Z, np.array(V)