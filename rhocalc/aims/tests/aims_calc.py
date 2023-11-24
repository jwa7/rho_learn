#!/usr/bin/bash
import inspect
import glob
import os
import shutil

import ase.io
import numpy as np
from rhocalc.aims import aims_calc

TOLERANCE = {"rtol": 1e-5, "atol": 1e-10}


def run_scf(aims_kwargs: dict, sbatch_kwargs: dict, calcs: dict):
    """
    Runs an SCF calculation for each of the systems in `calcs`, in separate
    directories.
    """

    top_dir = os.getcwd()
    os.chdir(top_dir)

    for calc_i, calc in calcs.items():

        # Define run dir and AIMS path
        run_dir = f"{calc_i}"

        # Settings for control.in
        aims_kwargs_calc = aims_kwargs.copy()
        aims_kwargs_calc.update(calc["aims_kwargs"])

        # Settings for sbatch run script
        sbatch_kwargs_calc = sbatch_kwargs.copy()
        sbatch_kwargs_calc.update(calc["sbatch_kwargs"])

        # Write AIMS input files
        aims_calc.write_input_files(
            atoms=calc["atoms"], 
            run_dir=run_dir, 
            aims_kwargs=aims_kwargs_calc,
        )

        # Write sbatch run script
        aims_calc.write_aims_sbatch(
            fname=os.path.join(run_dir, "run-aims.sh"), 
            aims=calc["aims_path"], 
            load_modules=["intel", "intel-oneapi-mkl", "intel-oneapi-mpi"],
            **sbatch_kwargs_calc
        )

        # Run aims
        aims_calc.run_aims_in_dir(run_dir)

    return



def run_ri(aims_kwargs: dict, sbatch_kwargs: dict, calcs: dict, use_restart: bool = False):
    """
    Runs an RI calculation for the total electron density and each KS-orbital
    with full output, for each of the systems in `calcs`, in separate
    directories.

    In each of the run directories, uses the priously converged SCF (using func
    `run_scf`) as the restart density matrix. Runs the RI calculation with no
    SCF iterations from this denisty matrix in a subfolder "ri/" for each
    system.
    """

    top_dir = os.getcwd()
    os.chdir(top_dir)

    for calc_i, calc in calcs.items():

        # Define run dir and AIMS path
        run_dir = f"{calc_i}/ri/"
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # Settings for control.in
        aims_kwargs_calc = aims_kwargs.copy()
        aims_kwargs_calc.update(calc["aims_kwargs"])

        # Force read density restart, no SCF, and to output sanity check data
        aims_kwargs_calc.update(
            {
                # "output ": ["cube total_density"],
                "output": ["cube ri_fit"],
                # "ri_fit_total_density": True,
                "ri_fit_sanity_check": True,
            }
        )

        if use_restart:
            # Copy restart density matrix, and set elsi_restart to read mode
            aims_kwargs_calc.update({
                "elsi_restart": "read",
                "sc_iter_limit": 0,
                "postprocess_anyway": True,
                "ri_fit_assume_converged": True,
                }
            )
            for density_matrix in glob.glob(os.path.join(f"{calc_i}/", "D*.csc")):
                shutil.copy(density_matrix, run_dir)


        # Settings for sbatch run script
        sbatch_kwargs_calc = sbatch_kwargs.copy()
        sbatch_kwargs_calc.update(calc["sbatch_kwargs"])

        # Write AIMS input files
        aims_calc.write_input_files(
            atoms=calc["atoms"], 
            run_dir=run_dir, 
            aims_kwargs=aims_kwargs_calc,
        )

        # Write sbatch run script
        aims_calc.write_aims_sbatch(
            fname=os.path.join(run_dir, "run-aims.sh"), 
            aims=calc["aims_path"], 
            load_modules=["intel", "intel-oneapi-mkl", "intel-oneapi-mpi"],
            **sbatch_kwargs_calc
        )

        # Run aims
        aims_calc.run_aims_in_dir(run_dir)

    return

# ======================================== 
# ========= Test functions ===============
# ========================================


def total_densities_integrate_to_n_electrons(calcs):
    """
    First finds the formal number of electrons in the system as the sum of
    atomic numbers of the nuclei. Calculates the number of electrons from the
    sum of state occupations printed in ks_orbital_info.out. Integrates the
    total electron densities on the AIMS grid to find the number of electrons
    present in the densities imported from physics.f90 and that constructed from
    the density matrix. Checks that all of these are equal.
    """
    print(f"Test: {inspect.stack()[0][3]}")
    print(f"    N_e: [formal, from ks_orbital_info, from physics, from densmat, from RI]")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        # Load files
        ri_dir = f"{calc_i}/ri/"
        ks_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))
        rho_from_physics = np.loadtxt(os.path.join(ri_dir, f"rho_physics.out"))
        rho_from_densmat = np.loadtxt(os.path.join(ri_dir, f"rho_ref.out"))
        # rho_from_ri = np.loadtxt(os.path.join(ri_dir, f"rho_ri.out"))
        grid = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))

        # Calc number of electrons
        N_formal = _get_formal_number_of_electrons(os.path.join(ri_dir, "geometry.in"), charge=0)
        N_info = _get_occ_number_of_electrons(ks_info)
        N_from_physics = _get_integrated_number_of_electrons(rho_from_physics, grid)
        N_from_densmat = _get_integrated_number_of_electrons(rho_from_densmat, grid)
        # N_from_RI = _get_integrated_number_of_electrons(rho_from_ri, grid)

        # Check for equivalence
        N_list = [N_formal, N_info, N_from_physics, N_from_densmat]#, N_from_RI]
        decimals = 10
        if np.all(  # relax slightly the tolerance on num electrons by multiplying by 10
            [
                np.abs(N - N_formal) / N_formal < TOLERANCE["rtol"] * 10 for N in N_list
            ]
        ):
            print(f"    PASS - Calc {calc_i} - {calc['name']}. N_e: {[np.round(N, decimals) for N in N_list]}")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}.  N_e: {[np.round(N, decimals) for N in N_list]}")
            all_passed = False
            failed_calcs[calc_i] = N_list

    print("\n")
    return failed_calcs


def coeff_matrices_sum_to_density_matrix(calcs: dict, return_all_calcs: bool = False):
    """
    Tests that the coefficient matrices for each KSO, wieghted by their
    electronic occupation, sum to the density matrix constructed in AIMS by the
    same procedure. 
    """
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"
        ks_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))

        # Load the density matrix constructed in AIMS
        densmat = np.loadtxt(os.path.join(ri_dir, "nao_coeff_matrix.out"))
        dim = int(np.sqrt(densmat.shape[0]))
        densmat = densmat.reshape((dim, dim))

        # Perform weighted sum over coeff matrices
        densmat_from_coeffmats = []
        for row in ks_info:
            kso_i, state_i, spin_i, kpt_i, k_weight, occ, eig, kso_weight = row
            if kso_weight < 1e-15:
                continue
            kso_i = int(kso_i)
            suffix = f"_{kso_i:04d}"
            C = np.loadtxt(os.path.join(ri_dir, f"nao_coeff_matrix{suffix}.out")).reshape((dim, dim))
            assert np.all(C == C.T)
            assert k_weight * occ == kso_weight
            densmat_from_coeffmats.append(kso_weight * C)

        densmat_from_coeffmats = np.sum(densmat_from_coeffmats, axis=0)
        assert densmat_from_coeffmats.shape == (dim, dim)

        # Check pass/fail
        mae = np.abs(densmat_from_coeffmats - densmat).mean()
        if np.allclose(mae, 0, **TOLERANCE):
            print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae}")
            if return_all_calcs:
                failed_calcs[calc_i] = (densmat_from_coeffmats, densmat)
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae}")
            all_passed = False
            failed_calcs[calc_i] = (densmat_from_coeffmats, densmat)
    print("\n")
    return failed_calcs
    

def density_from_densmat_equals_density_from_physics(calcs: dict):
    """
    Tests that the coefficient matrices for each KSO, wieghted by their
    electronic occupation, sum to the density matrix constructed in AIMS by the
    same procedure. 
    """
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():

        # Load files
        ri_dir = f"{calc_i}/ri/"
        rho_from_densmat = np.loadtxt(os.path.join(ri_dir, f"rho_ref.out"))
        rho_from_physics = np.loadtxt(os.path.join(ri_dir, f"rho_physics.out"))
        grid = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))
        
        # Check pass/fail
        mae = _get_percent_mae_between_fields(input=rho_from_densmat, target=rho_from_physics, grid=grid)
        if mae < TOLERANCE["rtol"] * 100:
            print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
            all_passed = False
            failed_calcs[calc_i] = (rho_from_physics, rho_from_densmat)
    print("\n")
    return failed_calcs


def density_from_ri_equals_density_from_densmat(calcs: dict):
    """
    Tests that the coefficient matrices for each KSO, wieghted by their
    electronic occupation, sum to the density matrix constructed in AIMS by the
    same procedure. 
    """
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():

        # Load files
        ri_dir = f"{calc_i}/ri/"
        rho_from_ri = np.loadtxt(os.path.join(ri_dir, f"rho_ri.out"))
        rho_from_densmat = np.loadtxt(os.path.join(ri_dir, f"rho_ref.out"))
        grid = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))
        
        # Check pass/fail
        mae = _get_percent_mae_between_fields(input=rho_from_ri, target=rho_from_densmat, grid=grid)
        if mae < TOLERANCE["rtol"] * 100:
            print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
            all_passed = False
            failed_calcs[calc_i] = (rho_from_ri, rho_from_densmat)
    print("\n")
    return failed_calcs


def ksos_from_coeffmats_sum_to_density_from_densmat(calcs: dict):
    """
    Tests that the real-space KS-orbitals formed from NAO coefficient matrices
    sum to the total density built from the NAO density matrix by a similar
    procedure in AIMS.
    """
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        # Load files
        ri_dir = f"{calc_i}/ri/"
        kso_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))
        rho_from_densmat = np.loadtxt(os.path.join(ri_dir, "rho_ref.out"))
        grid = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))

        # Build the density from a sum of KSOs
        rho_from_ksos = _sum_kso_fields(
            output_dir=ri_dir, 
            kso_file_prefix="rho_ref", 
            kso_info=kso_info,
            grid=grid,
        )

        # Check for equivalence with the density matrix
        mae = _get_percent_mae_between_fields(input=rho_from_ksos, target=rho_from_densmat, grid=grid)
        if mae < TOLERANCE["rtol"] * 100:
            print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
            all_passed = False
            failed_calcs[calc_i] = (rho_from_ksos, rho_from_densmat)
    print("\n")
    return failed_calcs


def ksos_from_coeffmats_sum_to_density_from_physics(calcs: dict):
    """
    Tests that the real-space KS-orbitals formed from NAO coefficient matrices
    sum to the total density imported from physics.f90 in AIMS.
    """
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        # Load files
        ri_dir = f"{calc_i}/ri/"
        kso_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))
        rho_from_densmat = np.loadtxt(os.path.join(ri_dir, "rho_physics.out"))
        grid = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))

        # Build the density from a sum of KSOs
        rho_from_ksos = _sum_kso_fields(
            output_dir=ri_dir, 
            kso_file_prefix="rho_ref",
            kso_info=kso_info,
            grid=grid,
        )

        # Check for equivalence with the density matrix
        mae = _get_percent_mae_between_fields(input=rho_from_ksos, target=rho_from_densmat, grid=grid)
        if mae < TOLERANCE["rtol"] * 100:
            print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
            all_passed = False
            failed_calcs[calc_i] = (rho_from_ksos, rho_from_densmat)
    print("\n")
    return failed_calcs


def ksos_from_ri_sum_to_density_from_densmat(calcs: dict):
    """
    Tests that the real-space KS-orbitals formed from RI coefficients sum to the
    total density formed from the NAO density matrix.
    """
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        # Load files
        ri_dir = f"{calc_i}/ri/"
        kso_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))
        rho_from_densmat = np.loadtxt(os.path.join(ri_dir, "rho_ref.out"))
        grid = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))

        # Build the density from a sum of KSOs
        rho_from_ksos = _sum_kso_fields(
            output_dir=ri_dir, 
            kso_file_prefix="rho_ri", 
            kso_info=kso_info,
            grid=grid,
        )

        # Check for equivalence with the density matrix
        mae = _get_percent_mae_between_fields(input=rho_from_ksos, target=rho_from_densmat, grid=grid)
        if mae < TOLERANCE["rtol"] * 100:
            print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
            all_passed = False
            failed_calcs[calc_i] = (rho_from_ksos, rho_from_densmat)
    print("\n")
    return failed_calcs


def ksos_from_ri_sum_to_density_from_physics(calcs: dict):
    """
    Tests that the real-space KS-orbitals formed from RI coefficients sum to the
    total density imported from physics.f90 in AIMS.
    """
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        # Load files
        ri_dir = f"{calc_i}/ri/"
        kso_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))
        rho_from_densmat = np.loadtxt(os.path.join(ri_dir, "rho_physics.out"))
        grid = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))

        # Build the density from a sum of KSOs
        rho_from_ksos = _sum_kso_fields(
            output_dir=ri_dir, 
            kso_file_prefix="rho_ri",
            kso_info=kso_info,
            grid=grid,
        )

        # Check for equivalence with the density matrix
        mae = _get_percent_mae_between_fields(input=rho_from_ksos, target=rho_from_densmat, grid=grid)
        if mae < TOLERANCE["rtol"] * 100:
            print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
            all_passed = False
            failed_calcs[calc_i] = (rho_from_ksos, rho_from_densmat)
    print("\n")
    return failed_calcs


def ri_coeffs_for_ksos_sum_to_total_density_from_densmat(calcs: dict):
    """
    Tests that the weighted sum of RI coefficients for each KS-orbital is
    equivaltent to the RI coefficients for the total density built from the
    NAO density matrix.
    """
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"
        ks_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))

        # Load the density matrix constructed in AIMS
        ri_coeffs = np.loadtxt(os.path.join(ri_dir, "ri_coeffs.out"))

        # Perform weighted sum over coeff matrices
        ri_coeffs_from_ksos = []
        for row in ks_info:
            kso_i, state_i, spin_i, kpt_i, k_weight, occ, eig, kso_weight = row
            if kso_weight < 1e-15:
                continue
            kso_i = int(kso_i)
            suffix = f"_{kso_i:04d}"
            ri_coeffs_kso = np.loadtxt(os.path.join(ri_dir, f"ri_coeffs{suffix}.out"))
            assert k_weight * occ == kso_weight
            ri_coeffs_from_ksos.append(kso_weight * ri_coeffs_kso)

        ri_coeffs_from_ksos = np.sum(ri_coeffs_from_ksos, axis=0)

        # Check pass/fail
        mae = np.abs(ri_coeffs_from_ksos - ri_coeffs).mean()
        if np.allclose(mae, 0, **TOLERANCE):
            print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae}")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae}")
            all_passed = False
            failed_calcs[calc_i] = (ri_coeffs_from_ksos, ri_coeffs)
    print("\n")
    return failed_calcs

def overlap_is_symmetric(calcs: dict):
    """Tests the the RI basis overlap matrix is symmetric"""
    raise NotImplementedError


# ==========================================
# =========== Helper functions =============
# ==========================================

def _get_formal_number_of_electrons(geometry_file: str, charge: float = 0.0) -> float:
    """
    Gets the formal number of electrons by reading the geometry.in file into an
    ASE Atoms object, summing nuclear charges, and subtracting the charge.
    """
    return ase.io.read(geometry_file).get_atomic_numbers().sum() - charge


def _get_occ_number_of_electrons(ks_info: np.ndarray) -> float:
    """
    Calculates the number of electrons by summing the KS-orbital occupations
    weighted by their k-weights, as found in "ks_orbital_info.out".
    """
    k_weights, occs = ks_info[:, 4], ks_info[:, 5]
    return np.sum(k_weights * occs)


def _get_integrated_number_of_electrons(density: np.ndarray, grid: np.ndarray) -> float:
    """
    Integrates the total electron density field over real space using the grid
    point integration weights to obtain the number of electrons.
    """
    if not np.all(density[:, :3] == grid[:, :3]):
        raise ValueError(
            "grid points not equivalent between scalar field and integration weights"
        )
    return np.dot(density[:, 3], grid[:, 3])


def _get_percent_mae_between_fields(input: np.ndarray, target: np.ndarray, grid: np.ndarray) -> float:
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
    return 100 * np.dot(np.abs(input[:, 3] - target[:, 3]), grid[:, 3]) / np.dot(target[:, 3], grid[:, 3])


def _sum_kso_fields(
    output_dir: str, kso_file_prefix: str, kso_info: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """
    Using the occupations and k-weights in `ks_info_file`, performs a weighted
    sum of the KS-orbital probability densities found in `output_dir`, with the
    given file prefix, i.e. "rho_ref" or "rho_ri".
    """
    # Perform weighted sum over KSOs
    weighted_ksos = []
    for row in kso_info:
        kso_i, state_i, spin_i, kpt_i, k_weight, occ, eig, kso_weight = row
        if kso_weight < 1e-15:
            continue
        kso_i = int(kso_i)
        suffix = f"_{kso_i:04d}"
        kso = np.loadtxt(os.path.join(output_dir, f"{kso_file_prefix}{suffix}.out"))
        assert np.all(kso[:, :3] == grid[:, :3])
        weighted_ksos.append(kso_weight * kso[:, 3])

    return np.concatenate([grid[:, :3], np.sum(weighted_ksos, axis=0).reshape(-1, 1)], axis=1)

def _order_scalar_field(field):
    """
    Returns the scalar field ordered by norm of the position vector.
    """
    rows = []
    for row in field:
        rows.append(tuple(row) + tuple([np.linalg.norm(row[:3])]))
    struct_arr = np.array(
        rows,
        dtype=[('x', 'float'), ('y', 'float'), ('z', 'float'), ('value', 'float'), ('length', 'float')]
    )
    return np.sort(struct_arr, order='length')