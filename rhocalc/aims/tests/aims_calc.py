#!/usr/bin/bash
import inspect
import glob
import os
import shutil
from typing import List

import ase.io
import numpy as np
import metatensor

from rhocalc.aims import aims_calc, aims_parser
from rholearn import rotations

TOLERANCE = {"rtol": 1e-5, "atol": 1e-10}
DF_ERROR_TOLERANCE = 0.3 # percent MAE tolerance for RI density fitting


def run_scf(
    aims_kwargs: dict,
    sbatch_kwargs: dict,
    calcs: dict,
    load_modules: List[str] = ["intel", "intel-oneapi-mkl", "intel-oneapi-mpi"],
    run_command: str = "srun",
):
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
            load_modules=load_modules,
            run_command=run_command,
            **sbatch_kwargs_calc
        )

        # Run aims
        aims_calc.run_aims_in_dir(run_dir)

    return



def run_ri(
    aims_kwargs: dict,
    sbatch_kwargs: dict,
    calcs: dict,
    use_restart: bool = False,
    load_modules: List[str] = ["intel", "intel-oneapi-mkl", "intel-oneapi-mpi"],
    run_command: str = "srun",
):
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
            load_modules=load_modules,
            run_command=run_command,
            **sbatch_kwargs_calc
        )

        # Run aims
        aims_calc.run_aims_in_dir(run_dir)

    return

def run_rebuild(
    aims_kwargs: dict,
    sbatch_kwargs: dict,
    calcs: dict,
    load_modules: List[str] = ["intel", "intel-oneapi-mkl", "intel-oneapi-mpi"],
    run_command: str = "srun",
):
    """
    Runs an RI rebuild calculation.

    In each of the run directories, uses the previously calculated RI
    coefficients (i.e. <...>/ri/ri_coeffs.out) are copied to the rebuild
    directory as ri_coeffs.in and used to rebuild the density.
    """

    top_dir = os.getcwd()
    os.chdir(top_dir)

    for calc_i, calc in calcs.items():

        # Define run dir and AIMS path
        run_dir = f"{calc_i}/rebuild/"
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # Settings for control.in
        aims_kwargs_calc = aims_kwargs.copy()
        aims_kwargs_calc.update(calc["aims_kwargs"])
        aims_kwargs_calc.pop("elsi_restart")
        aims_kwargs_calc.update(
            {
                # ===== Force no SCF
                "sc_iter_limit": 0,
                "postprocess_anyway": True,
                "ri_fit_assume_converged": True,
                # ===== What we want to do
                "ri_fit_rebuild_from_coeffs": True,
                # ===== What to write as output
                "ri_fit_write_rebuilt_field": True,
                "ri_fit_write_rebuilt_field_cube": True,
                "output": ["cube ri_fit"],  # needed for cube files
            }
        )
        aims_kwargs_calc.update(aims_calc.get_aims_cube_edges(calc["atoms"]))

        # Copy RI coeffs as the input coeffs to rebuild from
        shutil.copy(
            f"{calc_i}/ri/ri_coeffs.out",
            f"{calc_i}/rebuild/ri_coeffs.in"
        )

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
            load_modules=load_modules,
            run_command=run_command,
            **sbatch_kwargs_calc
        )

        # Run aims
        aims_calc.run_aims_in_dir(run_dir)

    return

# def run_process(aims_kwargs: dict, sbatch_kwargs: dict, calcs: dict):
#     """
#     Processes the aims output files to convert raw arrays to metatensor format.

#     In each of the run directories, processes the calculated RI coefficients
#     (i.e. <...>/ri/ri_coeffs.out) and overlap matrix (<...>/ri/ri_ovlp.out).
#     """

#     top_dir = os.getcwd()
#     os.chdir(top_dir)

#     for calc_i, calc in calcs.items():

#         # Define run dir and AIMS path
#         run_dir = f"{calc_i}/rebuild/"
#         if not os.path.exists(run_dir):
#             os.makedirs(run_dir)

#         # Settings for control.in
#         aims_kwargs_calc = aims_kwargs.copy()
#         aims_kwargs_calc.update(calc["aims_kwargs"])
#         aims_kwargs_calc.pop("elsi_restart")
#         aims_kwargs_calc.update(
#             {
#                 # ===== Force no SCF
#                 "sc_iter_limit": 0,
#                 "postprocess_anyway": True,
#                 "ri_fit_assume_converged": True,
#                 # ===== What we want to do
#                 "ri_fit_rebuild_from_coeffs": True,
#                 # ===== What to write as output
#                 "ri_fit_write_rebuilt_field": True,
#                 "ri_fit_write_rebuilt_field_cube": True,
#                 "output": ["cube ri_fit"],  # needed for cube files
#             }
#         )
#         aims_kwargs_calc.update(aims_calc.get_aims_cube_edges(calc["atoms"]))

#         # Copy RI coeffs as the input coeffs to rebuild from
#         shutil.copy(
#             f"{calc_i}/ri/ri_coeffs.out",
#             f"{calc_i}/rebuild/ri_coeffs.in"
#         )

#         # Settings for sbatch run script
#         sbatch_kwargs_calc = sbatch_kwargs.copy()
#         sbatch_kwargs_calc.update(calc["sbatch_kwargs"])

#         # Write AIMS input files
#         aims_calc.write_input_files(
#             atoms=calc["atoms"], 
#             run_dir=run_dir, 
#             aims_kwargs=aims_kwargs_calc,
#         )

#         # Write sbatch run script
#         aims_calc.write_aims_sbatch(
#             fname=os.path.join(run_dir, "run-aims.sh"), 
#             aims=calc["aims_path"], 
#             load_modules=[],
#             **sbatch_kwargs_calc
#         )

#         # Run aims
#         aims_calc.run_aims_in_dir(run_dir)

#     return

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
    Tests that the density built from the density matrix constructed manually in
    AIMS equals the density imported from physics.f90 in AIMS.
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
    Tests that the RI density reasonably approximates the density built from the
    the density matrix constructed in AIMS. As there will be some density
    fitting error associated with the RI procedure, exact equivalence is not
    checked but rather that the error is within a tolerance (usually between
    0.1% - 1% depending on settings).
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
        if mae < DF_ERROR_TOLERANCE:
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
    total density formed from the NAO density matrix, within a tolerance for the
    denisty fitting error.
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
        if mae < DF_ERROR_TOLERANCE:
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
        if mae < DF_ERROR_TOLERANCE:
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
    equivalent to the RI coefficients for the total density built from the
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
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"

        # Load the density matrix constructed in AIMS
        ri_ovlp = np.loadtxt(os.path.join(ri_dir, "ri_ovlp.out"))
        dim = int(np.sqrt(ri_ovlp.shape[0]))
        ri_ovlp = ri_ovlp.reshape((dim, dim))

        # Check pass/fail
        mae = np.abs(ri_ovlp - ri_ovlp.T).mean()
        if np.allclose(mae, 0, **TOLERANCE):
            print(f"    PASS - Calc {calc_i} - {calc['name']}")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}")
            all_passed = False
            failed_calcs[calc_i] = ri_ovlp
    print("\n")
    return failed_calcs

def w_equals_Sc(calcs: dict):
    """
    Tests that the relation w = S . c holds, i.e. that the RI projections (w)
    are equal to the dot product of the RI overlap matrix (S) and the RI
    coefficients (c).
    """
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"

        # Load the RI coefficients and projections
        ri_coeffs = np.loadtxt(os.path.join(ri_dir, "ri_coeffs.out"))
        ri_projs = np.loadtxt(os.path.join(ri_dir, "ri_projs.out"))
        dim = int(ri_coeffs.shape[0])

        # Load the density matrix constructed in AIMS and reshape to square.
        # Fortran vs C ordering for the reshape doesn't matter here as the
        # matrix is symmetric.
        ri_ovlp = np.loadtxt(os.path.join(ri_dir, "ri_ovlp.out"))
        ri_ovlp = ri_ovlp.reshape((dim, dim))

        # Check pass/fail - i.e. w - S . c = 0
        mae = np.abs(ri_projs - ri_ovlp @ ri_coeffs).mean()
        if np.allclose(mae, 0, **TOLERANCE):
            print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae}")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae}")
            all_passed = False
            failed_calcs[calc_i] = (ri_projs, ri_ovlp, ri_coeffs)
    print("\n")
    return failed_calcs




# ===== NOTE: not used as in RI coefficients not re-written in current API.
# ===== This test was only used for debugging, and requires modification of
# ===== AIMS source code in subroutine `read_ri_coeffs_and_rebuild_field` to
# ===== write the re-re-ordered coefficients to file.
# def rebuild_reordering_coeffs_in_equals_coeffs_out(calcs: dict):
#     """
#     Tests that the input RI coeffs used in a RI rebuild calculation are the same
#     as those output. This tests that the internal oreding/reordering and
#     application/undoing of Condon-Shortley convention going from rho_learn
#     convention -to-> AIMS convention -back_to-> rho_learn convention is
#     consistent.
#     """
#     print(f"Test: {inspect.stack()[0][3]}")
#     # Iterate over calculations
#     all_passed = True
#     failed_calcs = {}
#     for calc_i, calc in calcs.items():
#         ri_dir = f"{calc_i}/ri/"
#         rebuild_dir = f"{calc_i}/rebuild/"

#         # Load the density matrix constructed in AIMS
#         ri_coeffs_in = np.loadtxt(os.path.join(rebuild_dir, "ri_coeffs.in"))
#         ri_coeffs_out = np.loadtxt(os.path.join(rebuild_dir, "ri_coeffs.out"))

#         # Check pass/fail
#         mae = np.abs(ri_coeffs_in - ri_coeffs_out).mean()
#         if np.allclose(mae, 0, **TOLERANCE):
#             print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae}")
#         else:
#             print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae}")
#             all_passed = False
#             failed_calcs[calc_i] = (ri_coeffs_in, ri_coeffs_out)
#     print("\n")
#     return failed_calcs


def rebuilt_density_equal_between_ri_fit_and_ri_rebuild(calcs: dict):
    """
    Tests that the total density rebuilt from RI coefficients within the RI
    fitting procedure is exactly equivalent to those rebuilt in a separate
    calculation from the same coefficients.
    """
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"
        rebuild_dir = f"{calc_i}/rebuild/"

        # Load the RI and rebuilt densities
        rho_ri = np.loadtxt(os.path.join(ri_dir, "rho_ri.out"))
        rho_rebuilt = np.loadtxt(os.path.join(rebuild_dir, "rho_rebuilt.out"))
        grid = np.loadtxt(os.path.join(rebuild_dir, "partition_tab.out"))
        assert np.allclose(
            grid, 
            np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))
        )

        # Check for exact equivalence with the density matrix
        mae = _get_percent_mae_between_fields(input=rho_rebuilt, target=rho_ri, grid=grid)
        if mae < TOLERANCE["rtol"]:
            print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae} %")
            all_passed = False
            failed_calcs[calc_i] = (rho_rebuilt, rho_ri)
    print("\n")
    return failed_calcs


def ri_coeffs_so3_equivariant(calcs: dict):
    """
    Tests that the RI decomposition of the target scalar field is equivariant to
    O3 transformation. The test cases that test these have indices: [1,], with
    their rotated equivalent test case being indexed by [-1,] respectively.
    """
    test_cases = {1: "water_cluster"}
    print(f"Test: {inspect.stack()[0][3]}")
    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, fname in test_cases.items():
        calc = calcs[calc_i]

        # Read the frames and rotation angles
        frame = ase.io.read(f"systems/{fname}.xyz")
        frame_rot = ase.io.read(f"systems/{fname}_so3.xyz")
        angles = frame_rot.info["angles"]
        wigner_d = rotations.WignerDReal(lmax=8, angles=angles)

        # Define the dirs
        ri_dir = f"{calc_i}/ri/"
        ri_dir_rot = f"{1000 + calc_i}/ri/"

        # Load the basis set definition
        lmax, nmax = aims_parser.extract_basis_set_info(frame, ri_dir)

        # Load the coeff vectors
        c = np.loadtxt(os.path.join(ri_dir, "ri_coeffs.out"))
        c_rot = np.loadtxt(os.path.join(ri_dir_rot, "ri_coeffs.out"))
        # c = metatensor.load(os.path.join(ri_dir, "processed", "ri_coeffs.npz"))
        # c_rot = metatensor.load(os.path.join(ri_dir_rot, "processed", "ri_coeffs.npz"))

        # Rotate the original coefficients
        c_unrot_rot = wigner_d.rotate_coeff_vector(frame, c, lmax, nmax)
        # c_unrot_rot = wigner_d.transform_tensormap_o3(c)

        mae = np.abs(c_unrot_rot - c_rot).mean()
        if mae < TOLERANCE["rtol"]:
            print(f"    PASS - Calc {calc_i} - {calc['name']}. MAE: {mae}")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}. MAE: {mae}")
            all_passed = False
            failed_calcs[calc_i] = (c_unrot_rot, c_rot)

        # # Check for equivalence
        # if metatensor.allclose(c_unrot_rot, c_rot, **TOLERANCE):
        #     print(f"    PASS - Calc {calc_i} - {calc['name']}.")
        # else:
        #     print(f"    FAIL - Calc {calc_i} - {calc['name']}.")
        #     all_passed = False
        #     failed_calcs[calc_i] = (c_unrot_rot, c_rot)

    print("\n")
    return failed_calcs
   

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