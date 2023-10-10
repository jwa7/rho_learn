#!/usr/bin/bash
import inspect
import glob
import os
import shutil

import ase.io
import numpy as np
from rhocalc.aims import aims_calc

ALLCLOSE_TOLS = {"rtol": 1e-5, "atol": 1e-15}

aims_path = "/home/abbott/codes/new_aims/FHIaims/build/aims.230905.scalapack.mpi.x"
aims_kwargs = {
    "species_dir": "/home/abbott/codes/new_aims/FHIaims/species_defaults/defaults_2020/tight",
    "xc": "pbe0",
    "spin": "none",
    "relativistic": "none",
    "charge": 0,
    "sc_accuracy_rho": 1e-8,
    "wave_threshold": 1e-8,
    "elsi_restart": "write 1",
    # "elsi_restart": "read",
    # "sc_iter_limit": 0,
    # "postprocess_anyway": True,
    # ======================================== ri_fit: what to fit to
    # "ri_fit_total_density": True,
    # "ri_fit_ldos": True,
    # "ri_fit_ildos": True,
    # ======================================== ri_fit: settings
    # "ri_fit_ovlp_cutoff_radius": 2.0,
    # "ri_fit_assume_converged": True,
    # ======================================== ri_fit: what to write as outputs
    # "ri_fit_write_nao_coeff_matrix":True,
    # "ri_fit_write_coeffs": True,
    # "ri_fit_write_projs": True,
    # "ri_fit_write_ovlp": True,
    # "ri_fit_write_ref_field": True,
    # "ri_fit_write_rebuilt_field": True,
    # "ri_fit_write_ref_field_cube": True,
    # "ri_fit_write_rebuilt_field_cube": True,
    # "ri_fit_sanity_check": True,
    # ======================================== Keywords we don't want to have to use
    # "output": ["cube total_density"],
    # "collect_eigenvectors": True,
}
sbatch_kwargs = {
    "job-name": "checks_scf",
    "nodes": 1,
    "time": "00:30:00",
    "mem-per-cpu": 2000,
}

# Define parameters that are different for each run
calcs = {
    0: {
        "name": "cluster, serial",
        "atoms": ase.io.read("water_cluster.xyz"),
        "aims_path": aims_path,
        "aims_kwargs": {},
        "sbatch_kwargs": {"ntasks-per-node": 1},
    },
    1: {   
        "name": "cluster, parallel",
        "atoms": ase.io.read("water_cluster.xyz"),
        "aims_path": aims_path,
        "aims_kwargs": {},
        "sbatch_kwargs": {"ntasks-per-node": 4},
    },
    2: {
        "name": "periodic, 1 kpt, serial",
        "atoms": ase.io.read("water_periodic.xyz"),
        "aims_path": aims_path,
        "aims_kwargs": {"k_grid": [1, 1, 1]},
        "sbatch_kwargs": {"ntasks-per-node": 1},
    },
    3: {  
        "name": "periodic, 1 kpt, parallel, n_tasks > n_kpts",
        "atoms": ase.io.read("water_periodic.xyz"),
        "aims_path": aims_path,
        "aims_kwargs": {"k_grid": [1, 1, 1]},
        "sbatch_kwargs": {"ntasks-per-node": 10},
    },
    4: {  
        "name": "periodic, 4 kpt, serial",
        "atoms": ase.io.read("water_periodic.xyz"),
        "aims_path": aims_path,
        "aims_kwargs": {"k_grid": [1, 2, 2]},
        "sbatch_kwargs": {"ntasks-per-node": 1},
    },
    5: {  
        "name": "periodic, 4 kpt, parallel, 1 < n_tasks < n_kpts",
        "atoms": ase.io.read("water_periodic.xyz"),
        "aims_path": aims_path,
        "aims_kwargs": {"k_grid": [1, 2, 2]},
        "sbatch_kwargs": {"ntasks-per-node": 3},
    },
    6: {  
        "name": "periodic, 4 kpt, parallel, 1 < n_kpts == n_tasks",
        "atoms": ase.io.read("water_periodic.xyz"),
        "aims_path": aims_path,
        "aims_kwargs": {"k_grid": [1, 2, 2]},
        "sbatch_kwargs": {"ntasks-per-node": 4},
    },
    7: {  
        "name": "periodic, 4 kpt, parallel, 1 < n_kpts < n_tasks",
        "atoms": ase.io.read("water_periodic.xyz"),
        "aims_path": aims_path,
        "aims_kwargs": {"k_grid": [1, 2, 2]},
        "sbatch_kwargs": {"ntasks-per-node": 7},
    },
    8: {  
        "name": "periodic, 4 kpt, parallel, 1 < n_tasks < n_kpts, collect_eigenvectors",
        "atoms": ase.io.read("water_periodic.xyz"),
        "aims_path": aims_path,
        "aims_kwargs": {"k_grid": [1, 2, 2], "collect_eigenvectors": True},
        "sbatch_kwargs": {"ntasks-per-node": 3},
    },
    9: {  
        "name": "periodic, 4 kpt, parallel, 1 < n_kpts == n_tasks, collect_eigenvectors",
        "atoms": ase.io.read("water_periodic.xyz"),
        "aims_path": aims_path,
        "aims_kwargs": {"k_grid": [1, 2, 2], "collect_eigenvectors": True},
        "sbatch_kwargs": {"ntasks-per-node": 4},
    },
    10: {  
        "name": "periodic, 4 kpt, parallel, 1 < n_kpts < n_tasks, collect_eigenvectors",
        "atoms": ase.io.read("water_periodic.xyz"),
        "aims_path": aims_path,
        "aims_kwargs": {"k_grid": [1, 2, 2], "collect_eigenvectors": True},
        "sbatch_kwargs": {"ntasks-per-node": 7},
    },
}


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
                "output": ["cube total_density"],
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

def test_total_densities_integrate_to_n_electrons(calcs):
    """
    First finds the formal number of electrons in the system as the sum of
    atomic numbers of the nuclei. Calculates the number of electrons from the
    sum of state occupations printed in ks_orbital_info.out. Integrates the
    total electron densities on the AIMS grid to find the number of electrons
    present in the densities imported from physics.f90 and that constructed from
    the density matrix. Checks that all of these are equal.
    """

    print(f"Test function: {inspect.stack()[0][3]}")

    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"

        # Get the formal number of electrons - assumes charge neutral
        N_formal = ase.io.read(os.path.join(ri_dir, "geometry.in")).get_atomic_numbers().sum()

        # Get the number of electrons as printed in ks_orbital_info.out
        ks_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))
        k_weights = ks_info[:, 4]
        occs = k_weights * ks_info[:, 5]
        N_info = np.sum(occs)

        # Load the densities
        rho_from_physics = np.loadtxt(os.path.join(ri_dir, f"rho_physics.out"))
        rho_from_densmat = np.loadtxt(os.path.join(ri_dir, f"rho_ref.out"))
        grid = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))
        assert np.all(rho_from_physics[:, :3] == grid[:, :3])
        assert np.all(rho_from_densmat[:, :3] == grid[:, :3])

        N_from_physics = np.dot(rho_from_physics[:, 3], grid[:, 3])
        N_from_densmat = np.dot(rho_from_densmat[:, 3], grid[:, 3])

        # Check for equivalence
        if np.all(
            [
                np.isclose(n, N_formal, **ALLCLOSE_TOLS) for n in [N_info, N_from_densmat, N_from_physics]
            ]
        ):
            print(f"    PASS - Calc {calc_i} - {calc['name']}")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}, {(N_formal, N_info, N_from_physics, N_from_densmat)}")
            all_passed = False
            failed_calcs[calc_i] = (N_formal, N_info, N_from_physics, N_from_densmat)

    print("\n")
    return failed_calcs

def test_built_total_density_equals_physics_total_density(calcs: dict):
    """
    Tests that the coefficient matrices for each KSO, wieghted by their
    electronic occupation, sum to the density matrix constructed in AIMS by the
    same procedure. 
    """
    print(f"Test function: {inspect.stack()[0][3]}")

    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"

        # Load the density matrix constructed in AIMS
        rho_from_densmat = np.loadtxt(os.path.join(ri_dir, f"rho_ref.out"))
        grid = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))
        assert np.allclose(rho_from_densmat[:, :3], grid[:, :3], **ALLCLOSE_TOLS)
        rho_from_densmat = rho_from_densmat[:, 3]

        # Load the density matrix imported from physics.f90
        rho_physics = np.loadtxt(os.path.join(ri_dir, f"rho_physics.out"))
        assert np.allclose(rho_physics[:, :3], grid[:, :3], **ALLCLOSE_TOLS)
        rho_physics = rho_physics[:, 3]

        mae = 100 * np.dot(np.abs(rho_from_densmat - rho_physics), grid[:, 3]) / np.dot(rho_physics, grid[:, 3])

        # Check for equivalence
        # if np.allclose(rho_physics, rho_from_densmat, **ALLCLOSE_TOLS):
        #     print(f"    PASS - Calc {calc_i} - {calc['name']}, MAE: {mae} %")
        # else:
        #     print(f"    FAIL - Calc {calc_i} - {calc['name']}, MAE: {mae} %")
        #     all_passed = False
        #     failed_calcs[calc_i] = (rho_physics, rho_from_densmat)
        if mae < 1e-2:
            print(f"    PASS - Calc {calc_i} - {calc['name']}, MAE: {mae} %")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}, MAE: {mae} %")
            all_passed = False
            failed_calcs[calc_i] = (rho_physics, rho_from_densmat)

    print("\n")
    return failed_calcs


# ===== Test sum of SCF quantities =====
def test_coeff_matrices_sum_to_density_matrix_scf(calcs: dict):
    """
    Tests that the coefficient matrices for each KSO, wieghted by their
    electronic occupation, sum to the density matrix constructed in AIMS by the
    same procedure. 
    """
    print(f"Test function: {inspect.stack()[0][3]}")

    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"
        ks_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))

        # Load the density matrix constructed in AIMS
        densmat_built_in_aims = np.loadtxt(os.path.join(ri_dir, "nao_coeff_matrix.out"))
        dim = int(np.sqrt(densmat_built_in_aims.shape[0]))
        densmat_built_in_aims = densmat_built_in_aims.reshape((dim, dim))

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

        # Check for equivalence with the density matrix
        if np.allclose(densmat_from_coeffmats, densmat_built_in_aims, **ALLCLOSE_TOLS):
            print(f"    PASS - Calc {calc_i} - {calc['name']}")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}")
            all_passed = False
            failed_calcs[calc_i] = (densmat_from_coeffmats, densmat_built_in_aims)
    print("\n")
    return failed_calcs
    
def test_ksos_sum_to_total_density_from_densmat_scf(calcs: dict):
    """
    Tests that the coefficient matrices for each KSO, wieghted by their
    electronic occupation, sum to the density matrix constructed in AIMS by the
    same procedure. 
    """
    print(f"Test function: {inspect.stack()[0][3]}")

    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"
        ks_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))

        # Load the density matrix constructed in AIMS
        rho_from_densmat = np.loadtxt(os.path.join(ri_dir, "rho_ref.out"))
        grid_points = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))[:, :3]
        assert np.all(rho_from_densmat[:, :3] == grid_points)
        rho_from_densmat = rho_from_densmat[:, 3]

        # Perform weighted sum over coeff matrices
        rho_from_ksos = []
        for row in ks_info:
            kso_i, state_i, spin_i, kpt_i, k_weight, occ, eig, kso_weight = row
            if kso_weight < 1e-15:
                continue
            kso_i = int(kso_i)
            suffix = f"_{kso_i:04d}"
            kso = np.loadtxt(os.path.join(ri_dir, f"kso_ref{suffix}.out"))
            assert np.all(kso[:, :3] == grid_points)
            rho_from_ksos.append(kso_weight * kso[:, 3])

        rho_from_ksos = np.sum(rho_from_ksos, axis=0)

        # Check for equivalence with the density matrix
        if np.allclose(rho_from_ksos, rho_from_densmat, **ALLCLOSE_TOLS):
            print(f"    PASS - Calc {calc_i} - {calc['name']}")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}")
            all_passed = False
            failed_calcs[calc_i] = (rho_from_ksos, rho_from_densmat)
    print("\n")
    return failed_calcs

def test_ksos_sum_to_total_density_from_physics_scf(calcs: dict):
    """
    Tests that the coefficient matrices for each KSO, wieghted by their
    electronic occupation, sum to the density matrix constructed in AIMS by the
    same procedure. 
    """
    print(f"Test function: {inspect.stack()[0][3]}")

    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"
        ks_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))

        # Load the density matrix constructed in AIMS
        rho_from_physics = np.loadtxt(os.path.join(ri_dir, "rho_physics.out"))
        grid_points = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))[:, :3]
        assert np.all(rho_from_physics[:, :3] == grid_points)
        rho_from_physics = rho_from_physics[:, 3]

        # Perform weighted sum over coeff matrices
        rho_from_ksos = []
        for row in ks_info:
            kso_i, state_i, spin_i, kpt_i, k_weight, occ, eig, kso_weight = row
            if kso_weight < 1e-15:
                continue
            kso_i = int(kso_i)
            suffix = f"_{kso_i:04d}"
            kso = np.loadtxt(os.path.join(ri_dir, f"kso_ref{suffix}.out"))
            assert np.all(kso[:, :3] == grid_points)
            rho_from_ksos.append(kso_weight * kso[:, 3])

        rho_from_ksos = np.sum(rho_from_ksos, axis=0)

        # Check for equivalence with the density matrix
        if np.allclose(rho_from_ksos, rho_from_physics, **ALLCLOSE_TOLS):
            print(f"    PASS - Calc {calc_i} - {calc['name']}")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}")
            all_passed = False
            failed_calcs[calc_i] = (rho_from_ksos, rho_from_physics)
    print("\n")
    return failed_calcs



# ===== Test sum of RI quantities =====

def test_kso_ri_coeffs_sum_to_total_density_ri_coeffs(calcs: dict):
    """
    Tests that the coefficient matrices for each KSO, wieghted by their
    electronic occupation, sum to the density matrix constructed in AIMS by the
    same procedure. 
    """
    print(f"Test function: {inspect.stack()[0][3]}")

    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"
        ks_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))

        # Load the density matrix constructed in AIMS
        rho_ri_coeffs = np.loadtxt(os.path.join(ri_dir, "ri_coeffs.out"))

        # Perform weighted sum over coeff matrices
        rho_from_ksos = []
        for row in ks_info:
            kso_i, state_i, spin_i, kpt_i, k_weight, occ, eig, kso_weight = row
            if kso_weight < 1e-15:
                continue
            kso_i = int(kso_i)
            suffix = f"_{kso_i:04d}"
            kso_ri_coeffs = np.loadtxt(os.path.join(ri_dir, f"ri_coeffs{suffix}.out"))
            assert k_weight * occ == kso_weight
            rho_from_ksos.append(kso_weight * kso_ri_coeffs)

        rho_from_ksos = np.sum(rho_from_ksos, axis=0)

        # Check for equivalence with the density matrix
        if np.allclose(rho_from_ksos, rho_ri_coeffs, **ALLCLOSE_TOLS):
            print(f"    PASS - Calc {calc_i} - {calc['name']}")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}")
            all_passed = False
            failed_calcs[calc_i] = (rho_from_ksos, rho_ri_coeffs)
    print("\n")
    return failed_calcs

def test_ksos_sum_to_total_density_from_densmat_ri(calcs: dict):
    """
    Tests that the coefficient matrices for each KSO, wieghted by their
    electronic occupation, sum to the density matrix constructed in AIMS by the
    same procedure. 
    """
    print(f"Test function: {inspect.stack()[0][3]}")

    # Iterate over calculations
    all_passed = True
    failed_calcs = {}
    for calc_i, calc in calcs.items():
        ri_dir = f"{calc_i}/ri/"
        ks_info = np.loadtxt(os.path.join(ri_dir, "ks_orbital_info.out"))

        # Load the density matrix constructed in AIMS
        rho_from_densmat = np.loadtxt(os.path.join(ri_dir, "rho_ri.out"))
        grid_points = np.loadtxt(os.path.join(ri_dir, "partition_tab.out"))[:, :3]
        assert np.all(rho_from_densmat[:, :3] == grid_points)
        rho_from_densmat = rho_from_densmat[:, 3]

        # Perform weighted sum over coeff matrices
        rho_from_ksos = []
        for row in ks_info:
            kso_i, state_i, spin_i, kpt_i, k_weight, occ, eig, kso_weight = row
            if kso_weight < 1e-15:
                continue
            kso_i = int(kso_i)
            suffix = f"_{kso_i:04d}"
            kso = np.loadtxt(os.path.join(ri_dir, f"kso_ri{suffix}.out"))
            assert np.all(kso[:, :3] == grid_points)
            rho_from_ksos.append(kso_weight * kso[:, 3])

        rho_from_ksos = np.sum(rho_from_ksos, axis=0)

        # Check for equivalence with the density matrix
        if np.allclose(rho_from_ksos, rho_from_densmat, **ALLCLOSE_TOLS):
            print(f"    PASS - Calc {calc_i} - {calc['name']}")
        else:
            print(f"    FAIL - Calc {calc_i} - {calc['name']}")
            all_passed = False
            failed_calcs[calc_i] = (rho_from_ksos, rho_from_densmat)
    print("\n")
    return failed_calcs
