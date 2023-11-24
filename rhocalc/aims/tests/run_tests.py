import ase.io

from aims_calc import *
from test_cases import calcs

# TODO: add path to local AIMS executable
AIMS_PATH = "/home/abbott/codes/new_aims/FHIaims/build/aims.230905.scalapack.mpi.x"

AIMS_KWARGS = {
    # TODO: add path to AIMS species defaults
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

# TODO: add SBATCH settings for HPC cluster
SBATCH_KWARGS = {
    "job-name": "checks",
    "nodes": 1,
    "time": "01:00:00",
    "mem-per-cpu": 2000,
    "partition": "standard",
}



# TODO: uncomment tests to run
def run_tests(calcs: dict):
    """
    Runs a series of tests on AIMS outputs fpr each of the calculations in
    `calcs`. Each test returns the quantities that failed comparison tests. This
    function returns a list of
    """
    failed = []

    # # Get integration to the formal number of electrons
    # failed.append(total_densities_integrate_to_n_electrons(calcs))

    # # np.allclose of weighted sum of KSO coeff matrices vs density matrix
    # failed.append(coeff_matrices_sum_to_density_matrix(calcs, return_all_calcs=True))

    # MAE of total density between a) that built from densmat and b) that imported from physics
    failed.append(density_from_densmat_equals_density_from_physics(calcs))

    # # MAE of total density built from RI coefficients vs that built from NAO densmat
    # failed.append(density_from_ri_equals_density_from_densmat(calcs))

    # # MAE of total density between a) built from sum of KSOs and b) built from densmat
    # failed.append(ksos_from_coeffmats_sum_to_density_from_densmat(calcs))

    # # MAE of total density between a) built from sum of KSOs and b) imported from physics
    # failed.append(ksos_from_coeffmats_sum_to_density_from_physics(calcs))

    # # MAE of total density between a) built from sum of KSOs and b) built from densmat
    # failed.append(ksos_from_ri_sum_to_density_from_densmat(calcs))

    # # MAE of total density between a) built from sum of KSOs and b) imported from physics
    # failed.append(ksos_from_ri_sum_to_density_from_physics(calcs))

    # # np.allclose of RI coeffs: a) weighted sum of KSOs vs total density from densmat
    # failed.append(ri_coeffs_for_ksos_sum_to_total_density_from_densmat(calcs))

    # Test the the RI overlap matric is symmetric
    # failed.append(overlap_is_symmetric(calcs))

    return failed



if __name__ == "__main__":

    # Add AIMS path to calcs
    for key, calc in calcs.items():
        calc["aims_path"] = AIMS_PATH

    # ======= comment out this block to just re-run RI fitting....

    # Run SCF
    run_scf(aims_kwargs=AIMS_KWARGS, sbatch_kwargs=SBATCH_KWARGS, calcs=calcs)

    # Wait for calcs to finish
    all_aims_outs = [os.path.join(f"{i}", "aims.out") for i in calcs.keys()]
    print("Files not yet finished SCF:")
    print(all_aims_outs)
    while len(all_aims_outs) > 0:
        for aims_out in all_aims_outs:
            if os.path.exists(aims_out):
                with open(aims_out, "r") as f:
                    if "Leaving FHI-aims." in f.read():
                        all_aims_outs.remove(aims_out)
                        print(all_aims_outs)

    # ======= .... (comment out this block to just re-run RI fitting)

    # Remove RI restart dirs if they are present
    for calc_i in calcs.keys():
        try:
            shutil.rmtree(f"{calc_i}/ri/")
        except FileNotFoundError:
            continue

    # Run RI fitting
    run_ri(aims_kwargs=AIMS_KWARGS, sbatch_kwargs=SBATCH_KWARGS, calcs=calcs, use_restart=True)

    # Wait for calcs to finish
    all_aims_outs = [os.path.join(f"{i}", "ri", "aims.out") for i in calcs.keys()]
    print("Files not yet finished RI-fitting:")
    print(all_aims_outs)
    while len(all_aims_outs) > 0:
        for aims_out in all_aims_outs:
            if os.path.exists(aims_out):
                with open(aims_out, "r") as f:
                    if "Leaving FHI-aims." in f.read():
                        all_aims_outs.remove(aims_out)
                        print(all_aims_outs)

    # Gather the failed results. Calling this function also prints the summary
    # of results.
    failed = run_tests(calcs={i: calc for i, calc in calcs.items()})

    