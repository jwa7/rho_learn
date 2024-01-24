import ase.io

from aims_calc import *
from test_cases import calcs

# TODO: add SBATCH settings for HPC cluster
SBATCH_KWARGS = {
    "job-name": "tests",
    "nodes": 1,
    "time": "01:00:00",
    "mem-per-cpu": 2000,
    "partition": "standard",
}

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

# Settings for the RI rebuild procedure
# REBUILD_KWARGS = {
#     # ===== Force no SCF
#     "sc_iter_limit": 0,
#     "postprocess_anyway": True,
#     "ri_fit_assume_converged": True,
#     # ===== What we want to do
#     "ri_fit_rebuild_from_coeffs": True,
#     # ===== Specific settings for RI rebuild
#     "ri_fit_ovlp_cutoff_radius": RI_KWARGS["ri_fit_ovlp_cutoff_radius"],
#     "default_max_l_prodbas": RI_KWARGS["default_max_l_prodbas"],
#     # ===== What to write as output
#     "ri_fit_write_rebuilt_field": True,
#     "ri_fit_write_rebuilt_field_cube": True,
#     "output": ["cube ri_fit"],  # needed for cube files
# }

# TODO: uncomment tests to run
def run_tests(calcs: dict, test_ksos: bool = False):
    """
    Runs a series of tests on AIMS outputs fpr each of the calculations in
    `calcs`. Each test returns the quantities that failed comparison tests. This
    function returns a list of
    """
    failed = []

    # ===== Tests for `generate_sanity_check_data_total_density`

    # Get integration to the formal number of electrons
    failed.append(total_densities_integrate_to_n_electrons(calcs))

    # MAE of total density between a) that built from densmat and b) that imported from physics
    failed.append(density_from_densmat_equals_density_from_physics(calcs))

    # MAE of total density built from RI coefficients vs that built from NAO densmat
    failed.append(density_from_ri_equals_density_from_densmat(calcs))

    # Test the the RI overlap matric is symmetric
    failed.append(overlap_is_symmetric(calcs))

    # Test that w = Sc for the output RI projections, overlap, and coefficients
    # (respectively)
    failed.append(w_equals_Sc(calcs))

    # NOTE: test not runnable with current API as RI-coeffs not re-written to
    # file. Kept here in case of debugging. 
    # RI coeffs read in, convention changed, and changed back gives the same
    # coeffs as read in
    # failed.append(rebuild_reordering_coeffs_in_equals_coeffs_out(calcs))

    # MAE of total density rebuilt from RI coefficients within the RI
    # fitting procedure is exactly equivalent to those rebuilt in a separate
    # calculation from the same coefficients
    failed.append(rebuilt_density_equal_between_ri_fit_and_ri_rebuild(calcs))

    # ===== Tests for `generate_sanity_check_data_ksos`
    if test_ksos:

        # np.allclose of weighted sum of KSO coeff matrices vs density matrix
        failed.append(coeff_matrices_sum_to_density_matrix(calcs, return_all_calcs=True))

        # MAE of total density between a) built from sum of KSOs and b) built from densmat
        failed.append(ksos_from_coeffmats_sum_to_density_from_densmat(calcs))

        # MAE of total density between a) built from sum of KSOs and b) imported from physics
        failed.append(ksos_from_coeffmats_sum_to_density_from_physics(calcs))

        # MAE of total density between a) built from sum of KSOs and b) built from densmat
        failed.append(ksos_from_ri_sum_to_density_from_densmat(calcs))

        # MAE of total density between a) built from sum of KSOs and b) imported from physics
        failed.append(ksos_from_ri_sum_to_density_from_physics(calcs))

        # np.allclose of RI coeffs: a) weighted sum of KSOs vs total density from densmat
        failed.append(ri_coeffs_for_ksos_sum_to_total_density_from_densmat(calcs))

    return failed



if __name__ == "__main__":

    scf = True
    ri = True
    rebuild = True

    # Add AIMS path to calcs
    for key, calc in calcs.items():
        calc["aims_path"] = AIMS_PATH

    # ======= 1. Run SCF procedure

    if scf:
        
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

    # ======= 2. Run RI fitting

    if ri:

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

    # ======= 3. Run RI rebuild

    if rebuild:

        # Remove RI rebuild dirs if they are present
        for calc_i in calcs.keys():
            try:
                shutil.rmtree(f"{calc_i}/rebuild/")
            except FileNotFoundError:
                continue

        # Run RI fitting
        run_rebuild(aims_kwargs=AIMS_KWARGS, sbatch_kwargs=SBATCH_KWARGS, calcs=calcs)

        # Wait for calcs to finish
        all_aims_outs = [os.path.join(f"{i}", "rebuild", "aims.out") for i in calcs.keys()]
        print("Files not yet finished RI-rebuild:")
        print(all_aims_outs)
        while len(all_aims_outs) > 0:
            for aims_out in all_aims_outs:
                if os.path.exists(aims_out):
                    with open(aims_out, "r") as f:
                        if "Leaving FHI-aims." in f.read():
                            all_aims_outs.remove(aims_out)
                            print(all_aims_outs)

    # ======= 4. Run tests on output files and print results summary

    # Gather the failed results. Calling this function also prints the summary
    # of results.
    failed = run_tests(calcs={i: calc for i, calc in calcs.items()}, test_ksos=False)
    