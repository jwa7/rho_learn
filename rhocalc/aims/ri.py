import os
from os.path import exists, join
import glob
import shutil

from rhocalc.aims import aims_calc


def set_settings_gloablly(dft_settings: dict):
    """Sets the DFT settings globally."""
    global_vars = globals()
    for key, value in dft_settings.items():
        global_vars[key] = value


def ri(dft_settings: dict):
    """Runs the FHI-aims RI procedure"""

    set_settings_gloablly(dft_settings)

    all_aims_outs = [join(RI_DIR(A), "aims.out") for A in SYSTEM_ID]

    calcs = {}
    for A, frame in zip(SYSTEM_ID, SYSTEM):
        if not exists(RI_DIR(A)):  # make RI dir
            os.makedirs(RI_DIR(A))
        calcs[A] = {"atoms": frame}

        # Get SCF calculation info and path to KS-orbital info
        calc_info = utils.unpickle_dict(join(SCF_DIR(A), "calc_info.pickle"))
        kso_info_path = join(SCF_DIR(A), "ks_orbital_info.out")

        if FIELD_NAME == "ildos":  # define KSO weights and write to file

            # Save LDOS settings
            ldos_kwargs = {k: v for k, v in LDOS.items()}
            ldos_kwargs["target_energy"] = calc_info[ldos_kwargs["target_energy"]]
            utils.pickle_dict(join(RI_DIR(A), "ldos_settings.pkl"), ldos_kwargs)
            print(f"Structure {A}, target_energy: {ldos_kwargs['target_energy']}")

            # Write KS-orbital weight vector
            kso_weights = aims_fields.get_kso_weight_vector_for_named_field(
                field_name=FIELD_NAME, kso_info_path=kso_info_path, **ldos_kwargs
            )
            np.savetxt(join(RI_DIR(A), "ks_orbital_weights.in"), kso_weights)

        elif FIELD_NAME == "edensity":
            assert RI.get("ri_fit_total_density") is not None

        # Specify tailored cube edges
        if RI.get("output") == ["cube ri_fit"] and CUBE["slab"] is True:
            calcs[A]["aims_kwargs"] = aims_calc.get_aims_cube_edges_slab(
                frame, CUBE.get("n_points")
            )

        # Copy density matrix restart
        for density_matrix in glob.glob(join(SCF_DIR(A), "D*.csc")):
            shutil.copy(density_matrix, RI_DIR(A))

    # And the general settings for all calcs
    aims_kwargs = BASE_AIMS.copy()
    aims_kwargs.update(RI)

    # Run the RI fitting procedure in AIMS
    aims_calc.run_aims_array(
        calcs=calcs,
        aims_path=AIMS_PATH,
        aims_kwargs=aims_kwargs,
        sbatch_kwargs=SBATCH,
        run_dir=RI_DIR,
        load_modules=HPC["load_modules"],
        export_vars=HPC["export_vars"],
        run_command="srun",
    )


def process_ri(dft_settings: dict):
    """Processes the RI outputs to metatensor."""
    set_settings_gloablly(dft_settings)
    aims_calc.process_aims_results_sbatch_array(
        "run-process-aims.sh",
        structure_idxs=SYSTEM_ID,
        run_dir=RI_DIR,
        process_what=["coeffs", "ovlp"],
        **SBATCH,
    )


def cleanup_ri(dft_settings: dict):
    """
    Removes density matrix restart files from the RI dir. Also removes the large RI ovlp
    matrix files if processed into TensorMap format.
    """
    # Remove the density matrix restart files from the RI dir
    for A in SYSTEM_ID:
        for density_matrix in glob.glob(join(RI_DIR(A), "D*.csc")):
            os.remove(density_matrix)

    # Remove ri_ovlp.out from the RI dir if now processed into a TensorMap
    for A in SYSTEM_ID:
        if exists(join(PROCESSED_DIR(A), "ri_ovlp.npz")):
            try:
                os.remove(join(RI_DIR(A), "ri_ovlp.out"))
            except FileNotFoundError:
                print(f"ri_ovlp.out already removed for structure {A}")
        else:
            print(f"Structure {A} not yet processed")
