import os
from os.path import exists, join
import glob
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np

import metatensor.torch as mts

from rhocalc.aims import aims_calc, aims_fields
from rhocalc.ase import structure_builder
from rhocalc.cube import rho_cube
from rholearn import data, utils


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

        # Make RI dir and copy settings file
        if not exists(RI_DIR(A)):
            os.makedirs(RI_DIR(A))
        shutil.copy("dft_settings.py", RI_DIR(A))

        calcs[A] = {"atoms": frame}

        # Get SCF calculation info and path to KS-orbital info
        calc_info = utils.unpickle_dict(join(SCF_DIR(A), "calc_info.pickle"))
        kso_info_path = join(SCF_DIR(A), "ks_orbital_info.out")

        if FIELD_NAME == "ildos":  # define KSO weights and write to file

            assert RI.get("ri_fit_total_density") is None
            assert RI.get("ri_fit_field_from_kso_weights") is not None

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

        elif FIELD_NAME in ["homo", "lumo"]:
            # Write KS-orbital weight vector
            kso_weights = aims_fields.get_kso_weight_vector_for_named_field(
                field_name=FIELD_NAME, kso_info_path=kso_info_path,
            )
            np.savetxt(join(RI_DIR(A), "ks_orbital_weights.in"), kso_weights)


        elif FIELD_NAME == "edensity":
            assert RI.get("ri_fit_total_density") is not None
            assert RI.get("ri_fit_field_from_kso_weights") is None

        elif FIELD_NAME == "edensity_from_weights":
            
            assert RI.get("ri_fit_total_density") is None
            assert RI.get("ri_fit_field_from_kso_weights") is not None

            kso_weights = aims_fields.get_kso_weight_vector_for_named_field(
                field_name="edensity", kso_info_path=kso_info_path,
            )
            np.savetxt(join(RI_DIR(A), "ks_orbital_weights.in"), kso_weights)

        # Specify tailored cube edges
        if "cube ri_fit" in RI.get("output"):
            if CUBE["slab"] is True:
                calcs[A]["aims_kwargs"] = aims_calc.get_aims_cube_edges_slab(
                    frame,
                    CUBE.get("n_points"),
                    z_min=None,  # we want the the cube file generated on the full slab
                    z_max=None,
                )
            else:
                calcs[A]["aims_kwargs"] = aims_calc.get_aims_cube_edges(
                    frame,
                    CUBE.get("n_points"),
                )

        # Copy density matrix restart
        for dm in glob.glob(join(SCF_DIR(A), "D*.csc")):
            file_name = dm[dm.rfind(SCF_DIR(A)) + len(SCF_DIR(A)) + 1:]
            if file_name in os.listdir(RI_DIR(A)):  # already copied
                continue
            else:
                shutil.copy(dm, RI_DIR(A))

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
        process_what=PROCESS_WHAT,
        **SBATCH,
    )


def cleanup_ri(dft_settings: dict):
    """
    Removes density matrix restart files from the RI dir. Also removes the large RI ovlp
    matrix files if processed into TensorMap format.
    """
    set_settings_gloablly(dft_settings)
    # Remove the density matrix restart files from the RI dir
    for A in SYSTEM_ID:
        for density_matrix in glob.glob(join(RI_DIR(A), "D*.csc")):
            os.remove(density_matrix)

    # Remove ri_ovlp.out from the RI dir if now processed into a TensorMap
    for A in SYSTEM_ID:
        if "coeffs" in PROCESS_WHAT:
            if not exists(join(PROCESSED_DIR(A), "ri_coeffs.npz")):
                raise ValueError(f"Coeffs for structure {A} not yet processed")
        if "ovlp" in PROCESS_WHAT:
            if exists(join(PROCESSED_DIR(A), "ri_ovlp.npz")):
                try:
                    os.remove(join(RI_DIR(A), "ri_ovlp.out"))
                except FileNotFoundError:
                    warnings.warn(f"ri_ovlp.out already removed for structure {A}")
            else:
                raise ValueError(f"Ovlp for structure {A} not yet processed")


def rebuild(dft_settings: dict):
    """
    Runs a FHI-aims RI rebuild procedure.

    Assumes RI coefficients have already been processed into metatensor format. Loads
    them from the processed directory, and applies masking/unmasking if applicable.
    """

    set_settings_gloablly(dft_settings)

    coeff = []
    for A, frame in zip(SYSTEM_ID, SYSTEM):

        # Load RI coefficients as TensorMap
        coeff_A = mts.load(join(PROCESSED_DIR(A), "ri_coeffs.npz"))

        # Mask coeffs (and ovlp) if applicable
        if MASK is not None:

            # Find idxs to keep
            idxs_surface, idxs_buffer, idxs_bulk = (
                structure_builder.get_atom_idxs_by_region(
                    frame, MASK["surface_depth"], MASK["buffer_depth"]
                )
            )
            idxs_to_keep = list(idxs_surface) + list(idxs_buffer)

            # Mask coeffs
            coeff_A_masked = data.mask_coeff_vector_tensormap(coeff_A, idxs_to_keep)
            mts.save(
                join(
                    PROCESSED_DIR(A),
                    f"ri_coeffs_masked_{MASK['surface_depth']}_{MASK['buffer_depth']}.npz",
                ),
                coeff_A_masked,
            )
            coeff_A_unmasked = data.unmask_coeff_vector_tensormap(
                coeff_A_masked,
                in_keys=coeff_A.keys,
                properties=[coeff_A[key].properties for key in coeff_A.keys],
                frame=frame,
                system_id=A,
            )
            coeff_A = coeff_A_unmasked

            # Mask and write ovlp to file, if applicable. No need to store in memory
            if "ovlp" in PROCESS_WHAT:
                ovlp_A = mts.load(join(PROCESSED_DIR(A), "ri_ovlp.npz"))
                ovlp_A_masked = data.mask_ovlp_matrix_tensormap(ovlp_A, idxs_to_keep)
                mts.save(
                    join(
                        PROCESSED_DIR(A),
                        f"ri_ovlp_masked_{MASK['surface_depth']}_{MASK['buffer_depth']}.npz",
                    ),
                    ovlp_A_masked,
                )
                # Also take the diagonal of the masked overlap matrix and save
                diag_A_masked = convert.extract_matrix_diagonal(
                    utils.mts_tensormap_torch_to_core(ovlp_A_masked).to("numpy")
                )
                metatensor.save(
                    join(
                        PROCESSED_DIR(A),
                        f"ri_ovlp_masked_{MASK['surface_depth']}_{MASK['buffer_depth']}_diag.npz",
                    ),
                    utils.make_contiguous_numpy(diag_A_masked)
                )
        coeff.append(coeff_A)

    # Run the RI fitting procedure in AIMS
    aims_kwargs = BASE_AIMS.copy()
    aims_kwargs.update(REBUILD)

    # Load the RI basis set definition - assumes consistent for all RI calcs
    basis_set = utils.unpickle_dict(
        join(PROCESSED_DIR(SYSTEM_ID[0]), "calc_info.pickle")
    )["basis_set"]

    # Remove aims.out files if they exist
    all_aims_outs = [join(REBUILD_DIR(A), "aims.out") for A in SYSTEM_ID]
    for aims_out in all_aims_outs:
        if exists(aims_out):
            os.remove(aims_out)

    # Call rebuild routine
    aims_fields.field_builder(
        system_id=SYSTEM_ID,
        system=SYSTEM,
        coeff=coeff,
        save_dir=REBUILD_DIR,
        return_field=False,
        aims_kwargs=aims_kwargs,
        aims_path=AIMS_PATH,
        basis_set=basis_set,
        cube=CUBE,
        hpc_kwargs=HPC,
        sbatch_kwargs=SBATCH,
    )

    #  Wait for AIMS to finish
    while len(all_aims_outs) > 0:
        for aims_out in all_aims_outs:
            if exists(aims_out):
                with open(aims_out, "r") as f:
                    # Basic check to see if AIMS calc has finished
                    if "Leaving FHI-aims." in f.read():
                        all_aims_outs.remove(aims_out)

    # Evaluate MAE
    for A, frame in zip(SYSTEM_ID, SYSTEM):
        grid = np.loadtxt(join(RI_DIR(A), "partition_tab.out"))
        rho_rebuild = np.loadtxt(join(REBUILD_DIR(A), "rho_rebuilt.out"))

        grid = aims_fields.sort_field_by_grid_points(grid)
        rho_rebuild = aims_fields.sort_field_by_grid_points(rho_rebuild)

        if MASK is not None:  # set non-surface grid points to zero
            # Assumes surface of interest is at and below z=0
            surface_mask = grid[:, 2] >= -MASK["surface_depth"]
            grid = grid[surface_mask]
            rho_rebuild = rho_rebuild[surface_mask]

        tmp_mae = {}
        for ref_name, rho_ref in zip(
            ["ri", "dft"], 
            [
                np.loadtxt(join(RI_DIR(A), f"rho_ri.out")), 
                np.loadtxt(join(RI_DIR(A), f"rho_ref.out"))
            ]
        ):
            rho_ref = aims_fields.sort_field_by_grid_points(rho_ref)
            if MASK is not None:  
                rho_ref = rho_ref[surface_mask]

            # Get the MAE and normalization
            mae, norm = aims_fields.get_mae_between_fields(
                input=rho_rebuild,
                target=rho_ref,
                grid=grid,
            )
            tmp_mae[ref_name] = {"mae": mae, "norm": norm}

        np.savez(
            join(REBUILD_DIR(A), "mae.npz"), 
            mae_ri=tmp_mae["ri"]["mae"], 
            norm_ri=tmp_mae["ri"]["norm"],
            mae_dft=tmp_mae["dft"]["mae"], 
            norm_dft=tmp_mae["dft"]["norm"],
        )


def stm(dft_settings: dict):
    """
    Generates STM images, in either constant-current ("ccm") or constant-height ("chm")
    mode.
    """
    set_settings_gloablly(dft_settings)

    for A in SYSTEM_ID:
        paths = [
            # join(RI_DIR(A), "rho_ref.cube"),
            join(RI_DIR(A), "rho_ri.cube"),
            join(REBUILD_DIR(A), "rho_rebuilt.cube"),
        ]
        # if FIELD_NAME == "edensity":
        #     paths = paths[1:]  # TODO: no rho_ref.cube for edensity - fix!

        # Create a scatter matrix
        fig, axes = plt.subplots(
            len(paths),
            len(paths),
            figsize=(5 * len(paths), 5 * len(paths)),
            sharey=True,
            sharex=True,
        )

        X, Y, Z = [], [], []
        for path in paths:
            q = rho_cube.RhoCube(path)

            if STM["mode"] == "chm":
                x, y, z = q.get_slab_slice(
                    axis=2,
                    center_coord=q.ase_frame.positions[:, 2].max()
                    + STM["center_coord"],
                    thickness=STM["thickness"],
                )

            elif STM["mode"] == "ccm":
                x, y, z = q.get_height_profile_map(
                    isovalue=STM["isovalue"],
                    tolerance=STM["tolerance"],
                    grid_multiplier=STM.get("grid_multiplier"),
                    z_min=STM.get("z_min"),
                    z_max=STM.get("z_max"),
                    xy_tiling=STM.get("xy_tiling"),
                )
            else:
                raise ValueError("Invalid STM mode")
            X.append(x)
            Y.append(y)
            Z.append(z)

        for row, row_ax in enumerate(axes):
            for col, ax in enumerate(row_ax):
                if row == col:
                    x, y, z = X[row], Y[row], Z[row]
                elif row < col:
                    x, y, z = X[row], Y[col], Z[row] - Z[col]
                else:
                    continue
                cs = ax.contourf(x, y, z, cmap="viridis", levels=STM.get("levels"))
                fig.colorbar(cs)
                ax.set_aspect("equal")
                ax.set_xlabel("x / Ang")
                ax.set_ylabel("y / Ang")
                if row == col:
                    ax.set_title(f"{paths[row][paths[row].rfind('rho'):]}")
                else:
                    ax.set_title(
                        f"{paths[row][paths[row].rfind('rho'):]} vs {paths[col][paths[col].rfind('rho'):]}"
                    )

        plt.savefig(
            join(
                REBUILD_DIR(A),
                f"stm_{STM['mode']}_scatter.png",
            )
        )
        utils.pickle_dict(join(REBUILD_DIR(A), "stm_settings.pickle"), STM)
