from functools import partial
from os.path import exists, join
from typing import List
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from rhocalc.aims import aims_fields, aims_predictor
from rhocalc.cube import rho_cube
from rholearn import utils


def set_settings_gloablly(dft_settings: dict, ml_settings: dict):
    """Sets the DFT and ML settings globally."""
    global_vars = globals()
    for d in [dft_settings, ml_settings]:
        for key, value in d.items():
            global_vars[key] = value


def eval(dft_settings: dict, ml_settings: dict):
    """
    Runs model evaluation in the following steps:
        1. Load model
        2. Perform inference
        3. Rebuild fields by calling FHI-aims
        4. Evaluate MAE against reference fields
        5. Generate STM images
    """

    # ===== Setup =====
    t0_eval = time.time()
    set_settings_gloablly(dft_settings, ml_settings)
    torch.manual_seed(SEED)
    torch.set_default_dtype(DTYPE)

    log_path = join(ML_DIR, "eval.log")
    utils.log(log_path, f"===== BEGIN =====")
    utils.log(log_path, f"Working directory: {ML_DIR}")

    # ===== Model inference =====
    utils.log(log_path, f"Load model from checkpoint at epoch {EVAL['eval_epoch']}")
    eval_frames = [ALL_SYSTEM[A] for A in EVAL["eval_id"]]
    model = torch.load(join(CHKPT_DIR(EVAL["eval_epoch"]), "model.pt"))
    utils.log(log_path, f"Perform inference")
    eval_preds = model.predict(frames=eval_frames, system_id=EVAL["eval_id"])

    # ===== Rebuild fields =====
    utils.log(log_path, f"Rebuild fields")
    aims_kwargs = {k: v for k, v in BASE_AIMS.items()}
    aims_kwargs.update({k: v for k, v in REBUILD.items()})
    aims_predictor.field_builder(
        system_id=EVAL["eval_id"],
        system=eval_frames,
        predicted_coeffs=eval_preds,
        save_dir=partial(EVAL_DIR, epoch=EVAL["eval_epoch"]),
        return_field=False,
        aims_kwargs=aims_kwargs,
        aims_path=AIMS_PATH,
        basis_set=TARGET_BASIS,
        cube=CUBE,
        hpc_kwargs=HPC,
        sbatch_kwargs=SBATCH,
    )

    # ===== Wait for AIMS to finish =====
    utils.log(log_path, f"Waiting for FHI-aims rebuild to finish...")
    all_aims_outs = [
        join(EVAL_DIR(A, epoch=EVAL["eval_epoch"]), "aims.out") for A in EVAL["eval_id"]
    ]
    while len(all_aims_outs) > 0:
        for aims_out in all_aims_outs:
            if exists(aims_out):
                with open(aims_out, "r") as f:
                    # Basic check to see if AIMS calc has finished
                    if "Leaving FHI-aims." in f.read():
                        all_aims_outs.remove(aims_out)

    # ===== Evaluate MAE =====
    utils.log(log_path, f"Evaluate MAE versus reference field type: {EVAL['target_type']}")
    for A, frame in zip(EVAL["eval_id"], eval_frames):
        grid = np.loadtxt(join(RI_DIR(A), "partition_tab.out"))
        rho_ref = np.loadtxt(join(RI_DIR(A), f"rho_{EVAL['target_type']}.out"))
        rho_ml = np.loadtxt(
            join(EVAL_DIR(A, epoch=EVAL["eval_epoch"]), "rho_rebuilt.out")
        )

        grid = aims_fields.sort_field_by_grid_points(grid)
        rho_ref = aims_fields.sort_field_by_grid_points(rho_ref)
        rho_ml = aims_fields.sort_field_by_grid_points(rho_ml)

        if EVAL["evaluate_on"] == "surface":
            # Set non-surface grid points to zero
            grid[:, 3] *= grid[:, 2] > frame.positions[:, 2].max() - (
                DESCRIPTOR_HYPERS["slab_depth"]
            )

        mae = aims_fields.get_percent_mae_between_fields(
            input=rho_ml, target=rho_ref, grid=grid
        )
        utils.log(log_path, f"system {A} percent_mae {mae:.2f}")

    # ===== Generate STM =====
    if EVAL.get("stm") is not None:
        for A in EVAL["eval_id"]:
            utils.log(log_path, f"Generating STM image for system {A}")
            paths = [
                # join(RI_DIR(A), "rho_ref.cube"),
                join(RI_DIR(A), "rho_ri.cube"),
                join(EVAL_DIR(A, epoch=EVAL["eval_epoch"]), "rho_rebuilt.cube"),
            ]
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
                x, y, z = q.get_slab_slice(
                    axis=2,
                    center_coord=q.ase_frame.positions[:, 2].max()
                    + EVAL["stm"]["center_coord"],
                    thickness=EVAL["stm"]["thickness"],
                )
                X.append(x)
                Y.append(y)
                Z.append(z)

            for row, row_ax in enumerate(axes):
                for col, ax in enumerate(row_ax):
                    if row == col:
                        x, y, z = X[row], Y[row], np.tanh(Z[row])
                    elif row < col:
                        x, y, z = X[row], Y[col], np.tanh(Z[row] - Z[col])
                    else:
                        continue
                    cs = ax.contourf(x, y, z, cmap="gray")
                    cbar = fig.colorbar(cs)

            plt.savefig(join(EVAL_DIR(A, epoch=EVAL["eval_epoch"]), "stm_scatter.png"))

    # ===== Finish =====
    dt_eval = time.time() - t0_eval
    if dt_eval <= 60:
        utils.log(log_path, f"Evaluation complete in {dt_eval} seconds.")
    elif 60 < dt_eval <= 3600:
        utils.log(log_path, f"Evaluation complete in {dt_eval / 60} minutes.")
    else:
        utils.log(log_path, f"Evaluation complete in {dt_eval / 3600} hours.")
