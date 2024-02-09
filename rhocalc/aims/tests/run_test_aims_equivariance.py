"""
For generating data to test SO3 equivariance
"""
import glob
import time
import os
from functools import partial
import shutil
import sys

import ase
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import chemiscope

from rhocalc.aims import aims_calc, aims_fields, aims_parser
from rholearn import io, rotations


transf = sys.argv[1]
molecule = sys.argv[2]

if transf not in ["so3", "o3"]:
    raise ValueError('Must pass one of ["so3", "o3"]')

if molecule not in ["CO", "H2O"]:
    raise ValueError('Must pass one of ["CO", "H2O"]')


# ====================================================
# ===== Settings for generating learning targets =====
# ====================================================

# Define the top level dir
TOP_DIR = os.getcwd()

# Where the generated data should be written
DATA_DIR = os.path.join(TOP_DIR, f"equivariance_{transf}_{molecule}")

log_path = os.path.join(DATA_DIR, "test.log")

# Create data dir
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

os.chdir(DATA_DIR)

DATA_SETTINGS = {

    # Read in all frames in complete dataset
    "all_frames": os.path.join(DATA_DIR, f"frames_{transf}.xyz"),
    
    # Define a subset of frames
    "n_frames": 5,

    # Define the index of the RI calculation
    # "ri_restart_idx": "ildos_+1.5V",
    "ri_restart_idx": "edensity",

    # Name the field - special cases: "edensity", "ldos", "ildos", "homo", "lumo".
    "field_name": "edensity",

    # Global speices required
    "global_species": [1],

    # Define a random seed - None for no shuffling of idxs
    "seed": None,

    # Calculate the standard deviation?
    # i.e. to calc it for output training data:
    "calc_out_train_std_dev": ("output", "train"),
}

# Path to AIMS binary
AIMS_PATH = "/home/abbott/codes/new_aims/FHIaims/build/aims.230905.scalapack.mpi.x"

# Define the AIMS settings that are common to all calculations
BASE_AIMS_KWARGS = {
    "species_dir": "/home/abbott/rho/rho_learn/rhocalc/aims/aims_species/tight/default",
    "xc": "pbe0",
    "spin": "none",
    "charge": 0,
    "sc_accuracy_rho": 1e-8,
    "wave_threshold": 1e-8,
}

# Settings specific to SCF
SCF_KWARGS = {
    "elsi_restart": "write 1",
    "ri_fit_write_orbital_info": True,
}

# Settings for the RI procedure
RI_KWARGS = {
    # ===== To restart from a converged density matrix and force no SCF:
    "elsi_restart": "read",
    "sc_iter_limit": 0,
    "postprocess_anyway": True,
    # ===== What we want to fit to:
    "ri_fit_field_from_kso_weights": True,  # build custom scalar field
    # ===== Specific setting for RI fitting
    "ri_fit_ovlp_cutoff_radius": 1.0,
    "ri_fit_assume_converged": True,
    "default_max_l_prodbas": 5,
    # "default_max_n_prodbas": 6,  # currently doesn't work in FHI-aims
    # ===== What to write as output
    "ri_fit_write_coeffs": True,  # RI coeffs (the learning target)
    "ri_fit_write_ovlp": True,  # RI overlap (needed for loss evaluation)
    "ri_fit_write_ref_field": True,  # SCF converged scalar field on AIMS grid
    "ri_fit_write_rebuilt_field": True,  # RI-rebuilt scalar field on AIMS grid
    "ri_fit_write_ref_field_cube": True,  # SCF converged scalar field on CUBE grid
    "ri_fit_write_rebuilt_field_cube": True,  # RI-rebuilt scalar field on CUBE grid
    "output": ["cube ri_fit"],  # Allows output of cube files
}

# Settings for the RI rebuild procedure
REBUILD_KWARGS = {
    # ===== Force no SCF
    "sc_iter_limit": 0,
    "postprocess_anyway": True,
    "ri_fit_assume_converged": True,
    # ===== What we want to do
    "ri_fit_rebuild_from_coeffs": True,
    # ===== Specific settings for RI rebuild
    "ri_fit_ovlp_cutoff_radius": RI_KWARGS["ri_fit_ovlp_cutoff_radius"],
    "default_max_l_prodbas": RI_KWARGS["default_max_l_prodbas"],
    # ===== What to write as output
    "ri_fit_write_rebuilt_field": True,
    "ri_fit_write_rebuilt_field_cube": True,
    # ===== Controlling cube file output
    "output": ["cube ri_fit"],  # IMPORTANT! Needed for cube files
}

CUBE_KWARGS = {
    "slab": False,
    "n_points": (100, 100, 100),  # number of cube edge points
}

# Settings for HPC job scheduler
SBATCH_KWARGS = {
    "job-name": "debug_equiv",
    "nodes": 1,
    "time": "00:20:00",
    "mem-per-cpu": 0,
    "partition": "standard",
    "ntasks-per-node": 10,
}

# ===============================
# ===== Generate Structures =====
# ===============================

with open(log_path, "w") as f:
    f.write(f"Top directory defined as: {TOP_DIR}")

# bond_length = 0.74  # h2
bond_length = 1.128  # co

if molecule == "CO":
    mol0 = ase.Atoms(symbols=["C", "O"], positions=[[0, 0, 0], [0, 0, bond_length]])
elif molecule == "H2O":
    mol0 = ase.Atoms(
        symbols=["O", "H", "H"],
        positions=[
            [0.06633400,   0.00000000,  0.00370100 ],
            [-0.52638300, -0.76932700, -0.02936600 ],
            [-0.52638300,  0.76932700, -0.02936600 ],
        ],
    )
else:
    raise


angles = [
    np.array([0, np.pi / 2, np.pi / 2]),
    np.array([np.pi / 2, np.pi / 2, 0]),
    np.array([1.98738864, 0.3641072 , 0.97489991]),
    np.array([0.98738864, 0.8641072 , -1.97489991]),
]

frames = [mol0]
for angs in angles:
    if transf == "so3":
        frame_rot = rotations.transform_frame_so3(mol0, angs)
    elif transf == "o3":
        frame_rot = rotations.transform_frame_o3(mol0, angs)
    else:
        raise
    frame_rot.info["angles"] = angs
    frames.append(frame_rot)

# Shuffle the total set of structure indices
idxs = np.arange(len(frames))

ase.io.write(os.path.join(DATA_DIR, f"frames_{transf}.xyz"), frames)
    
# A callable that takes structure idx as an argument, returns path to AIMS SCF
# output data
def scf_dir(A):
    return os.path.join(DATA_DIR, f"{A}")

# A callable that takes structure idx as an argument, returns path to AIMS RI
# output data
def ri_dir(A, restart_idx):
    return os.path.join(scf_dir(A), f"{restart_idx}")

# A callable that takes structure idx as an argument, returns path to processed
# data (i.e. metatensor-format)
def processed_dir(A, restart_idx):
    return os.path.join(ri_dir(A, restart_idx), "processed")


# Build a dict of settings for each calculation (i.e. structure)
# IMPORTANT: zip() is used to pair up the structure index and the structure
calcs = {
    A: {"atoms": frame, "run_dir": scf_dir(A)} for A, frame in zip(idxs, frames)
}

# ===================
# ===== Run SCF =====
# ===================

# And the general settings for all calcs
aims_kwargs = BASE_AIMS_KWARGS.copy()
aims_kwargs.update(SCF_KWARGS)

# Define paths to the aims.out files for RI calcs
all_aims_outs = [os.path.join(scf_dir(A), "aims.out") for A in idxs]
for aims_out in all_aims_outs:
    if os.path.exists(aims_out):
        os.remove(aims_out)

# Run the SCF in AIMS
aims_calc.run_aims_array(
    calcs=calcs,
    aims_path=AIMS_PATH,
    aims_kwargs=aims_kwargs,
    sbatch_kwargs=SBATCH_KWARGS,
    run_dir=scf_dir,
)

# Wait until all AIMS calcs have finished
all_finished = False
while len(all_aims_outs) > 0:
    for aims_out in all_aims_outs:
        if os.path.exists(aims_out):
            with open(aims_out, "r") as f:
                # Basic check to see if AIMS calc has finished
                if "Leaving FHI-aims." in f.read():
                    all_aims_outs.remove(aims_out)

# ======================
# ===== RI Fitting =====
# ======================

for A, frame in zip(idxs, frames):
    # Make dir for the RI calculation
    if not os.path.exists(ri_dir(A, DATA_SETTINGS["ri_restart_idx"])):
        os.makedirs(ri_dir(A, DATA_SETTINGS["ri_restart_idx"]))

    # Copy density matrix restart
    for density_matrix in glob.glob(os.path.join(scf_dir(A), "D*.csc")):
        shutil.copy(density_matrix, ri_dir(A, DATA_SETTINGS["ri_restart_idx"]))

    # Specify tailored cube edges
    if RI_KWARGS.get("output") == ["cube ri_fit"]:
        if CUBE_KWARGS.get("slab") is True:
            calcs[A]["aims_kwargs"] = aims_calc.get_aims_cube_edges_slab(
                frame, CUBE_KWARGS.get("n_points")
            )
        else:
            calcs[A]["aims_kwargs"] = aims_calc.get_aims_cube_edges(
                frame, CUBE_KWARGS.get("n_points")
            )

    # Define KSO weights and write to file
    kso_weights = aims_fields.get_kso_weight_vector_for_named_field(
        field_name=DATA_SETTINGS["field_name"],
        kso_info_path=os.path.join(scf_dir(A), "ks_orbital_info.out"),
    )
    np.savetxt(
        os.path.join(
            ri_dir(A, DATA_SETTINGS["ri_restart_idx"]), 
            "ks_orbital_weights.in"
        ), 
        kso_weights
    )

# And the general settings for all calcs
aims_kwargs = BASE_AIMS_KWARGS.copy()
aims_kwargs.update(RI_KWARGS)

# Define paths to the aims.out files for RI calcs
all_aims_outs = [os.path.join(ri_dir(A, DATA_SETTINGS["ri_restart_idx"]), "aims.out") for A in idxs]
for aims_out in all_aims_outs:
    if os.path.exists(aims_out):
        os.remove(aims_out)

# Run the RI fitting procedure in AIMS
aims_calc.run_aims_array(
    calcs=calcs,
    aims_path=AIMS_PATH,
    aims_kwargs=aims_kwargs,
    sbatch_kwargs=SBATCH_KWARGS,
    run_dir=partial(ri_dir, restart_idx=DATA_SETTINGS["ri_restart_idx"]),
)

# Wait until all AIMS calcs have finished
all_finished = False
while len(all_aims_outs) > 0:
    for aims_out in all_aims_outs:
        if os.path.exists(aims_out):
            with open(aims_out, "r") as f:
                # Basic check to see if AIMS calc has finished
                if "Leaving FHI-aims." in f.read():
                    all_aims_outs.remove(aims_out)

# Remove the density matrix restart files
for A in idxs:
    for density_matrix in glob.glob(os.path.join(ri_dir(A, DATA_SETTINGS["ri_restart_idx"]), "D*.csc")):
        os.remove(density_matrix)

# Process to metatensor
aims_calc.process_aims_results_sbatch_array(
    "run-process-aims.sh",
    structure_idxs=idxs,
    run_dir=partial(ri_dir, restart_idx=DATA_SETTINGS["ri_restart_idx"]),
    process_what=["coeffs", "ovlp"],
    **SBATCH_KWARGS,
)

# ==============================
# ===== Check Equivariance =====
# ==============================

# Get the basis set definition
basis_set_exists = False
while not basis_set_exists:
    try:
        basis_set = io.unpickle_dict(
            os.path.join(ri_dir(0, DATA_SETTINGS["ri_restart_idx"]), "processed", "calc_info.pickle")
        )["basis_set"]
    except FileNotFoundError:
        time.sleep(3)
        continue
    basis_set_exists = True

lmax = max(basis_set["lmax"].values())
angles = [np.array([2 * np.pi, 2 * np.pi, 2 * np.pi])] + [frame.info["angles"] for frame in frames[1:]]

# Build the Wigner-D matrices for each rotation
wigner_d_matrices = [rotations.WignerDReal(lmax=lmax, angles=angs) for angs in angles]

# Get the coefficient vector for each frame
coeffs = [np.loadtxt(os.path.join(ri_dir(A, DATA_SETTINGS["ri_restart_idx"]), "ri_coeffs.out")) for A in idxs]

# Rotate the unrotated coefficient vector by each wigner-d matrix
if transf == "so3":
    coeffs_unrot_rot = [
        wigner_d.transform_flat_vector_so3(frames[0], coeffs[0], **basis_set)
        for wigner_d in wigner_d_matrices
    ]
else:
    coeffs_unrot_rot = [
        wigner_d.transform_flat_vector_o3(frames[0], coeffs[0], **basis_set)
        for wigner_d in wigner_d_matrices
    ]

# Parity plots
with open(log_path, "a") as f:
    f.write("MSE in rotated coefficients vs frame[0]")

fig, axes = plt.subplots(1, len(angles), figsize=(20, 10))
for A, c_rot, c_unrot_rot, ax in zip(idxs, coeffs, coeffs_unrot_rot, axes):
    mse = np.mean(np.abs(c_rot - c_unrot_rot) ** 1)
    with open(log_path, "a") as f:
        f.write(f"   Structure {A}: {mse}")
    ax.scatter(c_rot, c_unrot_rot)
    ax.set_xlabel("c(RA)")
    ax.set_ylabel("D(R) c(A)")
    ax.axline((0, 0), slope=1, color="black", linestyle="--")
    ax.set_aspect("equal")
    ax.set_title(f"MSE: {mse}")

plt.savefig(os.path.join(DATA_DIR, "parity_plots.png"))


# ======================
# ===== RI Rebuild =====
# ======================

# Try a rebuild of the density from the target RI coefficients
ri_rebuild_idx = os.path.join(DATA_SETTINGS["ri_restart_idx"], "rebuild")

# And the general settings for all calcs
aims_kwargs = BASE_AIMS_KWARGS.copy()
aims_kwargs.update(REBUILD_KWARGS)

# Define paths to the aims.out files for RI calcs
all_aims_outs = [os.path.join(ri_dir(A, ri_rebuild_idx), "aims.out") for A in idxs]
for aims_out in all_aims_outs:
    if os.path.exists(aims_out):
        os.remove(aims_out)

for A, c_unrot_rot in zip(idxs, coeffs_unrot_rot):
    if not os.path.exists(ri_dir(A, ri_rebuild_idx)):
        os.makedirs(ri_dir(A, ri_rebuild_idx))

    # Write the rotated coefficients of the unrotated structure as the rebuild
    # coefficients
    np.savetxt(
        os.path.join(ri_dir(A, ri_rebuild_idx), "ri_coeffs.in"),
        c_unrot_rot,
    )

# Run the RI fitting procedure in AIMS
aims_calc.run_aims_array(
    calcs=calcs,
    aims_path=AIMS_PATH,
    aims_kwargs=aims_kwargs,
    sbatch_kwargs=SBATCH_KWARGS,
    run_dir=partial(ri_dir, restart_idx=ri_rebuild_idx),
)

# Wait until all AIMS calcs have finished
all_finished = False
while len(all_aims_outs) > 0:
    for aims_out in all_aims_outs:
        if os.path.exists(aims_out):
            with open(aims_out, "a") as f:
                # Basic check to see if AIMS calc has finished
                if "Leaving FHI-aims." in f.read():
                    all_aims_outs.remove(aims_out)


# ============================================
# ===== Calculate error on rebuilt field =====
# ============================================

with open(log_path, "a") as f:
    f.write("% MAE in rebuilt field vs frame[0]")
for A in idxs:

    # Load the fields and grid
    rho_rebuilt = np.loadtxt(
        os.path.join(ri_dir(A, ri_rebuild_idx), "rho_rebuilt.out")
    )
    rho_ref = np.loadtxt(
        os.path.join(ri_dir(A, DATA_SETTINGS["ri_restart_idx"]), "rho_ref.out")
    )
    rho_ri = np.loadtxt(
        os.path.join(ri_dir(A, DATA_SETTINGS["ri_restart_idx"]), "rho_ri.out")
    )
    grid = np.loadtxt(
        os.path.join(ri_dir(A, DATA_SETTINGS["ri_restart_idx"]), "partition_tab.out")
    )

    # Calculate the error
    percent_mae_ref = aims_parser.get_percent_mae_between_fields(
        input=rho_rebuilt, target=rho_ref, grid=grid
    )
    percent_mae_ri = aims_parser.get_percent_mae_between_fields(
        input=rho_rebuilt, target=rho_ri, grid=grid
    )

    with open(log_path, "a") as f:
        f.write(f"   Structure {A}")
        f.write(f"      vs rho_ref: {percent_mae_ref}")
        f.write(f"      vs rho_ri: {percent_mae_ri}")

