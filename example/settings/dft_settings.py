"""
Module containing global varibales for running DFT calculations, more specifically
running SCF, RI, and density rebuild procedures.
"""

import os
from os.path import exists, join
import ase.io
import numpy as np

# ===== SETUP =====
SEED = 42
DATA_DIR = "/work/cosmo/abbott/april-24/si_masked/data"
FIELD_NAME = "ildos"
RI_FIT_ID = "ildos-1V"

# ===== DATA =====
ALL_SYSTEM = ase.io.read(
    join(DATA_DIR, "si_slabs_dimer_z_depth_geomopt_distorted.xyz"), ":"
)
ALL_SYSTEM_ID = np.arange(len(ALL_SYSTEM))
# np.random.default_rng(seed=SEED).shuffle(ALL_SYSTEM_ID)  # shuffle first?
SYSTEM_ID = ALL_SYSTEM_ID[6::13]
SYSTEM = [ALL_SYSTEM[A] for A in SYSTEM_ID]


# ===== HPC
SBATCH = {
    "job-name": "si_mask",
    "nodes": 1,
    "time": "8:00:00",
    "mem-per-cpu": 0,
    "partition": "standard",
    "ntasks-per-node": 72,
}
HPC = {
    "load_modules": ["intel", "intel-oneapi-mpi", "intel-oneapi-mkl"],
    "export_vars": [
        "OMP_NUM_THREADS=1",
        "MKL_DYNAMIC=FALSE",
        "MKL_NUM_THREADS=1",
        "I_MPI_FABRICS=shm",
    ],
    "run_command": "srun",
}

# ===== BASE AIMS =====
AIMS_PATH = "/home/abbott/codes/new_aims/FHIaims/build/aims.230905.scalapack.mpi.x"
BASE_AIMS = {
    "species_dir": "/home/abbott/codes/rho_learn/rhocalc/aims/aims_species/tight/default",
    "xc": "pbe",
    "spin": "none",
    # "relativistic": ["atomic_zora scalar"],
    "charge": 0,
    "sc_accuracy_rho": 1e-8,
    "wave_threshold": 1e-8,
    "k_grid_density": 12,
}

# ===== SCF =====
SCF = {
    "elsi_restart": "read",
    "ri_fit_write_orbital_info": True,
    # "evaluate_work_function": True,
    "output": [
        "atom_proj_dos -30 5 1000 0.3",
        "cube hartree_potential",
    ],  # atom_proj_dos Estart Eend n_points broadening
    # Geometry Optimization. Use "light" for pre-relaxation.
    # "relax_geometry": ["bfgs 0.01"],
    # "species_dir": "/home/abbott/march-24/rho_learn/rhocalc/aims/aims_species/light/default",  # jed
}

# ===== RI =====
RI = {
    # ===== To restart from a converged density matrix and force no SCF:
    "elsi_restart": "read",
    "sc_iter_limit": 0,
    "postprocess_anyway": True,
    # ===== What we want to fit to:
    "ri_fit_field_from_kso_weights": True,  # build custom scalar field
    # "ri_fit_total_density": True,
    # ===== Specific setting for RI fitting
    "ri_fit_ovlp_cutoff_radius": 1.5,
    "ri_fit_assume_converged": True,
    "default_max_l_prodbas": 3,
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

# ===== REBUILD =====
REBUILD = {
    # ===== Force no SCF
    "sc_iter_limit": 0,
    "postprocess_anyway": True,
    "ri_fit_assume_converged": True,
    # ===== What we want to do
    "ri_fit_rebuild_from_coeffs": True,
    # ===== Specific settings for RI rebuild
    "ri_fit_ovlp_cutoff_radius": RI["ri_fit_ovlp_cutoff_radius"],
    "default_max_l_prodbas": RI["default_max_l_prodbas"],
    # ===== What to write as output
    "ri_fit_write_rebuilt_field": True,
    "ri_fit_write_rebuilt_field_cube": True,
    # ===== Controlling cube file output
    "output": ["cube ri_fit"],  # IMPORTANT! Needed for cube files
}

# ===== FIELD AND CUBE =====
LDOS = {
    "target_energy": "fermi_integrated_eV",  # "fermi_integrated_eV", "vbm_eV", "fermi_eV"
    "gaussian_width": 0.3,  # eV
    "biasing_voltage": -1.0,  # V, for the integrated LDOS
    "method": "gaussian_analytical",
}
CUBE = {
    "slab": True,
    "n_points": (100, 100, 100),  # number of cube edge points
}

# ===== RELATIVE DIRS =====
SCF_DIR = lambda A: join(DATA_DIR, f"{A}")
RI_DIR = lambda A: join(SCF_DIR(A), RI_FIT_ID)
REBUILD_DIR = lambda A: join(RI_DIR(A), "rebuild")
PROCESSED_DIR = lambda A: join(RI_DIR(A), "processed")

# ===== CREATE DIRS =====
if not exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ===== FINAL DICT =====
DFT_SETTINGS = {
    "SEED": SEED,
    "DATA_DIR": DATA_DIR,
    "FIELD_NAME": FIELD_NAME,
    "RI_FIT_ID": RI_FIT_ID,
    "ALL_SYSTEM": ALL_SYSTEM,
    "ALL_SYSTEM_ID": ALL_SYSTEM_ID,
    "SYSTEM_ID": SYSTEM_ID,
    "SYSTEM": SYSTEM,
    "SBATCH": SBATCH,
    "HPC": HPC,
    "AIMS_PATH": AIMS_PATH,
    "BASE_AIMS": BASE_AIMS,
    "SCF": SCF,
    "RI": RI,
    "REBUILD": REBUILD,
    "LDOS": LDOS,
    "CUBE": CUBE,
}
