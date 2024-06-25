"""
Module containing global varibales for running DFT calculations, more specifically
running SCF, RI, and density rebuild procedures.
"""

import os
from os.path import exists, join
import ase.io
import numpy as np

# ===== SETUP =====
DATA_DIR = "/work/cosmo/abbott/june-24/ArgAu/data"
FIELD_NAME = "edensity"
RI_FIT_ID = "edensity"
# FIELD_NAME = "ildos"
# RI_FIT_ID = "ildos"

# ===== DATA =====
ALL_SYSTEM = ase.io.read(join(DATA_DIR, "ArgHAu_geomopt_energies_forces.xyz"), ":")
ALL_SYSTEM_ID = np.arange(len(ALL_SYSTEM))
SYSTEM_ID = np.array(list(ALL_SYSTEM_ID[1:100]) + list(ALL_SYSTEM_ID[721:821]))
# SYSTEM_ID = np.array([0])
SYSTEM = [ALL_SYSTEM[A] for A in SYSTEM_ID]

# ===== HPC
SBATCH = {
    "job-name": "ArgAu",
    "nodes": 1,
    "time": "02:00:00",
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
    "species_dir": "/home/abbott/codes/rho_learn/rhocalc/aims/aims_species/light/modified",
    # Physical settings
    "xc": "pbe",
    "spin": "none",
    "charge": 0,
    "relativistic": ["atomic_zora scalar"],
    "k_grid": [4, 4, 1],
    # Computational
    # "collect_eigenvectors": False,
    # "cpu_consistency_threshold": 1e-11,
    # "check_cpu_consistency": False,
    # "empty_states": 10,
}

# ===== SCF =====
SCF = {
    "elsi_restart": "write 1000",  # set to higher number here? Writes upon convergence in all cases.
    "ri_fit_write_orbital_info": True,
    # "evaluate_work_function": True,
    "output": [
        "atom_proj_dos -30 5 1000 0.3",
        # "cube hartree_potential",
    ],  # atom_proj_dos Estart Eend n_points broadening
    # Geometry Optimization. Use "light" for pre-relaxation.
    # "relax_geometry": ["bfgs 0.01"],
    # "species_dir": "/home/abbott/march-24/rho_learn/rhocalc/aims/aims_species/light/default",  # jed
    # SC accuracy
    "sc_accuracy_rho": 2e-4,
    "sc_accuracy_eev": 2e-3,
    "sc_accuracy_etot": 1e-6,
    # "sc_accuracy_forces": 2e-3,
    "sc_iter_limit": 10000,
    # Dispersion and dipoles
    "vdw_correction_hirshfeld": True,
    "vdw_pair_ignore": ["Au Au"],
    # params for vdW-surf for Au !!! species sub-tag needed?
    # "hirshfeld_param": [134, 15.6, 2.91],
    "use_dipole_correction": True,
    # Mixing
    "mixer": "pulay",
    "n_max_pulay": 10,
    "charge_mix_param": 0.1,
    "hartree_convergence_parameter": 5,
    "occupation_type gaussian": 0.1,
}

# ===== RI =====
RI = {
    # ===== To restart from a converged density matrix and force no SCF:
    "elsi_restart": "read",
    "sc_iter_limit": 0,
    "postprocess_anyway": True,
    # ===== What we want to fit to:
    # "ri_fit_field_from_kso_weights": True,  # build custom scalar field
    "ri_fit_total_density": True,
    # ===== Specific setting for RI fitting
    "ri_fit_ovlp_cutoff_radius": 1.5,
    "ri_fit_assume_converged": True,
    "default_max_l_prodbas": 1,
    # "default_max_n_prodbas": 3,
    # ===== What to write as output
    "ri_fit_write_coeffs": True,  # RI coeffs (the learning target)
    "ri_fit_write_ovlp": True,  # RI overlap (needed for loss evaluation)
    "ri_fit_write_ref_field": True,  # SCF converged scalar field on AIMS grid
    "ri_fit_write_rebuilt_field": True,  # RI-rebuilt scalar field on AIMS grid
    "ri_fit_write_ref_field_cube": True,  # SCF converged scalar field on CUBE grid
    "ri_fit_write_rebuilt_field_cube": True,  # RI-rebuilt scalar field on CUBE grid
    "output": ["cube total_density", "cube ri_fit"],  # Allows output of cube files
}
PROCESS_WHAT = []
if "ri_fit_write_coeffs" in RI:
    PROCESS_WHAT.append("coeffs")
if "ri_fit_write_ovlp" in RI:
    PROCESS_WHAT.append("ovlp")

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
    "target_energy": "fermi_eV",  # "fermi_integrated_eV", "vbm_eV", "fermi_eV"
    "gaussian_width": 0.01,  # eV
    "biasing_voltage": 0.01,  # V, for the integrated LDOS
    "method": "gaussian_analytical",
}
# MASK = None
MASK = {
    "surface_depth": 2.0,  # Ang
    "buffer_depth": 2.0,  # Ang
}
CUBE = {
    "slab": True,
    "n_points": (200, 200, 100),  # number of cube edge points
    # "z_min": - MASK["surface_depth"] - MASK["buffer_depth"],  # Ang
    # "z_max": None,
}
STM = {
    "mode": "ccm",
    "isovalue": 0.01,
    "tolerance": 1e-3,
    "grid_multiplier": 20,
    "z_min": -1,
    "z_max": None,
    "xy_tiling": [1, 1],
    "levels": 50,
}

# ===== RELATIVE DIRS =====
SCF_DIR = lambda A: join(DATA_DIR, f"{A}", "scf")
RI_DIR = lambda A: join(DATA_DIR, f"{A}", "ri", RI_FIT_ID)
REBUILD_DIR = lambda A: join(
    RI_DIR(A),
    "rebuild",
    (
        "unmasked"
        if MASK is None
        else f"masked_{MASK['surface_depth']}_{MASK['buffer_depth']}"
    ),
)
PROCESSED_DIR = lambda A: join(RI_DIR(A), "processed")

# ===== CREATE DIRS =====
if not exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ===== FINAL DICT =====
DFT_SETTINGS = {
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
    "PROCESS_WHAT": PROCESS_WHAT,
    "REBUILD": REBUILD,
    "LDOS": LDOS,
    "MASK": MASK,
    "CUBE": CUBE,
    "STM": STM,
    "SCF_DIR": SCF_DIR,
    "RI_DIR": RI_DIR,
    "REBUILD_DIR": REBUILD_DIR,
    "PROCESSED_DIR": PROCESSED_DIR,
}
