"""
Module for setting up and running FHI-AIMS calculations.
"""
import os
from typing import Optional

import ase
import ase.io
import ase.io.aims as aims_io


def write_input_files(
    atoms: ase.Atoms,
    run_dir: str,
    aims_kwargs: dict,
):
    """
    Writes input files geometry.in, control.in, and run-aims.sh to `aims_dir`
    """
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    # Write inputs
    aims_io.write_aims(os.path.join(run_dir, "geometry.in"), atoms=atoms)
    aims_io.write_control(os.path.join(run_dir, "control.in"), atoms=atoms, parameters=aims_kwargs)

    return


def write_aims_sbatch(fname: str, aims: str, load_modules: Optional[list] = None, **kwargs):
    """
    Writes a bash script to `fname` that allows running of FHI-AIMS with
    specification of sbatch parameters via `kwargs`.
    """
    with open(fname, "wt") as f:
        # Write the header
        f.write("#!/bin/bash\n")

        # Write the sbatch parameters
        for tag, val in kwargs.items():
            f.write(f"#SBATCH --{tag}={val}\n")
        f.write("\n\n")
        # Load modules
        if load_modules is not None:
            f.write("# Load modules, set env variables, increase stack size\n")
            f.write(f"module load {' '.join(load_modules)}\n\n")

        # Some environment varibales that need to be set
        f.write("export OMP_NUM_THREADS=1\n")
        f.write("export MKL_DYNAMIC=FALSE\n")
        f.write("export MKL_NUM_THREADS=1\n\n")

        # Increase stack size to unlim
        f.write("ulimit -s unlimited\n")
        f.write("\n")

        # Write the run AIMS command
        f.write("# Run AIMS\n")
        f.write(f"AIMS={aims}\n")
        f.write("srun $AIMS < control.in > aims.out\n")

    return


def run_aims_in_dir(run_dir: str):
    """
    Runs FHI-AIMS in the directory `run_dir`. 
    
    Files "geometry.in", "control.in", and "run-aims.sh" must be present in
    `run_dir`.
    """
    for file in ["geometry.in", "control.in", "run-aims.sh"]:
        if not os.path.exists(os.path.join(run_dir, file)):
            raise ValueError(f"File {file} does not exist in {run_dir}")
    # Enter the run dir, run the script, and return to the original dir
    curr_dir = os.getcwd()
    os.chdir(path=run_dir)
    os.system("sbatch run-aims.sh")
    os.chdir(path=curr_dir)

    return