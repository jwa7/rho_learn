"""
Module for setting up and running FHI-AIMS calculations.
"""
import inspect
import glob
import os
import shutil
from typing import Optional, List

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


def run_aims(
    aims_kwargs: dict, 
    sbatch_kwargs: dict, 
    calcs: dict, 
    restart_idx: Optional[int] = None, 
    copy_files: Optional[List[str]] = None
):
    """
    Runs an AIMS calculation for each of the calculations in `calcs`.

    The `aims_)kwargs` are used to write 
    
    RI calculation for the total electron density and each KS-orbital
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
        if restart_idx is None:
            run_dir = f"{calc_i}/"
        else:
            run_dir = f"{calc_i}/{restart_idx}/"
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # Update control.in settings if there are some specific to the calculation
        if calc.get("aims_kwargs") is not None:
            aims_kwargs_calc = aims_kwargs.copy()
            aims_kwargs_calc.update(calc["aims_kwargs"])

        # Update sbatch settings if there are some specific to the calculation
        if calc.get("sbatch_kwargs") is not None:
            sbatch_kwargs_calc = sbatch_kwargs.copy()
            sbatch_kwargs_calc.update(calc["sbatch_kwargs"])

        # Copy density matrices if using restart
        if restart_idx is not None:
            for density_matrix in glob.glob(os.path.join(f"{calc_i}/", "D*.csc")):
                shutil.copy(density_matrix, run_dir)

        if copy_files is not None:
            for fname in copy_files:
                shutil.copy(os.path.join(f"{calc_i}/", fname), run_dir)

        # Write AIMS input files
        write_input_files(
            atoms=calc["atoms"], 
            run_dir=run_dir, 
            aims_kwargs=aims_kwargs_calc,
        )

        # Write sbatch run script
        write_aims_sbatch(
            fname=os.path.join(run_dir, "run-aims.sh"), 
            aims=calc["aims_path"], 
            load_modules=["intel", "intel-oneapi-mkl", "intel-oneapi-mpi"],
            **sbatch_kwargs_calc
        )

        # Run aims
        run_aims_in_dir(run_dir)

    return