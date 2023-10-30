"""
Module for setting up and running FHI-AIMS calculations.
"""
import inspect
import glob
import os
import shutil
from typing import Callable, Optional, List

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
    aims_io.write_control(
        os.path.join(run_dir, "control.in"), atoms=atoms, parameters=aims_kwargs
    )

    return


def write_aims_sbatch(
    fname: str, aims: str, load_modules: Optional[list] = None, **kwargs
):
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


def write_aims_sbatch_array(
    fname: str,
    aims: str,
    structure_idxs: List[int],
    run_dir: Callable,
    load_modules: Optional[list] = None,
    **kwargs,
):
    """
    Writes a bash script to `fname` that allows running of FHI-AIMS with
    specification of sbatch parameters via `kwargs`.
    """
    with open(fname, "wt") as f:

        # Make a dir for the slurm outputs
        if not os.path.exists("slurm_out"):
            os.mkdir("slurm_out")

        # Write the header
        f.write("#!/bin/bash\n")

        # Write the sbatch parameters
        for tag, val in kwargs.items():
            f.write(f"#SBATCH --{tag}={val}\n")
        
        # Write the array of structure indices
        f.write(f"#SBATCH --array={','.join(map(str, structure_idxs))}\n")
        f.write("#SBATCH --output=slurm_out/%a_slurm.out")
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

        # Get the structure idx in the sbatch array
        f.write("# Get the structure idx from the SLURM job ID\n")
        f.write("STRUCTURE_ID=${SLURM_ARRAY_TASK_ID}\n\n")

        # Define the run directory and cd to it
        f.write("# Define the run directory and cd into it\n")
        f.write(f"RUNDIR={run_dir('${STRUCTURE_ID}')}\n")
        f.write("cd $RUNDIR\n\n")

        # Write the run AIMS command
        f.write("# Run AIMS\n")
        f.write(f"AIMS={aims}\n")
        f.write("srun $AIMS < control.in > aims.out\n")

    return


def run_aims_in_dir(run_dir: str, check_input_files: bool = True):
    """
    Runs FHI-AIMS in the directory `run_dir`.

    Files "geometry.in", "control.in", and "run-aims.sh" must be present in
    `run_dir`, unless `check_input_files` is False. This may be used when
    running an sbatch array job, whereby the slurm script changes to the
    relevant run directories before running AIMS.
    """
    if check_input_files:
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
    calcs: dict,
    aims_path: str,
    aims_kwargs: dict,
    sbatch_kwargs: dict,
    restart_idx: Optional[int] = None,
    copy_files: Optional[List[str]] = None,
):
    """
    Runs an AIMS calculation for each of the calculations in `calcs`. `calcs`
    must be dict with keys corresponding to the structure index, and values a
    dict containing "atoms" (i.e. an ase.Atoms object), and optionally
    "aims_kwargs" and "sbatch_kwargs".

    The `aims_kwargs` are used to write control.in files. The `sbatch_kwargs`
    are used to run the calulation, for instance on a HPC cluster.

    Any calculation-specific settings stored in `calcs` are used to update the
    `aims_kwargs` and `sbatch_kwargs` before writing the control.in file.

    If `restart_idx` is not None, then the density matrices from the structure
    directory at relative path f"{structure_idx}/" are copied into a
    subdirectory f"{structure_idx}/{restart_idx}/" and the calculation is run
    from there.
    """
    top_dir = os.getcwd()
    os.chdir(top_dir)

    for calc_i, calc in calcs.items():
        # Define run dir and AIMS path
        if restart_idx is None:
            run_dir = os.path.join(calc["run_dir"])
        else:
            run_dir = os.path.join(calc["run_dir"], f"{restart_idx}/")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # Update control.in settings if there are some specific to the
        # calculation
        aims_kwargs_calc = aims_kwargs.copy()
        if calc.get("aims_kwargs") is not None:
            aims_kwargs_calc.update(calc["aims_kwargs"])

        # Update sbatch settings if there are some specific to the calculation
        sbatch_kwargs_calc = sbatch_kwargs.copy()
        if calc.get("sbatch_kwargs") is not None:
            sbatch_kwargs_calc.update(calc["sbatch_kwargs"])

        # Copy density matrices if using restart
        if restart_idx is not None:
            for density_matrix in glob.glob(os.path.join(calc["run_dir"], "D*.csc")):
                shutil.copy(density_matrix, run_dir)

        if copy_files is not None:
            for fname in copy_files:
                shutil.copy(os.path.join(calc["run_dir"], fname), run_dir)

        # Write AIMS input files
        write_input_files(
            atoms=calc["atoms"],
            run_dir=run_dir,
            aims_kwargs=aims_kwargs_calc,
        )

        # Write sbatch run script
        write_aims_sbatch(
            fname=os.path.join(run_dir, "run-aims.sh"),
            aims=aims_path,
            load_modules=["intel", "intel-oneapi-mkl", "intel-oneapi-mpi"],
            **sbatch_kwargs_calc,
        )

        # Run aims
        run_aims_in_dir(run_dir)

    return


def run_aims_array(
    calcs: dict,
    aims_path: str,
    aims_kwargs: dict,
    sbatch_kwargs: dict,
    run_dir: Callable,
    restart_idx: Optional[int] = None,
    copy_files: Optional[List[str]] = None,
):
    """
    Runs an AIMS calculation for each of the calculations in `calcs`. `calcs`
    must be dict with keys corresponding to the structure index, and values a
    dict containing "atoms" (i.e. an ase.Atoms object), and optionally
    "aims_kwargs".

    A single sbatch script is used to run calculations for all structures, in
    parallel, as a batch array. As such, the same sbatch settings are used for
    all calculations.

    The `aims_kwargs` are used to write control.in files. The `sbatch_kwargs`
    are used to run the calulation, for instance on a HPC cluster.

    Any calculation-specific settings stored in `calcs` are used to update the
    `aims_kwargs` and `sbatch_kwargs` before writing the control.in file.

    If `restart_idx` is not None, then the density matrices from the structure
    directory at relative path f"{structure_idx}/" are copied into a
    subdirectory f"{structure_idx}/{restart_idx}/" and the calculation is run
    from there.
    """
    top_dir = os.getcwd()
    os.chdir(top_dir)

    # Define new run dir
    if restart_idx is None:
        new_run_dir = lambda calc_i: os.path.join(run_dir(calc_i))
    else:
        new_run_dir = lambda calc_i: os.path.join(run_dir(calc_i), f"{restart_idx}/")

    for calc_i, calc in calcs.items():

        if not os.path.exists(new_run_dir(calc_i)):
            os.makedirs(new_run_dir(calc_i))

        # Update control.in settings if there are some specific to the
        # calculation
        aims_kwargs_calc = aims_kwargs.copy()
        if calc.get("aims_kwargs") is not None:
            aims_kwargs_calc.update(calc["aims_kwargs"])

        # Update sbatch settings if there are some specific to the calculation
        sbatch_kwargs_calc = sbatch_kwargs.copy()
        if calc.get("sbatch_kwargs") is not None:
            sbatch_kwargs_calc.update(calc["sbatch_kwargs"])

        # Copy density matrices if using restart
        if restart_idx is not None:
            for density_matrix in glob.glob(os.path.join(run_dir(calc_i), "D*.csc")):
                shutil.copy(density_matrix, new_run_dir(calc_i))

        if copy_files is not None:
            for fname in copy_files:
                shutil.copy(os.path.join(run_dir(calc_i), fname), new_run_dir(calc_i))

        # Write AIMS input files
        write_input_files(
            atoms=calc["atoms"],
            run_dir=new_run_dir(calc_i),
            aims_kwargs=aims_kwargs_calc,
        )

    # Write sbatch run script in the top level directory, as this will run a
    # batch-array in all subdirectories corresponding to the structure indices
    # passed in `calcs`.
    write_aims_sbatch_array(
        fname=os.path.join(top_dir, "run-aims.sh"),
        aims=aims_path,
        structure_idxs=list(calcs.keys()),
        run_dir=new_run_dir,
        load_modules=["intel", "intel-oneapi-mkl", "intel-oneapi-mpi"],
        **sbatch_kwargs_calc,
    )

    # Run aims
    run_aims_in_dir(top_dir, check_input_files=False)

    return