#!/bin/bash
#SBATCH --job-name=h2o
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=bigmem
#SBATCH --ntasks-per-node=10
#SBATCH --array=89,186,673,893,199,322,874,103,458,431
#SBATCH --output=slurm_out/%a_slurm.out

# Load modules, set env variables, increase stack size
module load intel intel-oneapi-mkl intel-oneapi-mpi

export OMP_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export MKL_NUM_THREADS=1

ulimit -s unlimited

# Get the structure idx from the SLURM job ID
STRUCTURE_ID=${SLURM_ARRAY_TASK_ID}

# Define the run directory and cd into it
RUNDIR=/home/abbott/rho/rho_learn/docs/example/field/data/${STRUCTURE_ID}/0/
cd $RUNDIR

# Run AIMS
AIMS=/home/abbott/codes/new_aims/FHIaims/build/aims.230905.scalapack.mpi.x
srun $AIMS < control.in > aims.out
