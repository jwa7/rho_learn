#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=5
#SBATCH --output=./slurm_out/slurm_ri.out
#SBATCH --get-user-env

python3 -c 'from rhocalc.aims import ri; from dft_settings import DFT_SETTINGS; ri.ri(DFT_SETTINGS)'