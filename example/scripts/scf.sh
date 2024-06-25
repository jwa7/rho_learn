#!/bin/bash
#SBATCH --job-name=scf
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=5
#SBATCH --output=./slurm_out/slurm_scf.out
#SBATCH --get-user-env

# ===== Uncomment and run each in turn
# python3 -c 'from rhocalc.aims import scf; from dft_settings import DFT_SETTINGS; scf.scf(DFT_SETTINGS)'
# python3 -c 'from rhocalc.aims import scf; from dft_settings import DFT_SETTINGS; scf.process_scf(DFT_SETTINGS)'