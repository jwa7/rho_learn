#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=20
#SBATCH --output=./slurm_out/slurm_eval.out
#SBATCH --get-user-env

python3 -c 'from rholearn import eval; from dft_settings import DFT_SETTINGS; from ml_settings import ML_SETTINGS; eval.eval(DFT_SETTINGS, ML_SETTINGS)'