#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=20
#SBATCH --output=./slurm_out/slurm_train.out
#SBATCH --get-user-env

python3 -c 'from rholearn import train; from ml_settings import ML_SETTINGS; train.train(ML_SETTINGS)'