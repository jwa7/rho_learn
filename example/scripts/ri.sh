#!/bin/bash
#SBATCH --job-name=ri
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=standard
#SBATCH --ntasks-per-node=5
#SBATCH --output=./slurm_out/slurm_ri.out
#SBATCH --get-user-env

# ===== Uncomment and run each in turn

# === RI: do not run as sbatch job
# python3 -c 'from rhocalc.aims import ri; from dft_settings import DFT_SETTINGS; ri.ri(DFT_SETTINGS)'

# === Processing: do not run as sbatch job
# python3 -c 'from rhocalc.aims import ri; from dft_settings import DFT_SETTINGS; ri.process_ri(DFT_SETTINGS)'

# === Cleanup: do not run as sbatch job
# python3 -c 'from rhocalc.aims import ri; from dft_settings import DFT_SETTINGS; ri.cleanup_ri(DFT_SETTINGS)'

# === Rebuild: can run as sbatch job or locally (involves waiting for FHI-aims to finish, so maybe better as sbatch)
# python3 -c 'from rhocalc.aims import ri; from dft_settings import DFT_SETTINGS; ri.rebuild(DFT_SETTINGS)'

# === STM images: can run either as sbatch job or locally (if large / lots of cube files)
# python3 -c 'from rhocalc.aims import ri; from dft_settings import DFT_SETTINGS; ri.stm(DFT_SETTINGS)'