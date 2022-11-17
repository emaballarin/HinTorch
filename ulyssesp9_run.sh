#!/usr/bin/env bash
#SBATCH --job-name=hinton_repro_dsscdl      # <-- !CHANGE THIS!
#SBATCH --partition=power9
#SBATCH --nodes=1                           # No more needed
#SBATCH --sockets-per-node=2                # Only available
#SBATCH --cores-per-socket=16               # Max available
#SBATCH --threads-per-core=4                # Only available
#SBATCH --gpus-per-node=1                   # No more needed
#SBATCH --time=0-23:59                      # Max allowed
#SBATCH --mem=0                             # Whole node
#SBATCH --exclusive                         # Whole node
#SBATCH --mail-user=emanuele@ballarin.cc    # My email
#SBATCH --mail-type=ALL

# Multithreading
export OMP_NUM_THREADS=128 # 128 = 2*16*4

# Environment init
export HOME="/home/eballari/"
. $HOME/.bashrc

# Python
conda activate HinTorchP9

# Move to working directory
cd "$HOME/Downloads/HinTorch"

# The srun call will trigger the SLURM prologue on the compute nodes.
NPROCS=$(srun --nodes=${SLURM_NNODES} bash -c 'hostname' | wc -l)

# The actual commands
jupyter nbconvert --execute --to notebook repro.ipynb
python -O run.py

# Exit
exit 0
