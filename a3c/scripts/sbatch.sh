#!/bin/bash

#SBATCH --job-name=rl_rpc

#SBATCH --partition=q3

#SBATCH --nodes=2

#SBATCH --ntasks-per-node=8

#SBATCH --cpus-per-task=4

#SBATCH --gpus-per-node=8

#SBATCH --time=00:30:00

# Steps to run:
# 1. Activate environment
# 2. Run sbatch ./scripts/sbatch.sh <script_name>
# e.g. sbatch ./scripts/sbatch.sh ./scripts/cuda_rpc.sh
if [ -e $1 ];
then
    echo "Executing $1 with slurm batch."
else
    echo "$1 does not exist."
fi

srun --label $1