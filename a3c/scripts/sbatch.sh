#!/bin/bash

#SBATCH --job-name=rl_rpc

#SBATCH --partition=q3

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-node=8

#SBATCH --cpus-per-task=8

#SBATCH --gpus-per-node=8

#SBATCH --gpus-per-task=8

#SBATCH --time=5:00:00

# Run sbatch ./scripts/sbatch.sh <script_name>
# e.g. sbatch ./scripts/sbatch.sh ./scripts/cuda_rpc.sh
if [ -e $1 ];
then
    echo "Executing $1 with slurm batch."
else
    echo "$1 does not exist."
fi

srun --label $1