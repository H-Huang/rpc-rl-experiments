#!/bin/bash

#SBATCH --job-name=rl_rpc

#SBATCH --partition=q1

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=96

#SBATCH --time=5:00:00

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