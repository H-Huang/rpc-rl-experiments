#!/bin/bash

#SBATCH --job-name=single_process

#SBATCH --partition=q2

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --gpus-per-task=1

#SBATCH --time=20:00:00

srun --label ./benchmark_results/single_process/single_process.sh