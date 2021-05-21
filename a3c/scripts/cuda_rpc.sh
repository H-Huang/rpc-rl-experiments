#!/bin/bash

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export NTASKS=${SLURM_NTASKS}
export PROCID=${SLURM_PROCID}

python main.py --master_addr=${MASTER_ADDR} --world_size=${SLURM_NTASKS} --execution_mode=cuda_rpc --rank=${SLURM_PROCID} --num_episodes=100