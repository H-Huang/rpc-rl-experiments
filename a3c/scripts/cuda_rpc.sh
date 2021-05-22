#!/bin/bash

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

echo "master addr" $MASTER_ADDR
echo "cuda visible devices" $CUDA_VISIBLE_DEVICES
echo "ntasks" $SLURM_NTASKS
echo "procid" $SLURM_PROCID

python main.py --master_addr=${MASTER_ADDR} --world_size=12 --execution_mode=cuda_rpc --rank=${SLURM_PROCID} --num_episodes=100