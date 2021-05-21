#!/bin/bash

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
# export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
export NTASKS=${SLURM_NTASKS}
export PROCID=${SLURM_PROCID}

# echo "gloo socket" $GLOO_SOCKET_IFNAME
# echo "tp socket" $TP_SOCKET_IFNAME

echo "master addr" $MASTER_ADDR
echo "cuda visible devices" $CUDA_VISIBLE_DEVICES
echo "ntasks" $NTASKS
echo "procid" $PROCID
python main.py --master_addr=${MASTER_ADDR} --world_size=${NTASKS} --execution_mode=cuda_rpc --rank=${PROCID} --num_episodes=100