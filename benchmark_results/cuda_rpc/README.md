gpurun python main.py --execution_mode=cuda_rpc --world_size=2

python main.py --execution_mode=cuda_rpc --world_size=2 --log_interval=1 --num_episodes=10
python main.py --config_file=checkpoints/2021-05-11T22-33-05/config.json

50 episodes:
Avg elapsed time: 0.006021 seconds, stddev: 0.001943
Avg elapsed time: 0.005994 seconds, stddev: 0.001947
Avg elapsed time: 0.005972 seconds, stddev: 0.001952
Avg elapsed time: 0.005945 seconds, stddev: 0.001957
Avg elapsed time: 0.005921 seconds, stddev: 0.001961