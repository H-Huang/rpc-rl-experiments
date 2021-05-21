actor critic code based off of https://github.com/uvipen/Super-mario-bros-A3C-pytorch

grpc:

python main.py --world_size=2 --execution_mode=grpc

cpu rpc:

python main.py --world_size=2 --execution_mode=cpu_rpc

cuda rpc:

python main.py --world_size=2 --execution_mode=cuda_rpc

python main.py --world_size=8 --execution_mode=cpu_rpc --num_episodes=7010