import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from env import create_train_env
from model import ActorCritic
from optimizer import GlobalAdam
from process import local_train, initialize_global_model
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="right")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_episodes", type=int, default=50000)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--save_interval", type=int, default=500, help="Number of episodes between savings")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--use_gpu", type=bool, default=True)
    args = parser.parse_args()
    return args

def run_worker(rank, opt):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    WORKER_NAME = "worker{}"
    print(f"Rank {rank} start")
    if rank == 0:
        # rank0 will hold the global model and update it
        initialize_global_model(opt, rank)
        rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=opt.world_size)
    else:
        # other ranks will train on local model and update global
        options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
        mapping = {}
        mapping[rank] = 0
        print(f"setting cuda_rpc options {mapping}")
        options.set_device_map(WORKER_NAME.format(0), mapping)
        rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=opt.world_size, rpc_backend_options=options)
        # only the first rank will record loss/reward and save the model
        log = rank == 1
        local_train(rank, opt, log)

    # block until all rpcs finish, and shutdown the RPC instance
    rpc.shutdown()
    print(f"Rank {rank} shutdown")

if __name__ == "__main__":
    opt = get_args()
    # log_writer = SummaryWriter()
    mp.spawn(
        run_worker,
        args=(opt,),
        nprocs=opt.world_size,
        join=True,
    )
