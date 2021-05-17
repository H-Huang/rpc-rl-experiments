import os
import argparse
from process import local_train
from global_network import initialize_global_model, init_grpc_server
from execution_mode import ExecutionMode
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import datetime
import time

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
    parser.add_argument("--save_interval", type=int, default=5000, help="Number of episodes between savings")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--execution_mode", type=ExecutionMode, choices=list(ExecutionMode), default=ExecutionMode.cpu_rpc)
    args = parser.parse_args()
    return args

def run_worker(rank, opt):
    # https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    WORKER_NAME = "worker{}"
    print(f"Rank {rank} start")
    if rank == 0:
        # rank0 will hold the global model and update it
        initialize_global_model(opt, rank)
        if opt.execution_mode != ExecutionMode.grpc:
            rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=opt.world_size)
        else:
            grpc_server = init_grpc_server()
            grpc_server.wait_for_termination()
    else:
        # other ranks will train on local model and update global
        if opt.execution_mode != ExecutionMode.grpc:
            options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
            if opt.execution_mode == ExecutionMode.cuda_rpc:
                mapping = {}
                mapping[rank] = 0
                print(f"Setting device_map for Rank {rank} - {mapping}")
                options.set_device_map(WORKER_NAME.format(0), mapping)
            rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=opt.world_size, rpc_backend_options=options)
        else:
            # wait for grpc server intialization
            time.sleep(3)
        # only the first rank will record loss/reward and save the model
        if rank == 1:
            timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            log_dir = f"runs/{timestr}_{opt.execution_mode}_{opt.world_size}"
            os.mkdir(log_dir)
            with open(f"{log_dir}/args.txt", "a+") as f:
                for key, val in dict(vars(opt)).items():
                    f.write(f"{key}: {val}\n")
        else:
            log_dir = None
        local_train(rank, opt, log_dir)

    # block until all rpcs finish, and shutdown the RPC instance
    if opt.execution_mode != ExecutionMode.grpc:
        rpc.shutdown()
    print(f"Rank {rank} finish")

if __name__ == "__main__":
    opt = get_args()
    mp.spawn(
        run_worker,
        args=(opt,),
        nprocs=opt.world_size,
        join=True,
    )
