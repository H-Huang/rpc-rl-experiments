import argparse
from enum import Enum
from execution_mode.single_process import single_process_exec
from execution_mode.multi_process import multi_process_exec


class ExecutionMode(Enum):
    grpc = "grpc"
    cpu_rpc = "cpu_rpc"
    cuda_rpc = "cuda_rpc"
    cuda_rpc_with_batch = "cuda_rpc_with_batch"
    single_process = "single_process"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RPC Reinforcement Learning for Mario",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--execution_mode",
        default=ExecutionMode.single_process,
        type=ExecutionMode,
        choices=list(ExecutionMode),
        help="type of mode to run the experiment with",
    )
    parser.add_argument(
        "--world_size", default=5, type=int, metavar="W", help="number of workers"
    )
    parser.add_argument(
        "--num_episodes", default=100, type=int, metavar="N", help="number of episodes"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        metavar="N",
        help="episode interval between logs",
    )
    args = parser.parse_args()
    print(args)
    if args.execution_mode == ExecutionMode.single_process:
        single_process_exec(args)
    else:
        multi_process_exec(args)
