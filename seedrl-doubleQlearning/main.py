import argparse
from enum import Enum
import json
import datetime
from pathlib import Path
from execution_mode.single_process import single_process_exec
from execution_mode.multi_process import multi_process_exec
from execution_mode.grpc_multi_process import grpc_multi_process_exec
from execution_mode.execution_mode import ExecutionMode


def save_args_to_config_file(args):
    argparse_dict = dict(vars(args))
    # convert all arguments which are not nums to strings
    # e.g. ExecutionMode, PosixPath
    for key, value in argparse_dict.items():
        if type(value) != int:
            argparse_dict[key] = str(value)

    json_object = json.dumps(argparse_dict, indent=4)

    # record args as json file
    with open(args.config_file, "w") as f:
        f.write(json_object)

    return json_object


def load_args_from_config_file(config_file_path):
    with open(config_file_path, "rt") as f:
        json_args = json.load(f)

    dict_args = dict(json_args)

    for key, value in json_args.items():
        if key == "execution_mode":
            json_args[key] = ExecutionMode(value)
        if key == "save_dir":
            json_args[key] = Path(value)

    t_args = argparse.Namespace()
    t_args.__dict__.update(json_args)
    args = parser.parse_args(namespace=t_args)

    return dict_args, args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RPC Reinforcement Learning for Mario",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # user arguments
    parser.add_argument(
        "--execution_mode",
        default=ExecutionMode.cpu_rpc,
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
    parser.add_argument("--config_file", help="path of the json config file")
    args = parser.parse_args()

    # if config file is not provided then create one
    if not args.config_file:
        save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
            "%Y-%m-%dT%H-%M-%S"
        )
        save_dir.mkdir(parents=True)
        args.save_dir = save_dir
        config_file_name = "config.json"
        args.config_file = save_dir / config_file_name
        config = save_args_to_config_file(args)

    # load config file
    printable_args, args = load_args_from_config_file(args.config_file)

    # arguments currently look like
    """
    {
        "execution_mode": "single_process", 
        "world_size": 5,
        "num_episodes": 100,
        "log_interval": 10,
        "config_file": "checkpoints/2021-04-30T09-27-16/config.json",
        "save_dir": "checkpoints/2021-04-30T09-27-16"
    }
    """
    print("=" * 70)
    print(f"Running with config: {args.config_file}")
    print(json.dumps(printable_args, indent=4))
    print("=" * 70)

    if args.execution_mode == ExecutionMode.single_process:
        single_process_exec(args)
    elif args.execution_mode == ExecutionMode.grpc:
        grpc_multi_process_exec(args)
    else:
        multi_process_exec(args)
