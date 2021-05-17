import torch
import torch.multiprocessing as mp
import os
import datetime
from pathlib import Path
import threading
import sys
import pickle
import pathlib
import time
import statistics
from env_wrappers import create_env


import grpc

# adding grpc_utils to PYTHONPATH
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), "grpc_utils"))

from .grpc_utils import mario_pb2
from .grpc_utils import mario_pb2_grpc

from concurrent import futures

SERVER_ADDRESS = 'localhost:29501'


class Learner(mario_pb2_grpc.GRPCMarioServicer):
    def __init__(self):
        pass

    def get_action(self, request, context):
        observation = pickle.loads(request.observation)
        # dummy response
        next_action = 1
        # print(f"== server got request {observation}, and returns {next_action}")
        response = mario_pb2.Response(
            next_action=pickle.dumps(next_action))
        return response

    def run(self):
        server = grpc.server(futures.ThreadPoolExecutor())

        mario_pb2_grpc.add_GRPCMarioServicer_to_server(self, server)

        server.add_insecure_port(SERVER_ADDRESS)
        server.start()
        server.wait_for_termination()


class Actor:
    def __init__(self, rank):
        self.channel = grpc.insecure_channel(SERVER_ADDRESS)
        self.stub = mario_pb2_grpc.GRPCMarioStub(self.channel)
        self.env = create_env("SuperMarioBros-1-1-v0")

        # Seed environment with unique rank so each environment is different
        self.rank = rank
        self.env.seed(self.rank)
        self.is_done = True
        self.state = None

    def perform_step(self):
        action = 0
        rpc_actions = []

        is_done = True
        state = None
        while True:
            if is_done:
                # Start of a new game episode
                state = self.env.reset()
                is_done = False

            next_state, reward, done, info = self.env.step(action)

            observation = [self.rank, state, next_state, action, reward]
            start_time = time.perf_counter()
            request = mario_pb2.Request(observation=pickle.dumps(observation))
            response = self.stub.get_action(request)
            next_action = pickle.loads(response.next_action)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            rpc_actions.append(elapsed_time)
            if len(rpc_actions) % 50 == 0:
                print(f"Avg elapsed time: {statistics.mean(rpc_actions):0.6f} seconds, stddev: {statistics.stdev(rpc_actions):0.6f}")
            state = next_state

            end_episode = done or info["flag_get"]
            if end_episode:
                is_done = True
                state = None
            # print(f"== client got next action {next_action}")



def run_worker(rank):
    print(f"Rank {rank} start")
    if rank == 0:
        # rank0 is the learner
        learner = Learner()
        learner.run()
    else:
        # other ranks are the actors
        actor = Actor(rank)
        actor.perform_step()

    print(f"Rank {rank} shutdown")


# Multi-process training using RPC
def grpc_multi_process_exec(args):
    mp.spawn(
        run_worker,
        nprocs=args.world_size,
        join=True,
    )
