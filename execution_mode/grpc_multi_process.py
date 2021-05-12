import torch
import torch.multiprocessing as mp
import os
import datetime
from pathlib import Path
import threading
import sys
import pickle
import pathlib

import grpc

# adding grpc_utils to PYTHONPATH
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), "grpc_utils"))

from .grpc_utils import mario_pb2
from .grpc_utils import mario_pb2_grpc

from concurrent import futures

SERVER_ADDRESS = 'localhost:29500'


class Learner(mario_pb2_grpc.GRPCMarioServicer):
    def __init__(self):
        pass

    def get_action(self, request, context):
        observation = pickle.loads(request.observation)
        # dummy response
        next_action = torch.ones(2, 4) * observation[2].sum()
        print(f"== server got request {observation}, and returns {next_action}")
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
    def __init__(self):
        self.channel = grpc.insecure_channel(SERVER_ADDRESS)
        self.stub = mario_pb2_grpc.GRPCMarioStub(self.channel)

    def perform_step(self):
        rank = 7
        state = torch.zeros(3, 3)
        next_state = torch.ones(3, 3)
        action = torch.zeros(2, 4)
        reward = 100

        observation = [rank, state, next_state, action, reward]
        request = mario_pb2.Request(observation=pickle.dumps(observation))

        response = self.stub.get_action(request)

        next_action = pickle.loads(response.next_action)
        print(f"== client got next action {next_action}")



def run_worker(rank):
    print(f"Rank {rank} start")
    if rank == 0:
        # rank0 is the learner
        learner = Learner()
        learner.run()
    else:
        # other ranks are the actors
        actor = Actor()
        actor.perform_step()

    print(f"Rank {rank} shutdown")


# Multi-process training using RPC
def grpc_multi_process_exec(args):
    mp.spawn(
        run_worker,
        nprocs=args.world_size,
        join=True,
    )
