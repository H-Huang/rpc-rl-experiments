# import torch
import torch.multiprocessing as mp
import os
# import datetime
# from pathlib import Path
# import threading
import sys
import pickle
import pathlib
from model import ActorCritic

# import time
# import statistics
# from env_wrappers import create_env


import grpc

# adding grpc_utils to PYTHONPATH
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), "grpc_utils"))

from grpc_utils import rl_pb2
from grpc_utils import rl_pb2_grpc

from concurrent import futures

MAX_MESSAGE_LENGTH = 2**30
SERVER_ADDRESS = 'localhost:29506'

model = ActorCritic(1,1)
class Servicer(rl_pb2_grpc.RL_GRPCServicer):
    def __init__(self):
        pass

    def get_global_model_state_dict(self, request, context):
        response = rl_pb2.Response(
            data=pickle.dumps(model.state_dict()))
        return response
    
    def update_global_model_parameters(self, request, context):
        local_model = pickle.loads(request.data)
        model.zero_grad()
        for local_param, global_param in zip(local_model.parameters(), model.parameters()):
            # print(global_param.grad, global_param._grad)
            # if global_param.grad is not None:
            #     print("break early")
            #     break
            global_param.grad = local_param.grad
        model.step()

def init_grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(), options=[('grpc.max_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    rl_pb2_grpc.add_RL_GRPCServicer_to_server(Servicer(), server)
    server.add_insecure_port(SERVER_ADDRESS)
    server.start()
    return server

def init_grpc_client():
    channel = grpc.insecure_channel(SERVER_ADDRESS, options=[('grpc.max_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    return rl_pb2_grpc.RL_GRPCStub(channel)

# class Learner(mario_pb2_grpc.GRPCMarioServicer):
#     def __init__(self):
#         pass

#     def get_action(self, request, context):
#         observation = pickle.loads(request.observation)
#         # dummy response
#         next_action = 1
#         # print(f"== server got request {observation}, and returns {next_action}")
#         response = mario_pb2.Response(
#             next_action=pickle.dumps(next_action))
#         return response

#     def run(self):
#         server = grpc.server(futures.ThreadPoolExecutor())

#         mario_pb2_grpc.add_GRPCMarioServicer_to_server(self, server)

#         server.add_insecure_port(SERVER_ADDRESS)
#         server.start()
#         server.wait_for_termination()


# class Actor:
#     def __init__(self, rank):
#         self.channel = grpc.insecure_channel(SERVER_ADDRESS)
#         self.stub = mario_pb2_grpc.GRPCMarioStub(self.channel)
#         self.env = create_env("SuperMarioBros-1-1-v0")

#         # Seed environment with unique rank so each environment is different
#         self.rank = rank
#         self.env.seed(self.rank)
#         self.is_done = True
#         self.state = None

#     def perform_step(self):
#         action = 0
#         rpc_actions = []

#         is_done = True
#         state = None
#         while True:
#             if is_done:
#                 # Start of a new game episode
#                 state = self.env.reset()
#                 is_done = False

#             next_state, reward, done, info = self.env.step(action)

#             observation = [self.rank, state, next_state, action, reward]
#             start_time = time.perf_counter()
#             request = mario_pb2.Request(observation=pickle.dumps(observation))
#             response = self.stub.get_action(request)
#             next_action = pickle.loads(response.next_action)
#             end_time = time.perf_counter()
#             elapsed_time = end_time - start_time
#             rpc_actions.append(elapsed_time)
#             if len(rpc_actions) % 50 == 0:
#                 print(f"Avg elapsed time: {statistics.mean(rpc_actions):0.6f} seconds, stddev: {statistics.stdev(rpc_actions):0.6f}")
#             state = next_state

#             end_episode = done or info["flag_get"]
#             if end_episode:
#                 is_done = True
#                 state = None
#             # print(f"== client got next action {next_action}")

def run_worker(rank, opt):
    print(f"Rank {rank} start")
    if rank == 0:
        # rank0 is the learner
        # learner = Learner()
        # learner.run()
        # init_grpc_server()
        # server = init_grpc_server()
        # server.wait_for_termination()
        pass
    else:
        # other ranks are the actors
        # actor = Actor(rank)
        # actor.perform_step()
        stub = init_grpc_client()
        observation = [1]
        request = rl_pb2.Request(data=pickle.dumps(observation))
        response = stub.get_global_model_state_dict(request)
        next_action = pickle.loads(response.data)
        print(next_action)

    print(f"Rank {rank} shutdown")

if __name__ == "__main__":
    opt = {}
    server = init_grpc_server()
    mp.spawn(
        run_worker,
        args=(opt,),
        nprocs=2,
        join=True,
    )
    server.stop(None)
    server.wait_for_termination()
