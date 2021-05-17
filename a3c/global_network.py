import torch
from env import create_train_env
from model import ActorCritic
import pickle
import sys
import os
import pathlib
import grpc
from concurrent import futures
import time
from execution_mode import ExecutionMode

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), "grpc_utils"))
from grpc_utils import rl_pb2
from grpc_utils import rl_pb2_grpc

global_rank = None
global_model = None
global_optimizer = None

# ========== GRPC ==========
MAX_MESSAGE_LENGTH = 2**30
SERVER_ADDRESS = 'localhost:29506'
class Servicer(rl_pb2_grpc.RL_GRPCServicer):
    def __init__(self):
        pass

    def get_global_model_state_dict(self, request, context):
        response = rl_pb2.Response(
            data=pickle.dumps(global_model.state_dict()))
        return response
    
    def update_global_model_parameters(self, request, context):
        gradients = pickle.loads(request.data)
        global_model.zero_grad()
        for grad, global_param in zip(gradients, global_model.parameters()):
            global_param.grad = grad
        global_optimizer.step()
        response = rl_pb2.Response(data=None)
        return response

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

# ========== Pytorch RPC ==========
def initialize_global_model(opt, rank):
    global global_model, global_optimizer, global_rank
    global_rank = rank
    if opt.execution_mode == ExecutionMode.cuda_rpc:
        device = torch.device("cuda:{}".format(global_rank))
    else:
        device = torch.device("cpu")
    _, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    global_model = ActorCritic(num_states, num_actions)
    print(f"Global model on {device}")
    global_model.to(device)
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=opt.lr)

def get_global_model_state_dict():
    global global_model, global_rank
    # print("in get_global_model_state_dict")
    return global_model.state_dict()

def update_global_model_parameters(gradients):
    global global_model, global_optimizer, global_rank
    # print("in update_global_model_parameters")
    global_optimizer.zero_grad()
    for grad, global_param in zip(gradients, global_model.parameters()):
        # print(global_param.grad, global_param._grad)
        # if global_param.grad is not None:
        #     print("break early")
        #     break
        global_param.grad = grad
    global_optimizer.step()

# ============ Helper functions ============
WORKER_NAME = "worker{}"
def stamp_time(cuda=False):
    if cuda:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    else:
        return time.time()

def compute_delay(ts, cuda=False):
    if cuda:
        torch.cuda.synchronize()
        return ts["tik"].elapsed_time(ts["tok"]) / 1e3
    else:
        return ts["tok"] - ts["tik"]

def rpc_load_global(execution_mode, grpc_stub=None):
    if execution_mode in (ExecutionMode.cuda_rpc, ExecutionMode.cpu_rpc):
        cuda = execution_mode == ExecutionMode.cuda_rpc
        ts = {}
        ts["tik"] = stamp_time(cuda)
        globel_model_state_dict = torch.distributed.rpc.rpc_sync(WORKER_NAME.format(0), get_global_model_state_dict)
        ts["tok"] = stamp_time(cuda)
        delay = compute_delay(ts, cuda)
    elif execution_mode == ExecutionMode.grpc:
        assert grpc_stub != None
        ts = {}
        ts["tik"] = stamp_time()
        request = rl_pb2.Request(data=None)
        response = grpc_stub.get_global_model_state_dict(request)
        globel_model_state_dict = pickle.loads(response.data)
        ts["tok"] = stamp_time()
        delay = compute_delay(ts)
    return globel_model_state_dict, delay

def rpc_update_global(execution_mode, gradients, grpc_stub=None):
    if execution_mode != ExecutionMode.grpc:
        torch.distributed.rpc.rpc_sync(WORKER_NAME.format(0), update_global_model_parameters, args=(gradients,))
    else:
        request = rl_pb2.Request(data=pickle.dumps(gradients))
        grpc_stub.update_global_model_parameters(request)