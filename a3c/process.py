import torch
from env import create_train_env
from model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from optimizer import GlobalAdam

from collections import deque
from torch.utils.tensorboard import SummaryWriter
import datetime
import timeit
import pickle
import itertools
import statistics
import torch.distributed.rpc as rpc
from enum import Enum
import time

class ExecutionMode(Enum):
    grpc = "grpc"
    cpu_rpc = "cpu_rpc"
    cuda_rpc = "cuda_rpc"
    cuda_rpc_with_batch = "cuda_rpc_with_batch"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

global_rank = None
global_model = None
global_optimizer = None
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
    global_optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)

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

def rpc_load_global(execution_mode):
    if execution_mode in (ExecutionMode.cuda_rpc, ExecutionMode.cpu_rpc):
        ts = {}
        cuda = execution_mode == ExecutionMode.cuda_rpc
        ts["tik"] = stamp_time(cuda)
        global_state_dict = torch.distributed.rpc.rpc_sync(WORKER_NAME.format(0), get_global_model_state_dict)
        ts["tok"] = stamp_time(cuda)
        delay = compute_delay(ts, cuda)
    return global_state_dict, delay

def rpc_update_global(local_model_parameters):
    pass

def local_train(rank, opt, log_dir):
    torch.manual_seed(123 + rank)
    start_time = timeit.default_timer()
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    if opt.execution_mode == ExecutionMode.cuda_rpc:
        device = torch.device("cuda:{}".format(rank))
    else:
        device = torch.device("cpu")
    print(f"Rank: {rank}, Device: {device}")
    if log_dir:
        log_writer = SummaryWriter(log_dir)
    local_model = ActorCritic(num_states, num_actions)
    local_model.to(device)
    local_model.train()
    state = torch.from_numpy(env.reset()).to(device)
    done = True
    curr_episode = 0
    while True:
        curr_episode += 1
        # tic = timeit.default_timer()
        # pickle.dumps(global_model.state_dict())
        # dumped = toc = timeit.default_timer()
        # print(f"time: {toc - tic}")

        # Reset to global network
        global_state_dict, delay = rpc_load_global(opt.execution_mode)
        local_model.load_state_dict(global_state_dict)

        if log_dir:
            log_writer.add_scalar(f"Train_{rank}/LoadGlobalModelDelay", delay, curr_episode)
            if curr_episode and curr_episode % opt.save_interval == 0:
                torch.save(global_state_dict,
                           f"{log_dir}/model_{opt.world}_{opt.stage}_ep{curr_episode}")

        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if opt.execution_mode == ExecutionMode.cuda_rpc:
            # h_0 = h_0.cuda()
            h_0 = h_0.to(device)
            # c_0 = c_0.cuda()
            c_0 = c_0.to(device)

        log_policies = []
        values = []
        rewards = []
        entropies = []

        # perform an episode until finished
        for _ in itertools.count():
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            state, reward, done, _ = env.step(action)
            state = torch.from_numpy(state)
            if opt.execution_mode == ExecutionMode.cuda_rpc:
                # state = state.cuda()
                state = state.to(device)

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                state = torch.from_numpy(env.reset())
                if opt.execution_mode == ExecutionMode.cuda_rpc:
                    # state = state.cuda()
                    state = state.to(device)
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.execution_mode == ExecutionMode.cuda_rpc:
            # R = R.cuda()
            R = R.to(device)
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.execution_mode == ExecutionMode.cuda_rpc:
            # gae = gae.cuda()
            gae = gae.to(device)
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        if log_dir:
            record(log_writer, log_dir, rank, curr_episode, sum(rewards), total_loss)
        # print(f"{index}, ep:{curr_episode}, loss:{total_loss}")

        # perform backward locally
        local_model.zero_grad()
        total_loss.backward()
        gradients = []
        for param in local_model.parameters():
            gradients.append(param.grad)
        # update global parameters
        torch.distributed.rpc.rpc_sync(WORKER_NAME.format(0), update_global_model_parameters, args=(gradients,))
        # print(f"{index} finished step")

        if curr_episode == opt.num_episodes:
            # print("Training process {} terminated".format(index))
            end_time = timeit.default_timer()
            print(f"Worker {rank} ran for {curr_episode} in {end_time - start_time}s")
            return

def record(log_writer, log_dir, rank, curr_episode, total_reward, total_loss):
    log_writer.add_scalar(f"Train_{rank}/Reward", total_reward, curr_episode)
    log_writer.add_scalar(f"Train_{rank}/Loss", total_loss, curr_episode)
    with open(f"{log_dir}/output.txt", "a+") as f:
        f.write(f"{datetime.datetime.now()} Process {rank}. Episode {curr_episode}. Reward: {total_reward}. Loss: {total_loss.item()}\n")
