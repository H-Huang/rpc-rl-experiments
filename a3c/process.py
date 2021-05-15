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


class ExecutionMode(Enum):
    grpc = "grpc"
    cpu_rpc = "cpu_rpc"
    cuda_rpc = "cuda_rpc"
    cuda_rpc_with_batch = "cuda_rpc_with_batch"
    single_process = "single_process"

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
    device = torch.device("cuda:{}".format(global_rank))
    print(device)
    _, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    global_model = ActorCritic(num_states, num_actions)
    global_optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    global_model.to(device)

def get_global_model_state_dict():
    global global_model, global_rank
    # print("in get_global_model_state_dict")
    return global_model.state_dict()

def update_global_model_parameters(local_model):
    global global_model, global_optimizer, global_rank
    # print("in update_global_model_parameters")
    global_optimizer.zero_grad()
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        if global_param.grad is not None:
            break
        global_param._grad = local_param.grad
    global_optimizer.step()

WORKER_NAME = "worker{}"
def rpc_load_global(execution_mode):
    if execution_mode == ExecutionMode.cpu_rpc:
        pass

def rpc_update_global(local_model_parameters):
    pass

def local_train(rank, opt, log=False):
    torch.manual_seed(123 + rank)
    start_time = timeit.default_timer()
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    device = torch.device("cuda:{}".format(rank))
    print(device)
    if log:
        log_writer = SummaryWriter()
    else:
        log_writer = None
    local_model = ActorCritic(num_states, num_actions)
    optimizer = GlobalAdam(local_model.parameters(), lr=opt.lr)
    if opt.use_gpu:
        # local_model.cuda()
        local_model.to(device)
    local_model.train()
    state = torch.from_numpy(env.reset())
    if opt.use_gpu:
        # state = state.cuda()
        state = state.to(device)
    done = True
    curr_episode = 0
    while True:
        if log:
            if curr_episode and curr_episode % opt.save_interval == 0:
                torch.save(global_model.state_dict(),
                           "{}/a3c_super_mario_bros_{}_{}_ep{}".format(opt.saved_path, opt.world, opt.stage, curr_episode))
            # if curr_episode % 10 == 0:
            #     print("Process {}. Episode {}".format(index, curr_episode))
        curr_episode += 1
        # tic = timeit.default_timer()
        # pickle.dumps(global_model.state_dict())
        # dumped = toc = timeit.default_timer()
        # print(f"time: {toc - tic}")

        # Reset to global network
        global_state_dict = torch.distributed.rpc.rpc_sync(WORKER_NAME.format(0), get_global_model_state_dict)
        local_model.load_state_dict(global_state_dict)

        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if opt.use_gpu:
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
            if opt.use_gpu:
                # state = state.cuda()
                state = state.to(device)

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                state = torch.from_numpy(env.reset())
                if opt.use_gpu:
                    # state = state.cuda()
                    state = state.to(device)
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            # R = R.cuda()
            R = R.to(device)
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
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
        if log_writer:
            record(log_writer, rank, curr_episode, sum(rewards), total_loss)
        # print(f"{index}, ep:{curr_episode}, loss:{total_loss}")

        # perform backward locally
        optimizer.zero_grad()
        total_loss.backward()
        # update global parameters
        torch.distributed.rpc.rpc_sync(WORKER_NAME.format(0), update_global_model_parameters, args=(local_model,))
        # print(f"{index} finished step")

        if curr_episode == opt.num_episodes:
            # print("Training process {} terminated".format(index))
            end_time = timeit.default_timer()
            print('The code runs for %.2f s ' % (end_time - start_time))
            return

def record(writer, rank, curr_episode, total_reward, total_loss):
    writer.add_scalar(f"Train_{rank}/Reward", total_reward, curr_episode)
    writer.add_scalar(f"Train_{rank}/Loss", total_loss, curr_episode)
    with open("output.txt", "a+") as f:
        f.write(f"{datetime.datetime.now()} Process {rank}. Episode {curr_episode}. Reward: {total_reward}. Loss: {total_loss.item()}\n")
