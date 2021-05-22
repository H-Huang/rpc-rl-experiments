import torch
from env import create_train_env
from model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter
import datetime
import timeit
import itertools
from global_network import init_grpc_client, rpc_load_global, rpc_update_global
from execution_mode import ExecutionMode

def local_train(rank, opt, log_dir):
    torch.manual_seed(123 + rank)
    start_time = timeit.default_timer()
    if opt.execution_mode == ExecutionMode.grpc:
        grpc_stub = init_grpc_client(opt.master_addr)
    else:
        grpc_stub = None

    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    if opt.use_gpu:
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
        episode_start_time = timeit.default_timer()
        curr_episode += 1

        # Reset to global network
        global_state_dict, fetch_model_delay = rpc_load_global(opt.execution_mode, grpc_stub)
        local_model.load_state_dict(global_state_dict)

        if log_dir:
            log_writer.add_scalar(f"Train_{rank}/FetchModelDelay", fetch_model_delay, curr_episode)
            if curr_episode and curr_episode % opt.save_interval == 0:
                torch.save(global_state_dict,
                           f"{log_dir}/model_{opt.world}_{opt.stage}_ep{curr_episode}")

        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        h_0 = h_0.to(device)
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
            state = state.to(device)

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                state = torch.from_numpy(env.reset())
                state = state.to(device)
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        R = R.to(device)
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
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

        # perform backward locally
        local_model.zero_grad()
        total_loss.backward()
        
        # update global parameters
        gradients = []
        for param in local_model.parameters():
            if opt.execution_mode == ExecutionMode.cuda_rpc:
                gradients.append(param.grad)
            else:
                gradients.append(param.grad.cpu())
        update_model_delay = rpc_update_global(opt.execution_mode, gradients, grpc_stub)
        if log_dir:
            log_writer.add_scalar(f"Train_{rank}/UpdateModel", update_model_delay, curr_episode)

        episode_end_time = timeit.default_timer()
        if log_dir:
            log_writer.add_scalar(f"Train_{rank}/PerEpisodeTime", episode_end_time - episode_start_time, curr_episode)

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
