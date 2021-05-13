import torch
import time
from torch.distributed.rpc.api import rpc_sync
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, remote
import os
import datetime
from pathlib import Path
from env_wrappers import create_env

from agent import MarioAgent
from metric_logger import MetricLogger
import threading
from .execution_mode import ExecutionMode
import statistics 

WORKER_NAME = "worker{}"


class Learner:
    def __init__(self, execution_mode, world_size, log_interval, save_dir):
        env = create_env("SuperMarioBros-1-1-v0")

        self.logger = MetricLogger(save_dir)
        self.agent = MarioAgent(
            state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir
        )
        self.learner_rref = RRef(self)
        self.actor_rrefs = []

        for actor_rank in range(1, world_size):
            actor_info = rpc.get_worker_info(WORKER_NAME.format(actor_rank))
            self.actor_rrefs.append(remote(actor_info, Actor, args=(actor_rank, execution_mode)))

        self.update_lock = threading.Lock()
        self.episode_lock = threading.Lock()
        self.episode = 0

        self.log_interval = log_interval

        self.rpc_actions = []
        self.device = torch.device("cuda:0")
        self.execution_mode = execution_mode

    def get_action(self, state):
        # Choose best action to take based on the current model and the given state
        return self.agent.act(state)

    def update_model(self, actor_rank, state, next_state, action, reward, done, start_time):
        if self.execution_mode != ExecutionMode.cuda_rpc:
            state, next_state, action, reward, done = self._convert_to_tensor(state, next_state, action, reward, done)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        self.rpc_actions.append(elapsed_time)
        # print(state.device)

        # Save to replay buffer
        self.agent.cache(state, next_state, action, reward, done)

        # Update model
        with self.update_lock:
            q, loss = self.agent.learn()

        # Logging
        if len(self.rpc_actions) % 50 == 0:
            print(f"Avg elapsed time: {statistics.mean(self.rpc_actions):0.6f} seconds, stddev: {statistics.stdev(self.rpc_actions):0.6f}")
        self.logger.log_step(reward.cpu().detach().numpy(), loss, q, actor_rank)

    def _convert_to_tensor(self, state, next_state, action, reward, done):
        state = (
            torch.FloatTensor(state).to(self.device)
        )
        next_state = (
            torch.FloatTensor(next_state).to(self.device)
        )
        action = (
            torch.LongTensor([action]).to(self.device)
        )
        reward = (
            torch.DoubleTensor([reward]).to(self.device)
        )
        done = (
            torch.BoolTensor([done]).to(self.device)
        )
        return state, next_state, action, reward, done

    def _actor_thread_execution(self, actor_rref, num_episodes):
        while self.episode < num_episodes:
            done = False
            while not done:
                # Kick off a step on the actor
                actor_rank, done = rpc_sync(
                    actor_rref.owner(),
                    actor_rref.rpc_sync().perform_step,
                    args=(self.learner_rref,),
                )
                if done:
                    with self.episode_lock:
                        self.logger.log_episode(actor_rank)
                        if self.episode and self.episode % self.log_interval == 0:
                            self.logger.record(
                                episode=self.episode,
                                epsilon=self.agent.exploration_rate,
                                step=self.agent.curr_step,
                            )
                        self.episode += 1

    def run(self, num_episodes):
        actor_threads = []
        for actor_rref in self.actor_rrefs:
            t = threading.Thread(
                target=self._actor_thread_execution, args=(actor_rref, num_episodes)
            )
            t.start()
            actor_threads.append(t)
        for thread in actor_threads:
            thread.join()


class Actor:
    def __init__(self, rank, execution_mode):
        self.rank = rank
        self.device = torch.device(f"cuda:{self.rank}")
        self.execution_mode = execution_mode
        self.env = create_env("SuperMarioBros-1-1-v0")
        
        # Seed environment with unique rank so each environment is different
        self.env.seed(rank)
        self.is_done = True
        self.state = None

        # metrics
        self.rpc_actions = []

    def _convert_to_tensor(self, state, next_state, action, reward, done):
        convert = self.execution_mode == ExecutionMode.cuda_rpc

        state = (
            torch.FloatTensor(state).to(self.device)
            if convert
            else torch.FloatTensor(state)
        )
        next_state = (
            torch.FloatTensor(next_state).to(self.device)
            if convert
            else torch.FloatTensor(next_state)
        )
        action = (
            torch.LongTensor([action]).to(self.device)
            if convert
            else torch.LongTensor([action])
        )
        reward = (
            torch.DoubleTensor([reward]).to(self.device)
            if convert
            else torch.DoubleTensor([reward])
        )
        done = (
            torch.BoolTensor([done]).to(self.device)
            if convert
            else torch.BoolTensor([done])
        )
        return state, next_state, action, reward, done

    def _rpc_update_model(self, learner_rref, state, next_state, action, reward, done):
        state, next_state, action, reward, done = self._convert_to_tensor(state, next_state, action, reward, done)

        start_time = time.perf_counter()
        learner_rref.rpc_sync().update_model(
            self.rank, state, next_state, action, reward, done, start_time
        )

    def perform_step(self, learner_rref):
        """
        Performs a step on this actor's environment and sends the
        state, action, reward, next_state (s, a, r, s') back to the Learner
        to save in it's replay buffer
        """
        if self.is_done:
            # Start of a new game episode
            state = self.env.reset()
            self.is_done = False
        else:
            # Continue existing game episode
            state = self.state

        # Send state to learner to get an action
        # print(state)
        action = learner_rref.rpc_sync().get_action(state)

        # Perform action
        next_state, reward, done, info = self.env.step(action)
        # print(next_state)
        # print(torch.from_numpy(next_state))

        # Report the reward to the learner's replay buffer and update model
        # learner_rref.rpc_sync().update_model(
        #     self.rank, state, next_state, action, reward, done
        # )
        self._rpc_update_model(learner_rref, state, next_state, action, reward, done)

        # Save the next state to be used for the next episode
        self.state = next_state

        # Check if end of game episode
        end_episode = done or info["flag_get"]
        if end_episode:
            self.is_done = True
            self.state = None

        return self.rank, end_episode


def run_worker(rank, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    print(f"Rank {rank} start")
    if rank == 0:
        # rank0 is the learner
        rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=args.world_size)
        learner = Learner(args.execution_mode, args.world_size, args.log_interval, args.save_dir)
        learner.run(args.num_episodes)
    else:
        # other ranks are the actors
        options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)
        if args.execution_mode == ExecutionMode.cuda_rpc:
            mapping = {}
            mapping[rank] = 0
            print(f"setting cuda_rpc options {mapping}")
            options.set_device_map(WORKER_NAME.format(0), mapping)
        rpc.init_rpc(WORKER_NAME.format(rank), rank=rank, world_size=args.world_size, rpc_backend_options=options)

    # block until all rpcs finish, and shutdown the RPC instance
    rpc.shutdown()
    print(f"Rank {rank} shutdown")


# Multi-process training using RPC
def multi_process_exec(args):
    mp.spawn(
        run_worker,
        args=(args,),
        nprocs=args.world_size,
        join=True,
    )
