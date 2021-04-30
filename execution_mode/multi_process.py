import torch
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

LEARNER_NAME = "learner"
ACTOR_NAME = "actor{}"


class Learner:
    def __init__(self, world_size, log_interval, save_dir):
        env = create_env("SuperMarioBros-1-1-v0")

        self.logger = MetricLogger(save_dir)
        self.agent = MarioAgent(
            state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir
        )
        self.learner_rref = RRef(self)
        self.actor_rrefs = []

        for actor_rank in range(1, world_size):
            actor_info = rpc.get_worker_info(ACTOR_NAME.format(actor_rank))
            self.actor_rrefs.append(remote(actor_info, Actor, args=(actor_rank,)))

        self.update_lock = threading.Lock()
        self.episode_lock = threading.Lock()
        self.episode = 0

        self.log_interval = log_interval

    def get_action(self, state):
        # Choose best action to take based on the current model and the given state
        return self.agent.act(state)

    def update_model(self, state, next_state, action, reward, done):
        # Save to replay buffer
        self.agent.cache(state, next_state, action, reward, done)

        # Update model
        with self.update_lock:
            q, loss = self.agent.learn()

        # Logging
        self.logger.log_step(reward, loss, q)

    def _actor_thread_execution(self, actor_rref, num_episodes):
        while self.episode < num_episodes:
            done = False
            while not done:
                # Kick off a step on the actor
                done = rpc_sync(
                    actor_rref.owner(),
                    actor_rref.rpc_sync().perform_step,
                    args=(self.learner_rref,),
                )
                if done:
                    with self.episode_lock:
                        self.logger.log_episode()
                        if self.episode % self.log_interval == 0:
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
    def __init__(self, rank):
        self.env = create_env("SuperMarioBros-1-1-v0")
        # Seed environment with unique rank so each environment is different
        self.env.seed(rank)
        self.is_done = True
        self.state = None

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
        action = learner_rref.rpc_sync().get_action(state)

        # Perform action
        next_state, reward, done, info = self.env.step(action)

        # Report the reward to the learner's replay buffer and update model
        learner_rref.rpc_sync().update_model(state, next_state, action, reward, done)

        # Save the next state to be used for the next episode
        self.state = next_state

        # Check if end of game episode
        end_episode = done or info["flag_get"]
        if end_episode:
            self.is_done = True
            self.state = None

        return end_episode


def run_worker(rank, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    print(f"Rank {rank} start")
    if rank == 0:
        # rank0 is the learner
        rpc.init_rpc(LEARNER_NAME, rank=rank, world_size=args.world_size)
        learner = Learner(args.world_size, args.log_interval, args.save_dir)
        learner.run(args.num_episodes)
    else:
        # other ranks are the actors
        rpc.init_rpc(ACTOR_NAME.format(rank), rank=rank, world_size=args.world_size)

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
