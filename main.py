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
import argparse
import threading

LEARNER_NAME = "learner"
ACTOR_NAME = "actor{}"


class Learner:
    def __init__(self, world_size):
        save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
            "%Y-%m-%dT%H-%M-%S"
        )
        save_dir.mkdir(parents=True)
        env = create_env("SuperMarioBros-1-1-v0")

        self.logger = MetricLogger(save_dir)
        self.agent = MarioAgent(
            state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir
        )
        self.learner_rref = RRef(self)
        self.actor_rrefs = []

        for actor_rank in range(1, world_size):
            actor_info = rpc.get_worker_info(ACTOR_NAME.format(actor_rank))
            self.actor_rrefs.append(remote(actor_info, Actor))

        self.update_lock = threading.Lock()
        self.episode_lock = threading.Lock()
        self.episode = 0

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

    def _actor_thread_execution(self, actor_rref, max_episodes):
        while self.episode < max_episodes:
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
                        if self.episode % 5 == 0:
                            self.logger.record(
                                episode=self.episode,
                                epsilon=self.agent.exploration_rate,
                                step=self.agent.curr_step,
                            )
                        self.episode += 1

    def run(self, episodes=None):
        actor_threads = []
        for actor_rref in self.actor_rrefs:
            t = threading.Thread(
                target=self._actor_thread_execution, args=(actor_rref, 50000)
            )
            t.start()
            actor_threads.append(t)
        for thread in actor_threads:
            thread.join()


class Actor:
    def __init__(self):
        self.env = create_env("SuperMarioBros-1-1-v0")
        self.is_done = True
        self.state = None

    def perform_step(self, learner_rref):
        """
        Performs a step on this actor's environment and sends the
        state, action, reward, next_state (s, a, r, s') back to the Learner
        to save in it's replay buffer
        """
        # print("in perform_step")
        if self.is_done:
            state = self.env.reset()
            self.is_done = False
        else:
            state = self.state

        # Send state to learner to get an action
        action = learner_rref.rpc_sync().get_action(state)

        # Perform action
        next_state, reward, done, info = self.env.step(action)

        # Report the reward to the learner's replay buffer and update model
        learner_rref.rpc_sync().update_model(state, next_state, action, reward, done)

        # set state
        self.state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            self.is_done = True
            self.state = None

        return done or info["flag_get"]


parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning for Mario",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed  for reproducibility"
)
args = parser.parse_args()


def run_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    print(f"Rank {rank} start")
    if rank == 0:
        # rank0 is the learner
        rpc.init_rpc(LEARNER_NAME, rank=rank, world_size=world_size)
        learner = Learner(world_size)
        learner.run()
    else:
        # other ranks are the actors
        rpc.init_rpc(ACTOR_NAME.format(rank), rank=rank, world_size=world_size)

    # block until all rpcs finish, and shutdown the RPC instance
    rpc.shutdown()
    print(f"Rank {rank} shutdown")


# Multi-process training using RPC
def multi_process():
    mp.spawn(run_worker, args=(args.world_size,), nprocs=args.world_size, join=True)


# Single process implementation for reference and testing
def single_process():
    env = create_env("SuperMarioBros-1-1-v0")

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    save_dir.mkdir(parents=True)

    mario = MarioAgent(
        state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir
    )

    logger = MetricLogger(save_dir)

    episodes = 51
    for e in range(episodes):

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if e % 5 == 0:
            logger.record(
                episode=e, epsilon=mario.exploration_rate, step=mario.curr_step
            )


if __name__ == "__main__":
    # single_process()
    multi_process()
