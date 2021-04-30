import datetime
from env_wrappers import create_env
import torch

from agent import MarioAgent
from metric_logger import MetricLogger
from pathlib import Path


# Single process implementation for reference and testing
def single_process_exec(args):
    env = create_env("SuperMarioBros-1-1-v0")

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    mario = MarioAgent(
        state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=args.save_dir
    )

    logger = MetricLogger(args.save_dir)

    for e in range(args.num_episodes):

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

        if e and e % args.log_interval == 0:
            logger.record(
                episode=e, epsilon=mario.exploration_rate, step=mario.curr_step
            )