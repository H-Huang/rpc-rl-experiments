import gym
import torch
import random, datetime, numpy as np
import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from skimage import transform

from gym.spaces import Box

from nes_py.wrappers import JoypadSpace


class ResizeObservation(gym.ObservationWrapper):
    """
    ResizeObservation downsamples each observation into a square image.
    """

    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        # cast float back to uint8
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


class SkipFrame(gym.Wrapper):
    """
    SkipFrame is a custom wrapper that inherits from gym.Wrapper and implements the step() function.
    Because consecutive frames don’t vary much, we can skip n-intermediate frames without losing much information.
    The n-th frame aggregates rewards accumulated over each skipped frame.
    """

    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


def create_env(env_name="SuperMarioBros-1-1-v0"):
    env = gym_super_mario_bros.make(env_name)

    # Restricts action space to only "right" and "jump + right"
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    # Accumulates rewards every 4th frame
    env = SkipFrame(env, skip=4)
    # Transform RGB image to graycale, [240, 256]
    env = GrayScaleObservation(env)
    # Downsample to new size, [1, 84, 84]
    env = ResizeObservation(env, shape=84)
    # Add extra precision to np.array state
    env = TransformObservation(env, f=lambda x: x / 255.)
    # Squash 4 consecutive frames of the environment into a 
    # single observation point to feed to our learning model, [4, 84, 84]
    env = FrameStack(env, num_stack=4)
    return env