import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Space, Discrete
import numpy as np

class MultiArmedBandit(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_arms=2, num_steps=10):
        self.num_arms = num_arms
        self.num_steps = num_steps
        self.action_space = Discrete(num_arms)
        self.observation_space = Space((2 * num_steps,))

    def reset(self, seed=None, options=None):
        assert seed is None, "seeding is not yet supported"

        self.biases = np.random.normal(size=(self.num_arms,))

        self.action_history = []
        self.reward_history = []

        observation = np.array([-1.] * self.num_steps + [0.] * self.num_steps)
        # assert self.observation_space.contains(observation)
        info = {}
        return observation, info
    
    def step(self, action):
        assert self.action_space.contains(action)
        self.action_history.append(action)
        reward = np.random.normal(self.biases[action])
        self.reward_history.append(reward)
        observation = np.array(   self.action_history + [-1] * (self.num_steps - len(self.action_history))
                                + self.reward_history + [0.] * (self.num_steps - len(self.reward_history)) )
        # assert self.observation_space.contains(observation)
        done = len(self.action_history) >= self.num_steps
        truncated = False
        info = {}
        return observation, reward, done, truncated, info
    
gym.register( id="MultiArmedBandit-v0",
              entry_point="multi_armed_bandit:MultiArmedBandit" )

    
