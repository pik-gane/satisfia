from gymnasium import Env, Wrapper
from torch import Tensor
import numpy as np
import random
from typing import Tuple

DO_NOTHING_ACTION = 4

class RestrictToPossibleActionsWrapper(Wrapper):
    def __init__(self, env: Env, default_action: int = DO_NOTHING_ACTION):
        super().__init__(env)
        self.default_action = default_action

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
    
    def step(self, action, *args, **kwargs):
        # for some reason self.env.possible_actions(self.env.get_state()) behaves differently from
        # self.env.possible_actions(), although as far as i undestand they should behave the same
        possible_actions = self.env.possible_actions(self.env.get_state())
        if action not in possible_actions:
            # ah, this condition should always be true, but for some reason isn't in GW23
            if self.default_action in possible_actions:
                action = self.default_action
            else:
                action = random.choice(possible_actions)
        return self.env.step(action, *args, **kwargs)
    
class RescaleDeltaWrapper(Wrapper):
    def __init__(self, env: Env, from_interval: Tuple[int, int], to_interval: Tuple[int, int]):
        assert len(from_interval) == 2
        assert len(to_interval) == 2
        assert from_interval[0] < from_interval[1]
        assert to_interval[0] <= to_interval[1]

        super().__init__(env)

        self.from_interval = from_interval
        self.to_interval = to_interval

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
    
    def step(self, *args, **kwargs):
        observation, delta, done, truncated, info = self.env.step(*args, **kwargs)
        from_l, from_h = self.from_interval
        to_l, to_h = self.to_interval
        rescaled_delta = to_l + (to_h - to_l) / (from_h - from_l) * (delta - from_l)
        return observation, rescaled_delta, done, truncated, info

class ObservationToTupleWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        return self.to_tuple(observation), info

    def step(self, *args, **kwargs):
        observation, delta, done, truncated, info = self.env.step(*args, **kwargs)
        wrapped_observation = self.to_tuple(observation)
        return wrapped_observation, delta, done, truncated, info

    def to_tuple(self, x):
        if isinstance(x, (Tensor, np.ndarray)):
            x = tuple(x.tolist())
            print(f"Converted tuple shape: {len(x)}")  # Should match original dimension
        return x