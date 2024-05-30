from gymnasium import Env, Wrapper
import random

DO_NOTHING_ACTION = 4

class RestrictToPossibleActionsWrapper(Wrapper):
    def __init__(self, env: Env, default_action: int = DO_NOTHING_ACTION):
        super().__init__(env)
        self.default_action = default_action

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
    
    def step(self, action, *args, **kwargs):
        possible_actions = self.env.possible_actions()
        if action not in possible_actions:
            # ah, this condition should always be true, but for some reason isn't in GW23
            if self.default_action in possible_actions:
                action = self.default_action
            else:
                action = random.choice(possible_actions)
        return self.env.step(action, *args, **kwargs)