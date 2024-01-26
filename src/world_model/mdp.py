from typing import Any
from gymnasium import Env

class MDP(Env):
    """An abstract base class for (fully observed) Markov Decision Processes, extending Env by the ability to reset the environment to a specific state, and to enquire the current state.
    
    All observations are full states, so the environment can be reset to an arbitrary state by providing the respective observation.
    """

    def __init__(self):
        super().__init__()

    def reset(self, seed=None, state = None):
        """Reset the environment to the given state, or to the default initial state if None."""
        if state is None:
            super().reset(seed=seed)
        else:
            raise NotImplementedError()
        
    def state(self):
        """Return the current state of the environment."""
        raise NotImplementedError()
    