from . import MDP
from gymnasium import Env

class MDPEnv(MDP, Env):
    """An abstract base class for an MDP that is also an Env and thus has the ability to reset the environment to a specific state, and to enquire the current state.
    
    All observations are full states, so the environment can be reset to an arbitrary state by providing the respective observation.
    """

    def reset(self, *, seed = None, options = None, state = None):
        """Reset the environment to the given state, or to the default initial state if None."""
        if state is None:
            super().reset(seed=seed, options=options)
        else:
            raise NotImplementedError()
            