from . import MDP
from gymnasium import Env

class MDPEnv(MDP, Env):
    """An abstract base class for an MDP that is also an Env and thus is in a particular state at each point in time and has the ability to enquire the current state.
    
    All observations are full states, so the environment can also be reset to an arbitrary state by providing the respective observation in reset(..., options={"state":...}).
    """

    def state(self):
        """Return the current state of the environment."""
        raise NotImplementedError()

    def reset(self, *, seed = None, options = None):
        """Reset the environment to the given state, or to the default initial state if options[state] is None."""
        if "state" in options:
            raise NotImplementedError()
        else:
            super().reset(seed=seed, options=options)            