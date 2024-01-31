from gymnasium import Env
from . import WorldModel, MDPEnv

class MDPWorldModel(WorldModel, MDPEnv):
    """A WorldModel of a (fully observed) MDP environment.
    
    In all methods, history is now a single-element list [state].
    """

    def transition_distribution(self, action, history, n_samples = None):
        """Return a dictionary mapping results of calling step(action) after the given history 
        or, if action is None, of calling reset(state),
        to tuples of the form (probability: float, exact: boolean).
        
        If not overridden, this will sample n_samples times and return the empirical distribution."""
        old_state = self.state()
        state = history[0]
        frequencies = {}
        for i in range(n_samples):
            if action is None:
                result = self.reset(state)
            else:
                self.reset(state)
                result = self.step(action)
            try:
                frequencies[result] += 1
            except KeyError:
                frequencies[result] = 1
        self.reset(old_state)
        return {result: (frequency / n_samples, False) 
                for (result, frequency) in frequencies.items()}
    