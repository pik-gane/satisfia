from gymnasium import Env
from . import WorldModel, MDP

class MDPWorldModel(WorldModel, MDP):
    """A WorldModel of a (fully observed) MDP environment.
    
    In all methods, history is now a single-element list [state].
    """

    def __init__(self):
        super().__init__()

    def transition_distribution(self, action, history, n_samples = None):
        """Return a dictionary mapping results of calling step(action) after the given history 
        to tuples of the form (probability: float, exact: boolean).
        
        If not overridden, will sample n_samples times and return the empirical distribution."""
        old_state = self.state()
        state = history[0]
        frequencies = {}
        for i in range(n_samples):
            self.reset(state)
            result = self.step(action)
            try:
                frequencies[result] += 1
            except KeyError:
                frequencies[result] = 1
        self.reset(old_state)
        return {result: (frequency / n_samples, False) 
                for (result, frequency) in frequencies.items()}
    