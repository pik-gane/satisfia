import numpy as np
from numpy import random
from numpy.random import choice
from gymnasium import Env, ResetNeeded

# TODO: add typing

class WorldModel(Env):
    """An abstract base class for potentially probabilistic world models, 
    extending gymnasion.Env by providing methods for enquiring transition probabilities.
    Most implementations will probably combine this with a gymnasium.Env.

    Implementations must either override 
    - override both possible_results and transition_probability (in which case transition_distribution uses them),
    - or override transition_distribution in which case transition_distribution uses the other two,
    - or override reset and step (in which case transition_distribution uses them to estimate the distributions via sampling).

    In all additional methods:

    - history is a list of the form [observation, action, result, action, ..., result], the first being the observation returned by reset(), the other results being the main parts of the return values of consecutively calling step(action) after the given history up to that point.
    - result is a tuple (observation, reward, terminated) that could be returned by step().
    - n_samples is the number of samples to use for estimating the probability if no exact computation is possible.
    """

    history = None
    """The history of the current episode."""

    def __init__(self):
        super().__init__()

    def possible_results(self, history, action, n_samples = None):
        """Return a list of possible results of calling step(action) after the given history,
        or, if history and action are None, of calling reset()."""
        return self.transition_distribution(history, action, n_samples).keys()
    
    def transition_probability(self, history, action, result, n_samples = None):
        """Return the probability of the given result of calling step(action) after the given history,
        or, if history and action are None, of calling reset(),
        and a boolean flag indicating whether the probability is exact."""
        return self.transition_distribution(history, action, n_samples)[result]
    
    def transition_distribution(self, history, action, n_samples = None):
        """Return a dictionary mapping results of calling step(action) after the given history,
        or, if history and action are None, of calling reset(),
        to tuples of the form (probability: float, exact: boolean)."""
        return {result: self.transition_probability(history, action, result, n_samples) 
                for result in self.possible_results(history, action, n_samples)}
    
    def _result2reward(self, result):
        return result[1]
    
    def expected_reward(self, history, action, n_samples = None):
        """Return the expected reward of the given result of calling step(action) after the given history."""
        return np.sum([probability * self._result2reward(result)
                       for (result, (probability, _)) in self.transition_distribution(history, action, n_samples = 
                       None)])
    
    def expectation(self, history, f, action, additional_args = None, n_samples = None):
        """Return the expected value of f(step(action), *additional_args) after the giving history."""
        return np.sum([probability * f(result, *additional_args)
                       for (result, (probability, _)) in self.transition_distribution(history, action, n_samples = 
                       None)])
    
    # Our default implementation of standard gymnasium.Env methods uses ampling from the above distribution:

    def _sample(self, action = None):
        transition_distribution = self.transition_distribution(None if action is None else self.history, action)
        results = list(transition_distribution.keys())
        values = list(transition_distribution.values())
        result_index = choice(len(results), p = [probability for (probability, _) in values])
        result = results[result_index]
        probability, exact = values[result_index] 
        return result, {"probability": probability, "exact": exact}

    def reset(self, *, seed = None, options = None):
        """Reset the environment and return the initial observation."""
        if seed is not None:
            random.seed(seed)
        result, info = self._sample()
        self.history = [result]
        return result, info
    
    def step(self, action):
        """Perform the given action and return a tuple 
        (observation, reward, terminated, False, {"probability": probability, "exact": exact}, terminated)."""
        if self.history[-1][2]:  # episode was already terminated!
            raise ResetNeeded()
        result, info = self._sample(action)
        self.history.extend((action, result))
        return result + (False, info, result[2])
