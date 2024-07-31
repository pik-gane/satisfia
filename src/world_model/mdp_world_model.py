from typing import Any, Generic, Optional, TypeVar

from . import WorldModel, ObsType, Action, State
 
class MDPWorldModel(Generic[ObsType, Action, State],WorldModel[ObsType, Action, State]):
    """
    A WorldModel of a (fully observed) MDP environment, allowing the user to reset the environment to a given state.
    """

    def reset(self, *, seed = None, options = None):
        """Reset the environment to the given state, or to the default initial state if options[state] is None."""
        if options and "state" in options:
            raise NotImplementedError()
        else:
            return super().reset(seed=seed, options=options)

    def transition_distribution(self, state: State, action: Optional[Action], n_samples: int):
        """Return a dictionary mapping possible successor states after performing action in state,
        or, if state and action are None, of possible initial states,
        to tuples of the form (probability: float, exact: boolean).

        If not overridden, this will sample n_samples times and return the empirical distribution."""
        old_state = self._state
        frequencies: dict[Any, int] = {}
        for _ in range(n_samples):
            if action is None:
                result = self.reset(options={"state": state})
            else:
                self.reset(options={"state": state})
                result = self.step(action)
            frequencies[result] = frequencies.get(result, 0) + 1
        self.reset(options={"state": old_state})
        print("trans_dist len: ", len(frequencies))
        return {result: (frequency / n_samples, False) 
                for (result, frequency) in frequencies.items()}

