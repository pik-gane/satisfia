from collections.abc import Callable
from typing import Generic, NamedTuple, Optional, TypeVar, NewType, Tuple, Dict, List, Any
import numpy as np
from numpy import random
from numpy.random import choice
from gymnasium import Env #, ResetNeeded # TODO: replace ResetNeeded with a custom exception?
from functools import cache

# TODO: add typing

# TODO: define Exceptions for: action set empty, action not possible in state
ObsType = TypeVar("ObsType")
Action = TypeVar("Action")
State = TypeVar("State")

Probability = Tuple[NewType("Probability", float), bool]
AmbiguousProbability = NewType("AmbiguousProbability", List[Probability])
PotentiallyAmbiguousProbability = Probability | AmbiguousProbability
Distribution = Dict[Any, Probability]
AmbiguousDistribution = List[Distribution]
PotentiallyAmbiguousDistribution = Distribution | AmbiguousDistribution

class TuplePlus(tuple):
    """A tuple of numbers that supports element-wise addition, subtraction, multiplication, division, and exponentiation."""
    def __neg__(self):
        return TuplePlus(-a for a in self)
    def __add__(self, other):
        return TuplePlus(a + b for a, b in zip(self, other))
    def __sub__(self, other):
        return TuplePlus(a - b for a, b in zip(self, other))
    def __mul__(self, scalar):
        return TuplePlus(a * scalar for a in self)
    def __rmul__(self, scalar):
        return TuplePlus(a * scalar for a in self)
    def __truediv__(self, scalar):
        return TuplePlus(a / scalar for a in self)
    def __rtruediv__(self, scalar):
        return TuplePlus(a / scalar for a in self)
    def __pow__(self, scalar):
        return TuplePlus(a ** scalar for a in self)
    
class WorldModel(Generic[ObsType, Action, State], Env[ObsType, Action]):
    """An abstract base class for potentially probabilistic world models, 
    extending gymnasium.Env by providing methods for enquiring transition probabilities between 
    environmental states.

    In addition to all not implemented methods, implementations must also either... 
    - override both possible_successors and transition_probability (in which case transition_distribution uses them),
    - or override transition_distribution in which case transition_distribution uses the other two,
    - or override reset and step (in which case transition_distribution uses them to estimate the distributions via sampling).

    In all additional methods:

    - state is a detailed description of the current state of the environment that suffices to determine the
      unique or ambiguous probability distribution of possible successor states arising from a state and action.
    - result is a tuple (observation, reward, terminated) that could be returned by step().
    - history is a list of the form [observation, action, result, action, ..., result] (a full history) 
      or of the form [result, action, result, action, ..., result] (a truncated history), where:
        - observation is the observation returned by reset(),
        - the other results are the main parts of the return values of consecutively calling step(action) after the given history up to that point.
    - n_samples is the number of samples to use for estimating probabilities if no exact computation is possible.
    """
    class Result(NamedTuple):
        observation: ObsType
        reward: float
        terminated: bool

    _state: Optional[State] = None
    """The current state of the environment, possibly not directly observable by the agent."""

    # methods for enquiring transition probabilities between states:

    @cache
    def state_embedding(self, state: State):
        """vector representation of state"""
        try:
            return np.array(state, dtype=np.float32)
        except:
            raise ValueError("state must be an iterable of numbers")

    def is_terminal(self, state: State) -> bool:
        """Return whether the given state is terminal, i.e., 
        an episode ends and the agent can no longer perform actions when reaching this state."""
        raise NotImplementedError()
    
    @cache
    def possible_actions(self, state: Optional[State] = None) -> List[Action]:
        """Return the list of all actions possible in a given state or in the current state if state is None.
        
        This default implementation assumes that the action space is of type gymnasium.spaces.Discrete,
        representing a range of integers."""
        space = self.action_space
        return range(space.start, space.start + space.n)
    
    def default_policy(self, state: State) -> Optional[Action]:
        """Return a default action, if any"""
        return None

    @cache
    def possible_successors(self, state:State, action:Optional[Action]=None, n_samples:Optional[int] = None) -> set[State]:
        """Return the of possible successor states after performing action in state,
        or, if action is None, of all possible successor states after any action in state,
        or, if state and action are None, a list of possible initial states."""
        if action is None:
            res = set()
            for action in self.possible_actions(state):
                res.update(self.possible_successors(state, action, n_samples=n_samples))
            return list(res)
        else:
            data = self.transition_distribution(state, action, n_samples)
            if isinstance(data, dict):
                keys = data.keys()
            else:
                keys = set()
                for d in data:
                    keys.update(d.keys())
            return list(keys)
    
    @cache
    def reachable_states(self, state: State) -> set[State]:
        """Return a list of all states that can be reached from the given state by taking any sequence of actions."""
        res = {state}
        if not self.is_terminal(state):
            res |= { ns 
                for action in self.possible_actions(state)
                for successor in self.possible_successors(state, action)
                for ns in self.reachable_states(successor)
             }
        return res

    @cache
    def transition_probability(self, state: Optional[State], action: Optional[Action], successor: State, n_samples:Optional[int] = None) -> PotentiallyAmbiguousProbability:
        """Return the probability of the successor state after performing action in state,
        or, if state and action are None, of successor being the initial state,
        and a boolean flag indicating whether the information is exact.
        If the probability is ambiguous, return a list of pairs (p, exact)"""
        dist = self.transition_distribution(state, action, n_samples)
        if isinstance(dist, dict):
            return dist.get(successor, (0, True))
        else: # ambiguous distribution
            return [d.get(successor, (0, True)) for d in dist]
    
    @cache
    def transition_distribution(self, state:Optional[State], action:Optional[Action], n_samples:Optional[int] = None) -> PotentiallyAmbiguousDistribution:
        """Return a dictionary mapping possible successor states after performing action in state,
        or, if state and action are None, of possible initial states,
        to tuples of the form (probability: float, exact: boolean) or,
        if the transition distribution is ambiguous, return a list of such dictionaries."""
        data = {successor: self.transition_probability(state, action, successor, n_samples) 
                for successor in self.possible_successors(state, action, n_samples)}
        if all(isinstance(p, tuple) for p in data.values()):
            return data
        else:
            lens = [len(p) for p in data.values() if isinstance(p, list)]
            maxlen = max(lens)
            dists = [{} for _ in range(maxlen)]
            for succ, prob_or_probs in data.items():
                if isinstance(prob_or_probs, tuple):
                    for d in dists:
                        d[succ] = prob_or_probs
                else:
                    assert len(prob_or_probs) == maxlen, "ambiguous distribution must have the same length for all successors"
                    for i, prob in enumerate(prob_or_probs):
                        dists[i][succ] = prob
            return dists
    
    def observation_and_reward_distribution(self, state:Optional[State], action:Optional[Action], successor:State, n_samples:Optional[int] = None) -> Distribution:
        """Return a dictionary mapping possible pairs of observation and reward after performing action in state
        and reaching successor, or, if state and action are None, of starting in successor as the initial state,
        to tuples of the form (probability: float, exact: boolean).
        Note that this distribution is currently not allowed to be ambiguous."""
        raise NotImplementedError()

    # methods for enquiring expected values in states:

    def expectation_of_fct_of_reward(self, state:State, action:Action, f, additional_args = (), n_samples:Optional[int]= None):
        """Return the expected value of f(reward, *additional_args) after taking action in state or,
        if the transition distribution is ambiguous, return a list of possible values."""
        trans_dists = self.transition_distribution(state, action, n_samples = n_samples)
        if not isinstance(trans_dists, list):
            trans_dists = [trans_dists]
        res = [sum(successor_probability * reward_probability * f(reward, *additional_args)
                       for (successor, (successor_probability, _)) in trans_dist.items()
                       if successor_probability > 0
                       for ((observation, reward), (reward_probability, _)) in self.observation_and_reward_distribution(state, action, successor, n_samples = n_samples).items()
                       if reward_probability > 0
                       )
                for trans_dist in trans_dists]
        return res[0] if len(res) == 1 else res
        
    expectation_of_fct_of_delta = expectation_of_fct_of_reward

    @cache
    def raw_moment_of_reward(self, state:State, action:Action, degree:int = 1, n_samples:Optional[int] = None):
        """Return a raw moment of reward (or list of ambiguous raw moments) after taking action in state."""
        return self.expectation_of_fct_of_reward(state, action, lambda reward: reward**degree, n_samples = n_samples)
    
    raw_moment_of_delta = raw_moment_of_reward

    @cache
    def expected_reward(self, state:State, action:Action, n_samples:Optional[int] = None):
        """Return the expected reward (or list of ambiguous expected reward) after taking action in state."""
        return self.raw_moment_of_reward(state, action, 1, n_samples = n_samples)
    
    expected_delta = expected_reward
    
    def expectation( self, state:State, action:Action,
            f:Callable,
              additional_args=(), n_samples:Optional[int] = None):
        """Return the expected value of f(successor, *additional_args) after taking action in state."""
        trans_dists = self.transition_distribution(state, action, n_samples = n_samples)
        if not isinstance(trans_dists, list):
            trans_dists = [trans_dists]
        res = [sum(probability * f(successor, *additional_args)
                       for (successor, (probability, _)) in trans_dist.items()
                       if probability > 0)
                for trans_dist in trans_dists]
        return res[0] if len(res) == 1 else res
    
    def expectation_of_fct_of_probability(self, state: State, action: Action,
                                                      f:Callable,
                                          additional_args=(), n_samples = None):
        """Return the expected value of f(successor, probability, *additional_args) after taking action in state,
        where probability is the probability of reaching successor after taking action in state."""
        trans_dists = self.transition_distribution(state, action, n_samples = n_samples)
        if not isinstance(trans_dists, list):
            trans_dists = [trans_dists]
        res = [sum(probability * f(successor, probability, *additional_args)
                       for (successor, (probability, _)) in trans_dist.items()
                       if probability > 0)
                for trans_dist in trans_dists]
        return res[0] if len(res) == 1 else res
    
    # methods for enquiring observation probabilities given histories:

    @cache
    def possible_results(self, history, action: Action, n_samples: Optional[int] = None):
        """Return a list of possible results of calling step(action) after the given history,
        or, if history and action are None, of calling reset()."""
        data = self.result_distribution(history, action, n_samples)
        if isinstance(data, dict):
            keys = data.keys()
        else:
            keys = set()
            for d in data:
                keys.update(d.keys())
        return list(keys)
    
    @cache
    def result_probability(self, history, action:Action, result, n_samples:Optional[int] = None):
        """Return the probability of the given result of calling step(action) after the given history,
        or, if history and action are None, of calling reset(),
        and a boolean flag indicating whether the probability is exact.
        If the probability is ambiguous, return a list of pairs (p, exact)"""
        dists = self.result_distribution(history, action, n_samples)
        if isinstance(dists, dict):
            dists = [dists]
        res = [dist.get(result, (0, True)) for dist in dists]
        return res[0] if len(res) == 1 else res
    
    @cache
    def result_distribution(self, history, action:Action, n_samples :Optional[int]= None) -> PotentiallyAmbiguousDistribution:
        """Return a dictionary mapping results of calling step(action) after the given history,
        or, if action is None, of calling reset(history[0] or None),
        to tuples of the form (probability: float, exact: boolean) or,
        if the transition distribution is ambiguous, return a list of such dictionaries."""
        data = {result: self.result_probability(history, action, result, n_samples) 
                for result in self.possible_results(history, action, n_samples)}
        if all(isinstance(p, tuple) for p in data.values()):
            return data
        else:
            lens = [len(p) for p in data.values() if isinstance(p, list)]
            maxlen = max(lens)
            dists = [{} for _ in range(maxlen)]
            for result, prob_or_probs in data.items():
                if isinstance(prob_or_probs, tuple):
                    for d in dists:
                        d[result] = prob_or_probs
                else:
                    assert len(prob_or_probs) == maxlen, "ambiguous distribution must have the same length for all possible results"
                    for i, prob in enumerate(prob_or_probs):
                        dists[i][result] = prob
            return dists
    
    # methods for enquiring expected values after histories:

    def expectation_of_fct_of_reward_after_history(self, history, action: Action, 
                                                   f:Callable, additional_args =(), n_samples:Optional[int] = None):
        """Return the expected value of f(reward, *additional_args) when calling step(action) after the given history or,
        if the transition distribution is ambiguous, return a list of possible values.."""
        res_dists = self.result_distribution(history, action, n_samples = None)
        if not isinstance(res_dists, list):
            res_dists = [res_dists]
        res = [sum(probability * f(result.reward, *additional_args)
                       for (result, (probability, _)) in res_dist.items()
                       if probability > 0)
                for res_dist in res_dists]
        return res[0] if len(res) == 1 else res
    
    expectation_of_fct_of_delta_after_history = expectation_of_fct_of_reward_after_history

    @cache
    def raw_moment_of_reward_after_history(self, history, action:Action, degree:float, n_samples:Optional[int] = None):
        """Return a raw moment of the reward of the given result of calling step(action) after the given history."""
        return self.expectation_of_fct_of_reward_after_history(history, action, lambda reward: reward**degree, n_samples = None)

    raw_moment_of_delta_after_history = raw_moment_of_reward_after_history

    @cache
    def expected_reward_after_history(self, history, action, n_samples = None):
        """Return the expected reward of the given result of calling step(action) after the given history."""
        return self.raw_moment_of_reward_after_history(history, action, 1, n_samples = None)
    
    expected_delta_after_history = expected_reward_after_history

    def expectation_after_history(self, history, action, f, additional_args = (), n_samples = None):
        """Return the expected value of f(step(action), *additional_args) after the giving history."""
        res_dists = self.result_distribution(history, action, n_samples = None)
        if not isinstance(res_dists, list):
            res_dists = [res_dists]
        res = [sum(probability * f(result, *additional_args)
                       for (result, (probability, _)) in res_dist.items()
                       if probability > 0)
                for res_dist in res_dists]
        return res[0] if len(res) == 1 else res

    # Our default implementation of standard gymnasium.Env methods uses sampling from the above distribution:

    def _sample_successor_observation_reward(self, action: Optional[Action] = None) -> tuple[State, ObsType, float, dict]:
        """Auxiliary method for sampling successor, observation, and reward given action in current state.
        Also returns an info dict as the fourth item."""
        # draw a successor according to the transition distribution or, if ambiguous, from a possible transition distribution drawn uniformly at random:
        transition_distribution = self.transition_distribution(None if action is None else self._state, action)
        if isinstance(transition_distribution, list):
            transition_distribution = choice(transition_distribution)
        successors = list(transition_distribution.keys())
        succ_probs = list(transition_distribution.values())
        try:
            drawn_succ_index = choice(len(successors), p = [succ_prob for (succ_prob, _) in succ_probs])
        except:
            print("!", successors, succ_probs)
        successor = successors[drawn_succ_index]
        succ_prob, succ_prob_exact = succ_probs[drawn_succ_index] 
        # draw an observation and reward according to the observation and reward distribution:
        observation_and_reward_distribution = self.observation_and_reward_distribution(None if action is None else self._state, action, successor)
        observations_and_rewards = list(observation_and_reward_distribution.keys())
        res_probs = list(observation_and_reward_distribution.values())
        drawn_res_index = choice(len(observations_and_rewards), p = [res_prob for (res_prob, _) in res_probs])
        observation, reward = observations_and_rewards[drawn_res_index]
        res_prob, res_prob_exact = res_probs[drawn_res_index] 
        # return the sampled result in the form reset() or step(action) would return it:
        return (successor, observation, reward, {
            "total_p": succ_prob * res_prob, "total_p_exact": succ_prob_exact and res_prob_exact, 
            "p_successor": succ_prob, "p_successor_exact": succ_prob_exact, 
            "p_obs_and_reward": res_prob, "p_obs_and_reward_exact": res_prob_exact})

    def _set_state(self, state: State):
        """Implement if you want to use the standard implementations of reset and step below."""
        raise NotImplementedError()
    
    def reset(self, *, seed: Optional[float] = None, options = None):
        """Reset the environment and return the initial observation, reward, terminated, False, {}."""
        if seed is not None:
            random.seed(seed)
        successor, observation, reward, info = self._sample_successor_observation_reward()
        self._set_state(successor)
        assert self.observation_space.contains(observation), f"{observation} not in {self.observation_space.__dict__}"
        return observation, {}
    
    def step(self, action: Action):
        """Perform the given action and return a tuple 
        (observation, reward, terminated, False, {})."""
        assert action in self.possible_actions(self._state), f"{action} not possible in {self._state}"
        assert self.action_space.contains(action), f"{action} not in {self.action_space.__dict__}"
        if self.is_terminal(self._state):  # episode was already terminated!
            raise Exception() # TODO: ResetNeeded() no longer available?
        successor, observation, reward, info = self._sample_successor_observation_reward(action)
        self._set_state(successor)
        assert self.observation_space.contains(observation), f"{observation} not in {self.observation_space.__dict__}"
        return observation, reward, self.is_terminal(successor), False, {}
    
    # Methods for enabling the computation of reversibility metrics:

    def get_prolonged_version(self, horizon=None):
        """Return a version of this world model that allows for at least horizon many further steps 
        at each terminal state of the original world model. 
        This requires modification of terminal states, adding actions to the former terminal states, 
        and possibly adding new states.
        All formerly non-terminal states, their action spaces, and the corresponding transitons must remain unchanged."""
        raise NotImplementedError()
