import numpy as np
from numpy import random
from numpy.random import choice
from gymnasium import Env #, ResetNeeded # TODO: replace ResetNeeded with a custom exception?

# TODO: add typing

# TODO: define Exceptions for: action set empty, action not possible in state

class WorldModel(Env):
    """An abstract base class for potentially probabilistic world models, 
    extending gymnasion.Env by providing methods for enquiring transition probabilities between 
    environmental states.

    In addition to all not implemented methods, implementations must also either... 
    - override both possible_successors and transition_probability (in which case transition_distribution uses them),
    - or override transition_distribution in which case transition_distribution uses the other two,
    - or override reset and step (in which case transition_distribution uses them to estimate the distributions via sampling).

    In all additional methods:

    - state is a detailed description of the current state of the environment that suffices to determine the
      probability distribution of possible successor states arising from a state and action.
    - result is a tuple (observation, reward, terminated) that could be returned by step().
    - history is a list of the form [observation, action, result, action, ..., result] (a full history) 
      or of the form [result, action, result, action, ..., result] (a truncated history), where:
        - observation is the observation returned by reset(),
        - the other results are the main parts of the return values of consecutively calling step(action) after the given history up to that point.
    - n_samples is the number of samples to use for estimating the probability if no exact computation is possible.
    """

    _state = None
    """The current state of the environment, possibly not directly observable by the agent."""

    # methods for enquiring transition probabilities between states:

    def is_terminal(self, state):
        """Return whether the given state is terminal, i.e., 
        an episode ends and the agent can no longer perform actions when reaching this state."""
        raise NotImplementedError()
    
    def possible_actions(self, state = None):
        """Return the list of all actions possible in a given state or in the current state if state is None.
        
        This default implementation assumes that the action space is of type gymnasium.spaces.Discrete,
        representing a range of integers."""
        space = self.action_space
        return range(space.start, space.start + space.n)
    
    def possible_successors(self, state, action, n_samples = None):
        """Return a list of possible successor states after performing action in state,
        or, if state and action are None, a list of possible initial states."""
        return list(self.transition_distribution(state, action, n_samples).keys())
    
    def reachable_states(self, state):
        """Return a list of all states that can be reached from the given state by taking any sequence of actions."""
        res = set([state])
        if not self.is_terminal(state):
            for action in self.possible_actions(state):
                for successor in self.possible_successors(state, action):
                    res.update(self.reachable_states(successor))
        return list(res)

    def transition_probability(self, state, action, successor, n_samples = None):
        """Return the probability of the successor state after performing action in state,
        or, if state and action are None, of successor being the initial state,
        and a boolean flag indicating whether the probability is exact."""
        return self.transition_distribution(state, action, n_samples).get(successor, (0, True))
    
    def transition_distribution(self, state, action, n_samples = None):
        """Return a dictionary mapping possible successor states after performing action in state,
        or, if state and action are None, of possible initial states,
        to tuples of the form (probability: float, exact: boolean)."""
        return {successor: self.transition_probability(state, action, successor, n_samples) 
                for successor in self.possible_successors(state, action, n_samples)}
    
    def observation_and_reward_distribution(self, state, action, successor, n_samples = None):
        """Return a dictionary mapping possible pairs of observation and reward after performing action in state
        and reaching successor, or, if state and action are None, of starting in successor as the initial state,
        to tuples of the form (probability: float, exact: boolean)."""
        raise NotImplementedError()

    # methods for enquiring expected values in states:

    def expectation_of_fct_of_reward(self, state, action, f, additional_args = (), n_samples = None):
        """Return the expected value of f(reward, *additional_args) after taking action in state."""
        return np.sum([successor_probability * reward_probability * f(reward, *additional_args)
                       for (successor, (successor_probability, _)) in self.transition_distribution(state, action, n_samples = n_samples).items()
                       for ((observation, reward), (reward_probability, _)) in self.observation_and_reward_distribution(state, action, successor, n_samples = n_samples).items()
                       ])
    
    expectation_of_fct_of_delta = expectation_of_fct_of_reward

    def raw_moment_of_reward(self, state, action, degree = 1, n_samples = None):
        """Return a raw moment of reward after taking action in state."""
        return self.expectation_of_fct_of_reward(state, action, lambda reward: reward**degree, n_samples = n_samples)
    
    raw_moment_of_delta = raw_moment_of_reward

    def expected_reward(self, state, action, n_samples = None):
        """Return the expected reward after taking action in state."""
        return self.raw_moment_of_reward(state, action, 1, n_samples = n_samples)
    
    expected_delta = expected_reward
    
    def expectation(self, state, action, f, additional_args = (), n_samples = None):
        """Return the expected value of f(successor, *additional_args) after taking action in state."""
        return np.sum([probability * f(successor, *additional_args)
                       for (successor, (probability, _)) in self.transition_distribution(state, action, n_samples = n_samples).items()])

    def expectation_of_fct_of_probability(self, state, action, f, additional_args = (), n_samples = None):
        """Return the expected value of f(successor, probability, *additional_args) after taking action in state,
        where probability is the probability of reaching successor after taking action in state."""
        return np.sum([probability * f(successor, probability, *additional_args)
                       for (successor, (probability, _)) in self.transition_distribution(state, action, n_samples = n_samples).items()])

    # methods for enquiring observation probabilities given histories:

    def possible_results(self, history, action, n_samples = None):
        """Return a list of possible results of calling step(action) after the given history,
        or, if history and action are None, of calling reset()."""
        return list(self.result_distribution(history, action, n_samples).keys())
    
    def result_probability(self, history, action, result, n_samples = None):
        """Return the probability of the given result of calling step(action) after the given history,
        or, if history and action are None, of calling reset(),
        and a boolean flag indicating whether the probability is exact."""
        return self.result_distribution(history, action, n_samples).get(result, (0, True))
    
    def result_distribution(self, history, action, n_samples = None):
        """Return a dictionary mapping results of calling step(action) after the given history,
        or, if action is None, of calling reset(history[0] or None),
        to tuples of the form (probability: float, exact: boolean)."""
        return {result: self.result_probability(history, action, result, n_samples) 
                for result in self.possible_results(history, action, n_samples)}
    
    # methods for enquiring expected values after histories:

    def _result2reward(self, result):
        return result[1]  # since result is a tuple (observation, reward, terminated)
    
    def expectation_of_fct_of_reward_after_history(self, history, action, f, additional_args = (), n_samples = None):
        """Return the expected value of f(reward, *additional_args) when calling step(action) after the given history."""
        return np.sum([probability * f(self._result2reward(result), *additional_args)
                       for (result, (probability, _)) in self.result_distribution(history, action, n_samples = None)])
    
    expectation_of_fct_of_delta_after_history = expectation_of_fct_of_reward_after_history

    def raw_moment_of_reward_after_history(self, history, action, degree, n_samples = None):
        """Return a raw moment of the reward of the given result of calling step(action) after the given history."""
        return self.expectation_of_fct_of_reward_after_history(history, action, lambda reward: reward**degree, n_samples = None)

    raw_moment_of_delta_after_history = raw_moment_of_reward_after_history

    def expected_reward_after_history(self, history, action, n_samples = None):
        """Return the expected reward of the given result of calling step(action) after the given history."""
        return self.raw_moment_of_reward_after_history(history, action, 1, n_samples = None)
    
    expected_delta_after_history = expected_reward_after_history

    def expectation_after_history(self, history, action, f, additional_args = (), n_samples = None):
        """Return the expected value of f(step(action), *additional_args) after the giving history."""
        return np.sum([probability * f(result, *additional_args)
                       for (result, (probability, _)) in self.result_distribution(history, action, n_samples = 
                       None)])

    # Our default implementation of standard gymnasium.Env methods uses sampling from the above distribution:

    def _sample_successor_observation_reward(self, action = None):
        """Auxiliary method for sampling successor, observation, and reward given action in current state.
        Also returns an info dict as the fourth item."""
        # draw a successor according to the transition distribution:
        transition_distribution = self.transition_distribution(None if action is None else self._state, action)
        successors = list(transition_distribution.keys())
        succ_probs = list(transition_distribution.values())
        drawn_succ_index = choice(len(successors), p = [succ_prob for (succ_prob, _) in succ_probs])
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

    def _set_state(self, state):
        """Implement if you want to use the standard implementations of reset and step below."""
        raise NotImplementedError()
    
    def reset(self, *, seed = None, options = None):
        """Reset the environment and return the initial observation, reward, terminated, False, {}."""
        if seed is not None:
            random.seed(seed)
        successor, observation, reward, info = self._sample_successor_observation_reward()
        self._set_state(successor)
        return observation, reward, self.is_terminal(successor), False, {}
    
    def step(self, action):
        """Perform the given action and return a tuple 
        (observation, reward, terminated, False, {})."""
        if self.is_terminal(self._state):  # episode was already terminated!
            raise Exception() # TODO: ResetNeeded() no longer available?
        successor, observation, reward, info = self._sample_successor_observation_reward(action)
        self._set_state(successor)
        return observation, reward, self.is_terminal(successor), False, {}
