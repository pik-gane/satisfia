class MDP(object):
    """An abstract base class for (fully observed) Markov Decision Processes, offering the abiluty to enquire the current state.
    """

    def state(self):
        """Return the current state of the environment."""
        raise NotImplementedError()
    