from typing import Iterable

from numpy import abs, max, sum


class MDP(object):
    """A finite Markov Decision Process"""

    S: set = None
    """State space"""

    T: dict = None
    """Dict of dicts of float transition probabilities, keyed by s, then a, then s' """

    R: dict = None
    """Dict of dicts of float rewards, keyed by s, then a, then s' """

    gamma: float = None
    """Discount factor (0...1)"""

    s0 = None
    """Initial state (from S)"""

    _kwargs = ["S", "T", "R", "gamma", "s0"]
    _sets = ["S"]
    _dicts = ["T", "R"]

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            assert key in self._kwargs
            if key in self._sets:
                assert isinstance(value, Iterable)
                value = set(value)
            if key in self._dicts:
                assert isinstance(value, dict)
            self.__dict__[key] = value
        if self.s0:
            assert self.s0 in self.S
        self._clear_cache()
        # TODO: more checks & convenience (e.g. fill T with self-transitions of remaining probability)

    _cache: dict = None
    """Cache"""

    def _clear_cache(self):
        self._cache = {}

    @property
    def r(self):
        """Expected reward dict, keyed by s, then a"""
        try:
            r = self._cache["r"]
        except:
            S, T, R = self.S, self.T, self.R
            self._cache["r"] = r = {
                s: {
                    a: sum([p * Rs[a][s2] for s2, p in Tsa.items()])
                    for a, Tsa in T[s].items()
                }
                for s, Rs in R.items()
            }
        return r

    @property
    def V(self):
        """State value dict, keyed by s"""
        assert "V" in self._cache, "Please compute V first using do_value_iteration()"
        return self._cache["V"]

    # ALGORITHMS:

    def do_value_iteration(self, v0: float = 0, tol: float = 1e-5, maxiter: int = 10000):
        """Performs standard value iteration to approximate the state value function"""
        S, T, gamma, r = self.S, self.T, self.gamma, self.r
        last_V = {s: v0 for s in S}
        for it in range(maxiter):
            next_V = {
                s: max(
                    [
                        rs[a] + gamma * sum([p * last_V[s2] for s2, p in Tsa.items()])
                        for a, Tsa in T[s].items()
                    ]
                )
                for s, rs in r.items()
            }
            if max([abs(next_V[s] - last_V[s]) for s in S]) < tol:
                self._cache["V"] = next_V
                return next_V
            last_V = next_V
        return None


if __name__ == "__main__":
    mdp = MDP(
        S=[1, 2],
        T={1: {1: {1: 1}, 2: {1: 0.3, 2: 0.7}}, 2: {1: {2: 1}, 2: {2: 0.2, 1: 0.8}}},
        R={1: {1: {1: 1}, 2: {1: 1, 2: 1}}, 2: {1: {2: 0}, 2: {2: 0, 1: 0}}},
        gamma=0.9,
    )
    mdp.do_value_iteration()
    print(mdp.V)
