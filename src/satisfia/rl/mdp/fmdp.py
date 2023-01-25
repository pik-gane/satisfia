from numpy import log
from mdp import MDP

class FMDP(MDP):
    """A Feasibility Markov Decision Process"""

    S_good: set = None
    """Set of good states (from S). There must be no path between good states!"""

    _kwargs = MDP._kwargs + ["S_good"]
    _sets = MDP._sets + ["S_good"]

    def __init__(self, **kwargs) -> None:
        assert "R" not in kwargs and "gamma" not in kwargs  # since they will be computed
        super().__init__(**kwargs)

    def set_binary_reward(self):
        """Set R(s,a,s') to 1 iff s' is good and 0 otherwise"""
        S, T, S_good = self.S, self.T, self.S_good
        self.R = {
            s: {
                a: {s2: 1 if s2 in S_good else 0 for s2 in Tsa.keys()}
                for a, Tsa in T[s].items()
            }
            for s in S 
        }
        self.gamma = 1  # no discounting necessary, each trajectory can only contain one good state.

    def set_entropy_adjusted_reward(self, eta: float):
        """Set R(s,a,s') to 1 iff s' is good and 0 otherwise, 
        plus eta times log(T(s,a,s')) to discourage increasing the entropy of the trajectory"""
        assert eta > 0
        self.set_binary_reward()
        R, T = self.R, self.T
        self.R = {
            s: {
                a: {s2: Rsas2 + eta * log(T[s][a][s2]) for s2, Rsas2 in Rsa.items()}
                for a, Rsa in Rs.items()
            }
            for s, Rs in R.items()
        }

if __name__ == "__main__":
    pass_end = {"pass": {"end": 1}}
    fmdp = FMDP(
        S = ["start", "bad", "acceptable", "good", "better", "excellent", "end"], 
        T = {
            "start": {
                "safe": {"good": 1}, 
                "risky": {"better": 0.7, "acceptable": 0.3},  
                "unsafe": {"excellent": 0.6, "bad": 0.4}
            },
            "bad": {"repeat": {"excellent": 0.6, "bad": 0.4}, "stop": {"end": 1}},
            "acceptable": pass_end,
            "good": pass_end,
            "better": pass_end,
            "excellent": pass_end,
            "end": pass_end
        },
        S_good = {"good", "better", "excellent"}
        )
    fmdp.set_entropy_adjusted_reward(.01)
    fmdp.do_value_iteration()
    print(fmdp.V)