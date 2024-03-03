from numpy import inf, log, argsort, sum
from .mdp import MDP

class FMDP(MDP):
    """A Feasibility Markov Decision Process"""

    S_good: set = None
    """Set of good states (from S). There must be no path between good states!"""

    P_good_min: float = None
    """Minimal acceptable probability of reaching a good state."""

    _kwargs = MDP._kwargs + ["S_good", "P_good_min"]
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

    # ALGORITHMS:

    def do_greedy_entropy_min(self, max_time: int = None, max_H: float = inf):
        """Find a low-entropy policy that reaches a good state with acceptable probability,
        if possible."""
        # calculate maximal fulfilment probabilities Pm:
        self.set_binary_reward()
        self.do_value_iteration()
        Pm = self.V.copy()
        S, T, s0, S_good = self.S, self.T, self.s0, self.S_good
        # sort actions by entropy:
        s2A = {
            s: argsort([- sum([p * log(p) for s2, p in Tsa.items()]) for Tsa in T[s].values()])
            for s in S
        }
        # low-entropy greedy search through tree of partial policies:
        F = {(s0,)}
        Aind = {(s0,): 0}
        Pr = {(s0,): 1}
        while True:
            # check if current F is still feasible:
            PmF = HF = 0
            for path in F:
                Prpath = Pr[path]
                PmF += Prpath * Pm[path[-1]]
                HF += Prpath * log(Prpath)
            if PmF < self.P_good_min or HF > max_H:
                # F is not feasibe, so don't traverse this branch:
                go_back = True
            else:
                # check if current F is complete:
                is_complete = True
                for path in F:
                    if len(path) < 1 + 2 * max_time and path[-1] not in S_good: 
                        is_complete = False
                        break
                if is_complete:
                    # TODO: evaluate it, break if acceptable! Otherwise:
                    go_back = True
                else:
                    # find next continuation:
                    a = None
                    for path in F:
                        s = path[-1]
                        A = T[s].keys()
                        i = Aind[path]
                        if i < len(A):
                            a = A[i]
                            Aind[path] += 1
                            break
                    if not a:
                        go_back = True
                    else:
                        # replace F by its child F/(path,a) and update Pr:
                        F.remove(path)
                        Prpath = Pr[path]
                        for s2, p in T[s][a].items():
                            path2 = (*path, a, s2)
                            F.add(path2)
                            Pr[path2] = Prpath * p
                        del Pr[path]
            if go_back:
                pass # TODO!!

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
            "bad": {"repeat": {"excellent": 0.3, "bad": 0.7}, "stop": {"end": 1}},
            "acceptable": pass_end,
            "good": pass_end,
            "better": pass_end,
            "excellent": pass_end,
            "end": pass_end
        },
        S_good = {"good", "better", "excellent"},
        s0 = "start",
        P_good_min = 0.7
        )
    fmdp.set_entropy_adjusted_reward(.01)
    fmdp.do_value_iteration()
    print(fmdp.V)
