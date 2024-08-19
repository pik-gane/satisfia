#!/usr/bin/env python3

import itertools
import math
from functools import cache
import random
from networkx import center
import numpy as np
import polytope
from scipy.optimize import linprog

from satisfia.util import distribution
from satisfia.util.helper import *

from abc import ABC, abstractmethod

import pylab as plt

VERBOSE = False
DEBUG = False

linprog_options = {'tol':1e-5}

prettyState = str

def pad(state):
    return " :            " * (state[0] if isinstance(state, tuple) else 1)  # state[0] is the time step

def probability_add(p, key, weight):
    if weight < 0:
        raise ValueError("invalid weight")
    if weight > 0:
        if key in p:
            p[key] += weight
            if p[key] == 0:
                del p[key]
        else:
            p[key] = weight

class AspirationAgent(ABC):

    reachable_states = None
    default_transition = None

    def __init__(self, params):
        """
        If world is provided, maxAdmissibleQ, minAdmissibleQ, Q, Q2, ..., Q6 are not needed because they are computed from the world. Otherwise, these functions must be provided, e.g. as learned using some reinforcement learning algorithm. Their signature is
        - maxAdmissibleQ|minAdmissibleQ: (state, action) -> float
        - Q,Q2,...,Q6: (state, action, action, aleph4action) -> float

        disorderingPotential_action, agency_action, LRAdev_action, Q_ones, Q_DeltaSquare, behaviorEntropy_action, and behaviorKLdiv_action are only needed if their respective loss coefficients 
        (lossCoeff4DP, lossCoeff4AgencyChange, lossCoeff4LRA, lossCoeff4Time, lossCoeff4Entropy, lossCoeff4KLdiv)
        are nonzero and no world model is provided. Their signature is
        - disorderingPotential|agency_action: (state, action) -> float
        - LRAdev_action|Q_ones|Q_DeltaSquare: (state, action, aleph4action) -> float
        - behaviorEntropy_action|behaviorKLdiv_action: (state, actionProbability, action, aleph4action) -> float

        if lossCoeff4DP > 0, uninformedPolicy must be provided  
        if lossCoeff4Entropy > 0, referencePolicy or uninformedPolicy must be provided
        if lossCoeff4StateDistance > 0, referenceState must be provided

        """
        defaults = {
            # admissibility parameters:
            "maxLambda": 1, # upper bound on local relative aspiration in each step (must be minLambda...1)    # TODO: rename to lambdaHi
            "minLambda": 0, # lower bound on local relative aspiration in each step (must be 0...maxLambda)    # TODO: rename to lambdaLo
            # policy parameters:
            "lossTemperature": 1, # temperature of softmin mixture of actions w.r.t. loss, must be > 0
            # "rescaling4Actions": 0, # degree (0...1) of aspiration rescaling from state to action. (larger implies larger variance) # TODO: disable this because a value of >0 can't be taken into account in a consistent way easily
            # "rescaling4Successors": 1, # degree (0...1) of aspiration rescaling from action to successor state. (expectation is only preserved if this is 1.0) # TODO: disable also this since a value <1 leads to violation of the expectation guarantee 

            # THESE LOSS COMPONENTS DO NOT USE THE WORLD MODEL:

            # coefficients for cheap to compute loss functions:
            "lossCoeff4Random": 0, # weight of random tie breaker in loss function, must be >= 0
            "lossCoeff4FeasibilityPower": 1, # weight of power of squared admissibility interval width in loss function, must be >= 0
            "lossCoeff4LRA1": 1, # weight of current-state deviation of LRA from 0.5 in loss function, must be >= 0
            "lossCoeff4Time1": 1, # weight of not terminating in loss function, must be >= 0
            "lossCoeff4Entropy1": 1, # weight of current-state action entropy in loss function, must be >= 0
            "lossCoeff4KLdiv1": 0, # weight of current-state KL divergence in loss function, must be >= 0

            # THE FOLLOWING CAN IN PRINCIPLE ALSO BE COMPUTED OR LEARNED UPFRONT:

            "lossCoeff4DP": 0, # weight of disordering potential in loss function, must be >= 0
            "lossCoeff4AgencyChange": 0, # weight of expected absolute agency change in loss function, must be >= 0

            "uninformedStatePriorScore": lambda s: 0,
            "defaultTransitionScore": {},
            "internalTransitionEntropy": 0,

            # THESE LOSS COMPONENTS USE THE WORLD MODEL BECAUSE THEY DEPEND ON THE TRANSITION FUNCTION AND THE POLICY:

            # coefficients for expensive to compute loss functions (all zero by default except for variance):
            "lossCoeff4Variance": 0, # weight of variance of total in loss function, must be >= 0
            "lossCoeff4Fourth": 0, # weight of centralized fourth moment of total in loss function, must be >= 0
            "lossCoeff4Cup": 0, # weight of "cup" loss component, based on sixth moment of total, must be >= 0
            "lossCoeff4LRA": 0, # weight of deviation of LRA from 0.5 in loss function, must be >= 0
            "lossCoeff4Time": 0, # weight of time in loss function, must be >= 0
            "lossCoeff4DeltaVariation": 0, # weight of variation of Delta in loss function, must be >= 0
            "lossCoeff4WassersteinTerminalState": 0, # weight of Wasserstein distance to default terminal state distribution in loss function, must be >= 0
            "lossCoeff4Entropy": 0, # weight of action entropy in loss function, must be >= 0
            "lossCoeff4KLdiv": 0, # weight of KL divergence in loss function, must be >= 0
            "lossCoeff4TrajectoryEntropy": 0, # weight of trajectory entropy in loss function, must be >= 0
            "lossCoeff4StateDistance": 0, # weight of distance of terminal state from reference state in loss function, must be >= 0
            "lossCoeff4Causation": 0, # weight of causation in loss function, must be >= 0
            "lossCoeff4CausationPotential": 0, # weight of causation potential in loss function, must be >= 0
            "lossCoeff4OtherLoss": 0, # weight of other loss components specified by otherLossIncrement, must be >= 0
            "allowNegativeCoeffs": False, # if true, allow negative loss coefficients

            "varianceOfDelta": (lambda state, action: 0),
            "skewnessOfDelta": (lambda state, action: 0),
            "excessKurtosisOfDelta": (lambda state, action: 0),
            "fifthMomentOfDelta": (lambda state, action: 8 * self.params["varianceOfDelta"](state, action) ** 2.5), # assumes a Gaussian distribution
            "sixthMomentOfDelta": (lambda state, action: 15 * self.params["varianceOfDelta"](state, action) ** 3), # assumes a Gaussian distribution

            "delta_dim": 1, # dimension of delta space (1 for scalar delta, >1 for vector delta)
            "ref_dirs": None, # reference directions in delta space if delta_dim >1, given as an np.array with delta_dim+1 many rows containing coefficient vectors

            "debug": None,
            "verbose" : None
        }

        self.params = defaults.copy()
        self.params.update(params)
        # TODO do I need to add params_.options

        self.stateActionPairsSet = set()

        assert self.params["lossTemperature"] > 0, "lossTemperature must be > 0"
        #assert 0 <= rescaling4Actions <= 1, "rescaling4Actions must be in 0...1"
        #assert 0 <= rescaling4Successors <= 1, "rescaling4Successors must be in 0...1"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4Random"] >= 0, "lossCoeff4random must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4FeasibilityPower"] >= 0, "lossCoeff4FeasibilityPower must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4DP"] >= 0, "lossCoeff4DP must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4AgencyChange"] >= 0, "lossCoeff4AgencyChange must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4LRA1"] >= 0, "lossCoeff4LRA1 must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4Time1"] >= 0, "lossCoeff4Time1 must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4Entropy1"] >= 0, "lossCoeff4Entropy1 must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4KLdiv1"] >= 0, "lossCoeff4KLdiv1 must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4Variance"] >= 0, "lossCoeff4variance must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4Fourth"] >= 0, "lossCoeff4Fourth must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4Cup"] >= 0, "lossCoeff4Cup must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4LRA"] >= 0, "lossCoeff4LRA must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4Time"] >= 0, "lossCoeff4time must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4DeltaVariation"] >= 0, "lossCoeff4DeltaVariation must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4WassersteinTerminalState"] >= 0, "lossCoeff4WassersteinTerminalState must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4Entropy"] >= 0, "lossCoeff4entropy must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4KLdiv"] >= 0, "lossCoeff4KLdiv must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4TrajectoryEntropy"] >= 0, "lossCoeff4TrajectoryEntropy must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4StateDistance"] >= 0, "lossCoeff4StateDistance must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4Causation"] >= 0, "lossCoeff4Causation must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4CausationPotential"] >= 0, "lossCoeff4CausationPotential must be >= 0"
        assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4OtherLoss"]     >= 0, "lossCoeff4OtherLoss must be >= 0"

        if not "defaultPolicy" in self.params:
            self.params["defaultPolicy"] = self.world.default_policy
        assert self.params["lossCoeff4Entropy"] == 0 or self.params["lossCoeff4DP"] == 0 or ("uninformedPolicy" in self.params), "uninformedPolicy must be provided if lossCoeff4DP > 0 or lossCoeff4Entropy > 0"
        assert self.params["lossCoeff4StateDistance"] == 0 or ("referenceState" in self.params), "referenceState must be provided if lossCoeff4StateDistance > 0"

        assert isinstance(self.params["delta_dim"], int) and self.params["delta_dim"] > 0, "delta_dim must be a positive integer"
        self.dim = self.params["delta_dim"]
        ref_dirs = self.params["ref_dirs"]
        if ref_dirs is not None:
            ref_dirs = np.array(ref_dirs)
            assert ref_dirs.shape == (self.dim+1, self.dim), f"ref_dirs must be a numpy array with shape (delta_dim+1, delta_dim) but is {ref_dirs}"
        self.ref_dirs = nested_tuple(ref_dirs)

        self.debug = DEBUG if self.params["debug"] is None else self.params["debug"]
        self.verbose = VERBOSE if self.params["verbose"] is None else self.params["verbose"] 

        self.seen_state_alephs = set()
        self.seen_action_alephs = set()

        if self.verbose or self.debug:
            print("makeMDPAgentSatisfia with parameters", self.params)

        def deltaVar(s, a, al4s, al4a, p):
            if not(q_ones := self.Q_ones(s,a,al4a)):
                return 0
            return self.Q_DeltaSquare(s, a, al4a) / q_ones - self.Q2(s, a, al4a) / (q_ones ** 2)

        self.lossesDef = {
            "lossCoeff4Random":  lambda s, a, al4s, al4a, p: self.randomTieBreaker(s, a),
            "lossCoeff4FeasibilityPower":  lambda s, a, al4s, al4a, p: (
                self.maxAdmissibleQ(s, a) - self.minAdmissibleQ(s, a)
            )
            ** 2,
            "lossCoeff4DP":  lambda s, a, al4s, al4a, p: self.disorderingPotential_action(s, a),
            "lossCoeff4AgencyChange":  lambda s, a, al4s, al4a, p: self.agencyChange_action(s, a),
            "lossCoeff4LRA1":  lambda s, a, al4s, al4a, p: self.LRAdev_action(s, a, al4a, True),
            "lossCoeff4Entropy1":  lambda s, a, al4s, al4a, p: self.behaviorEntropy_action(s, p, a),
            "lossCoeff4KLdiv1":  lambda s, a, al4s, al4a, p: self.behaviorKLdiv_action(s, p, a),
            # moment-based criteria:
            # (To compute expected powers of deviation from V(s), we cannot use the actual V(s)
            # because we don't know the local policy at s yet. Hence we use a simple estimate based on aleph4state)
            "lossCoeff4Variance":  lambda s, a, al4s, al4a, p: self.relativeQ2(s, a, al4a, midpoint(al4s)),
            "lossCoeff4Fourth":  lambda s, a, al4s, al4a, p: self.relativeQ4(s, a, al4a, midpoint(al4s)),
            "lossCoeff4Cup":  lambda s, a, al4s, al4a, p: self.cupLoss_action(s, a, al4s, al4a),
            "lossCoeff4LRA":  lambda s, a, al4s, al4a, p: self.LRAdev_action(s, a, al4a),
            "lossCoeff4Time":  lambda s, a, al4s, al4a, p: self.Q_ones(s, a, al4a),
            "lossCoeff4DeltaVariation":  deltaVar,
            # change-related criteria:
            "lossCoeff4WassersteinTerminalState":  lambda s, a, al4s, al4a, p: self.wassersteinTerminalState_action(
                s, a, al4a
            ),
            # randomization-related criteria:
            "lossCoeff4Entropy":  lambda s, a, al4s, al4a, p: self.behaviorEntropy_action(s, p, a, al4a),
            "lossCoeff4KLdiv":  lambda s, a, al4s, al4a, p: self.behaviorKLdiv_action(s, p, a, al4a),
            "lossCoeff4TrajectoryEntropy":  lambda s, a, al4s, al4a, p: self.trajectoryEntropy_action(s, p, a, al4a),
            "lossCoeff4StateDistance":  lambda s, a, al4s, al4a, p: self.stateDistance_action(s, a, al4a),
            "lossCoeff4Causation":  lambda s, a, al4s, al4a, p: self.causation_action(s, al4s, a, al4a),
            "lossCoeff4CausationPotential":  lambda s, a, al4s, al4a, p: self.causationPotential_action(
                s, al4s, a, al4a
            ),
            "lossCoeff4OtherLoss": lambda s, a, al4s, al4a, p:  self.otherLoss_action(s, a, al4a) if "otherLocalLoss" in self.params else 0
        }

        self.losses = { k: (self.params[k], v) for k, v in self.lossesDef.items() if self.params[k] }

    """The dependency/callback graph of the following functions is partially recursive 
        and involves aggregation (MIN, MAX, E) operations as follows:
    
        simulate
        → localPolicy
            → aspiration4state
            → minAdmissibleV, maxAdmissibleV
                → MIN(minAdmissibleQ), MAX(maxAdmissibleQ)
                → E(minAdmissibleV), E(maxAdmissibleV) (RECURSION)
            → estAspiration4action
            → minAdmissibleV, maxAdmissibleV, minAdmissibleQ, maxAdmissibleQ
            → combinedLoss
            → Q, ..., Q6, Q_DeltaSquare, Q_ones etc.
                → propagateAspiration (see below)
                → E(V), ..., E(V6), E(V_DeltaSquare), E(V_ones) etc.
                    → localPolicy (RECURSION)
                    → E(Q), ..., E(Q6), E(Q_DeltaSquare), E(Q_ones) etc. (RECURSION)
            → otherLoss_action
                → propagateAspiration (see below)
                → E(otherLoss_state)
                → localPolicy (RECURSION)
                → E(otherLoss_action) (RECURSION)
            → similarly for other loss components (RECURSION)
        → world.raw_moment_of_delta, varianceOfDelta, transition
        → propagateAspiration
            → aspiration4state
        → simulate (RECURSION)"""

    def __getitem__(self, name):
        return self.params[name]

    @cache
    def maxAdmissibleV(self, state, coeffs=None): # recursive
        """The maximum admissible expected Total in state (if dim==1 and coeffs not given)
        or the extremal admissible expected Total in state when maximizing 
        the given linear combination of its components (if coeffs given)."""
        if self.dim == 1 and coeffs is None:
            if self.verbose or self.debug:
                print(pad(state), "| | | maxAdmissibleV, state", state, "...")

            v = 0
            actions = self.possible_actions(state)
            if actions != []:
                qs = [self.maxAdmissibleQ(state, a) for a in actions] # recursion
                v = max(qs) if self["maxLambda"] == 1 else interpolate(min(qs), self["maxLambda"], max(qs))

            if self.verbose or self.debug:
                print(pad(state), "| | | ╰ maxAdmissibleV, state", state, ":", v)
        else:
            assert (self["minLambda"], self["maxLambda"]) == (0, 1), "minLambda,maxLambda must be 0,1 for dim > 1"
            if self.verbose or self.debug:
                print(pad(state), "| | | maxAdmissibleV, state", state, "coeffs", coeffs, "...")

            v = np.zeros(self.dim)
            actions = self.possible_actions(state)
            if actions != []:
                qs = np.array([self.maxAdmissibleQ(state, a, coeffs) for a in actions]) # recursion
                amax = np.argmax(np.dot(qs, coeffs))  # action that maximizes given linear comb. of components of Total
                v = qs[amax]

            if self.verbose or self.debug:
                print(pad(state), "| | | ╰ maxAdmissibleV, state", state, "coeffs", coeffs, ":", v)

        return v

    @cache
    def minAdmissibleV(self, state): # recursive
        """The minimum admissible expected Total in state (if dim==1, otherwise unused)"""
        if self.verbose or self.debug:
            print(pad(state), "| | | minAdmissibleV, state", state, "...")

        v = 0
        actions = self.possible_actions(state)
        if actions != []:
            qs = [self.minAdmissibleQ(state, a) for a in actions] # recursion
            v = min(qs) if self["minLambda"] == 0 else interpolate(min(qs), self["minLambda"], max(qs))

        if self.verbose or self.debug:
            print(pad(state), "| | | ╰ minAdmissibleV, state", state, ":", v)

        return v

    # The resulting admissibility interval for states.
#    def admissibility4state(self, state):
#        """The admissibility interval for expected Total in state (only used if dim==1) """
#        return self.minAdmissibleV(state), self.maxAdmissibleV(state)
    
    @cache
    def simplex4state(self, state):
        """The admissibility simplex (if dim>1) for expected Total in state"""
        vertices = np.array([self.maxAdmissibleV(state, coeffs) for coeffs in self.ref_dirs])
        print("vertices",vertices)
        verts = [vertices[0]]
        for v in vertices[1:]:
            if not np.any([np.allclose(v,vprime) for vprime in verts]):
                verts.append(v)
        verts = np.array(verts)
        return verts, polytope.qhull(verts)

    # The resulting admissibility interval for actions.
    def admissibility4action(self, state, action):
        """The admissibility interval for expected Total in state at action"""
        return self.minAdmissibleQ(state, action), self.maxAdmissibleQ(state, action)

    # The resulting reference simplex for actions.
    @cache
    def simplex4action(self, state, action):
        """The admissibility simplex (if dim>1) for expected Total in state at action"""
        vertices = np.array([self.maxAdmissibleQ(state, action, coeffs) for coeffs in self.ref_dirs])
        # calculate the H-representation of the simplex defined by these vertices: 
        print("vertices",vertices)
        verts = [vertices[0]]
        for v in vertices[1:]:
            if not np.any([np.allclose(v,vprime) for vprime in verts]):
                verts.append(v)
        verts = np.array(verts)
        return verts, polytope.qhull(verts)

    # When in state, we can get any expected total in the interval
    # [minAdmissibleV(state), maxAdmissibleV(state)] (or the admissible simplex if dim>1).
    # So when having aspiration aleph, we can still fulfill it in expectation if it lies in the interval.
    # Therefore, when in state at incoming aspiration aleph,
    # we adjust our aspiration to aleph clipped to that interval 
    # (if dim==1, otherwise we require it to be a subset already):  # TODO: also clip when dim>1
    @cache
    def aspiration4state(self, state, unclippedAleph):
        if self.verbose or self.debug:
            print(pad(state),"| | aspiration4state, state",prettyState(state),"unclippedAleph",unclippedAleph,"...")
        res = clip2(self.minAdmissibleV(state), Interval(unclippedAleph), self.maxAdmissibleV(state))
        if self.verbose or self.debug:
            print(pad(state),"| | ╰ aspiration4state, state",prettyState(state),"unclippedAleph",unclippedAleph,":",res)
        return res

    @cache 
    def rmax_schedule(self, state):
        tmax = self.world.max_episode_length
        t = state[0]
        return ((tmax - t - 1) / (tmax - 1))**(1/self.dim)

    # When constructing the local policy, we use an action aspiration interval
    # that does not depend on the local policy but is simply based on the state's aspiration interval,
    # moved from the admissibility interval (or simplex) of the state to the admissibility interval (or simplex) of the action and properly restricted to a subset of it.
    #@cache
    def aspiration4action(self, state, action, aleph4state, 
                          dir_index=None  # if dim>1, index (0...dim+1) of direction in which to shift
                          ):
        """Set an action aspiration interval or simplex based on the state aspiration and the admissibility interval or simplex of the state and action, and if dim>1 also based on the direction in which to shift the state's aspiration. Return the resulting action aspiration interval or simplex, and if dim>1 also the shifting vector so that the probability distribution can be computed."""
        if self.dim == 1:
            if self.debug:
                print(pad(state),"| | aspiration4action, state",prettyState(state),"action",action,"aleph4state",aleph4state,"...")

            phi = self.admissibility4action(state, action)

            # We use a steadfast version that does make sure that 
            # - aleph(a) is no wider than aleph(s)
            # - one can mix the midpoint of aleph(s) from midpoints of alephs(a)
            # - hence one can mix an interval inside aleph(s) from alephs(a).
            # The rule finds the largest subinterval of phi(a) (the admissibility interval of a)
            # that is no wider than aleph(s) and is closest to aleph(s).
            # More precisely:
            # - If phi(a) contains aleph(s), then aleph(a) = aleph(s)
            # - If aleph(s) contains phi(a), then aleph(a) = phi(a)
            # - If phiLo(a) < alephLo(s) and phiHi(a) < alephHi(s), then aleph(a) = [max(phiLo(a), phiHi(a) - alephW(s)), phiHi(a)]
            # - If phiHi(a) > alephHi(s) and phiLo(a) > alephLo(s), then aleph(a) = [phiLo(a), min(phiHi(a), phiLo(a) + alephW(s))]

            if isSubsetOf(aleph4state, phi):  # case (1)
                res = aleph4state
                # as a consequence, midpoint(res) = midpoint(aleph4state)
            elif isSubsetOf(phi, aleph4state):  # case (2)
                res = phi
                # this case has no guarantee for the relationship between midpoint(res) and midpoint(aleph4state),
                # but that's fine since there will always either be an action with case (1) above,
                # or both an action with case (3) and another with case (4) below,
                # so that the midpoint of aleph4state can always be mixed from midpoints of alephs4action
            else:
                phiLo, phiHi = phi
                alephLo, alephHi = aleph4state
                w = alephHi - alephLo
                if phiLo < alephLo and phiHi < alephHi:  # case (3)
                    res = Interval(max(phiLo, phiHi - w), phiHi)
                    # as a consequence, midpoint(res) < midpoint(aleph4state)
                elif phiHi > alephHi and phiLo > alephLo:  # case (4)
                    res = Interval(phiLo, min(phiHi, phiLo + w))
                    # as a consequence, midpoint(res) > midpoint(aleph4state)
                else:
                    raise ValueError("impossible relationship between phi and aleph4state")

            # memorize that we encountered this state, action, aleph4action:
            self.seen_action_alephs.add((state, action, res))

            if self.verbose or self.debug:
                print(pad(state),"| | ╰ aspiration4action, state",prettyState(state),"action",action,"aleph4state",aleph4state,":",res,"(steadfast)") 

        else: # follow the "shrink" strategy from the ADT24 paper

            # dirty fix: round aleph4state to multiples of 1e-10:
            aleph4state = nested_tuple(np.round(aleph4state, 10))

            if self.debug:
                print(pad(state),"| | aspiration4action, state",prettyState(state),"action",action,"aleph4state",aleph4state,"dir_index",dir_index,"...")

            Es = aleph4state
            Es_center, Es_shape = center_and_shape(Es)
            flat_Es_shape = Es_shape.flatten()
            Es_n = Es_shape.shape[0]

            VR, _ = self.simplex4state(state)  # vertices
            QRa_vertices, QRa_poly = self.simplex4action(state, action) 
            QRa_A = QRa_poly.A
            QRa_b = QRa_poly.b

            if self.plot:
                if self.offset is None:
                    self.offset = 0 * Es_center 
                self.history.append(self.offset + np.mean(Es,axis=0))
                # plot history of aspirations:
                if len(self.history) > 1:
                    for i,Ecenter in enumerate(self.history[:-1]):
                        Ecenter2 = self.history[i+1]
                        plt.plot([Ecenter[0], Ecenter2[0]],
                                 [Ecenter[1], Ecenter2[1]],"r-", lw=3)

                # plot Es:
                if len(Es) > 1:
                    plt.plot([self.offset[0] + Es[i][0] for i in list(range(len(Es)))+[0]],
                            [self.offset[1] + Es[i][1] for i in list(range(len(Es)))+[0]],"r-", lw=3)
                else:
                    plt.plot([self.offset[0] + Es[i][0] for i in list(range(len(Es)))+[0]],
                            [self.offset[1] + Es[i][1] for i in list(range(len(Es)))+[0]],"r.", ms=20)
                # plot VR:
                plt.plot([self.offset[0] + VR[i][0] for i in list(range(len(VR)))+[0]],
                         [self.offset[1] + VR[i][1] for i in list(range(len(VR)))+[0]],"k--", lw=2)
                # plot QRa:
                plt.plot([self.offset[0] + QRa_vertices[i][0] for i in list(range(len(QRa_vertices)))+[0]],
                         [self.offset[1] + QRa_vertices[i][1] for i in list(range(len(QRa_vertices)))+[0]],"b-", lw=1)

            # We are shifting the Es towards a certain target point:
            # For dir_index>0, the respective vertex of the state's reference simplex,
            # otherwise the center of the action's reference simplex:
            target = VR[dir_index-1] if dir_index > 0 and dir_index-1 < VR.shape[0] else QRa_vertices.mean(axis=0)
            shifting_direction = target - Es_center
            #print("ind",dir_index,"dir",shifting_direction)

            if self.plot: 
                # plot target and direction:
                plt.plot([self.offset[0] + Es_center[0], self.offset[0] + target[0]],
                         [self.offset[1] + Es_center[1], self.offset[1] + target[1]], "k:", lw=1)


            if QRa_A.size > 0:
                if Es_n > 1:
                    # Find the largest r <= 1 so that there is an l >= 0 with
                    # l*shifting_direction + Es_center + r*Es_shape is a subset of QRa.
                    # The inequality constraints are thus
                    #    QRa_poly.A @ row.T <= QRa_poly.b
                    # for all rows in l*shifting_direction + Es_center + r*Es_shape. 
                    # In terms of the two variables (l,r), the (d+1)² many constraints are:
                    #    QRa_poly.A @ (shifting_direction.T, Es_shape[i].T) @ (l,r) 
                    #    <= QRa_poly.b - QRa_poly.A @ Es_center
                    # for i=0...d.
                    # If the variable vector is (l,r), the arguments for linprog are thus:
                    c = [0,-1]  # minimize -r
                    #print(QRa_b, QRa_A, Es_center)
                    b_ub = np.vstack([QRa_b - QRa_A @ Es_center for i in range(Es_n)]).flatten()  # d+1 repetitions of the vector
                    A_ub = np.concatenate([
                        np.vstack([QRa_A for i in range(Es_n)]) @ shifting_direction.reshape((-1,1)),
                        (Es_shape @ QRa_A.T).flatten().reshape(-1,1)
                    ], axis=1)
                    rmax = self.rmax_schedule(state)
                    print("A",A_ub,"\nb",b_ub,"\nc",c,"\ncheck:", A_ub @ np.array([0,0]) - b_ub)
                    linprogres = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0,1e10),(0,rmax)])
                    if self.debug: print("linprog for l,r:", linprogres)
                    if not linprogres.success:
                        # action is not in directional action set
                        if self.debug: print(linprogres)
                        #print("ta",target,QRa_vertices)
                        if target in QRa_vertices:
                            print("OOOPS!", dir_index)
                            return nested_tuple(shifting_direction), nested_tuple(shifting_direction + Es_center + 0 * Es_shape)  
                        return None, None
                    _, r = linprogres.x            
                    b_ub -= np.dot(A_ub[:,1], r)
                    A_ub = A_ub[:,[0]]
                else:
                    r = 0
                    A_ub = QRa_A @ shifting_direction.reshape((-1,1))
                    b_ub = QRa_b - QRa_A @ Es_center
                    #print("A",A_ub,"b",b_ub)
                # Now find the smallest l >= 0 for this r:
                linprogres = linprog(1, A_ub=A_ub, b_ub=b_ub, bounds=(0,None), #options=linprog_options
                                     )
                if self.debug: print("linprog for l:", linprogres)
                if not linprogres.success:
                    # action is not in directional action set
                    if self.debug: print(linprogres)
                    #print("ta",target,QRa_vertices)
                    if target in QRa_vertices:
                        print("OOOPS 1!", dir_index)
                        return nested_tuple(shifting_direction), nested_tuple(shifting_direction + Es_center + 0 * Es_shape)  
                    return None, None
                l = linprogres.x
            else:
                # QRa is a singleton q, so we have r=0 and can compute l directly so that 
                # l*shifting_direction + Es_center = q
                r = 0
                sdn = np.linalg.norm(shifting_direction)
                l = 0 if sdn==0 else (QRa_vertices[0] - Es_center) @ shifting_direction / sdn**2
                if l < 0 or np.linalg.norm(l*shifting_direction + Es_center - QRa_vertices[0]) > 1e-5:
                    # action is not in directional action set
                    if self.debug: print(Es_center, shifting_direction, QRa_vertices)
                    #print("ta",target,QRa_vertices)
                    if target in QRa_vertices:
                        print("OOOPS 2!", dir_index)
                        return nested_tuple(shifting_direction), nested_tuple(shifting_direction + Es_center + 0 * Es_shape)  
                    return None, None

            shift = min(1,l+1e-5) * shifting_direction
            Esa = nested_tuple(shift + Es_center + r * Es_shape)
            res = nested_tuple(shift), Esa

            if self.plot: 
                # plot res:
                if len(Esa) > 1:
                    plt.plot([self.offset[0] + Esa[i][0] for i in list(range(len(Esa)))+[0]],
                            [self.offset[1] + Esa[i][1] for i in list(range(len(Esa)))+[0]],"g-", lw=1.5)
                else:
                    plt.plot([self.offset[0] + Esa[i][0] for i in list(range(len(Esa)))+[0]],
                            [self.offset[1] + Esa[i][1] for i in list(range(len(Esa)))+[0]],"g.", ms=10)

            # memorize that we encountered this state, action, dir_index, res:
            self.seen_action_alephs.add((state, action, dir_index, res))

            if self.verbose or self.debug:
                print(pad(state),"| | ╰ aspiration4action, state",prettyState(state),"action",action,"aleph4state",aleph4state,"dir_index",dir_index,":",res,f"(l={l},r={r})") 

        return res

    @cache
    def disorderingPotential_state(self, state): # recursive
        if self.debug or self.verbose:
            print(pad(state),"| | disorderingPotential_state", prettyState(state), "...")
        actions = self.possible_actions(state)
        maxMPpolicyWeights = [math.exp(self.disorderingPotential_action(state, a)) for a in actions]
        if sum(maxMPpolicyWeights) == 0: return 0 # TODO this shouldn't be 0
        res = math.log(sum(maxMPpolicyWeights))
        if self.debug or self.verbose:
            print(pad(state),"| | ╰ disorderingPotential_state", prettyState(state), ":", res)
        return res

    @cache
    def _compute_default_transition(self, state):
        if not self.reachable_states:
            self.reachable_states = list(self.world.reachable_states(state))
            if self.debug or self.verbose:
                print("no. of reachable states:", len(self.reachable_states))
        scores = self.params["defaultTransitionScore"]
        def default_transition(source):
            the_scores = scores.get(source, {})
            targets = list(the_scores.keys())
            if len(targets) == 0:  # use a uniform distribution over all possible successors
                targets = self.world.possible_successors(source)
                return distribution.categorical(list(targets), [1 for target in targets])
            else:
                return distribution.categorical(targets, [math.exp(the_scores[target]) for target in targets])
        self.default_transition = default_transition

    @cache
    def agency_state(self, state): # recursive
        if self.debug or self.verbose:
            print(pad(state),"| | | | agency_state", prettyState(state), "...")
        if self.world.is_terminal(state):
            res = 0
        else:
            if not self.default_transition:
                self._compute_default_transition(state)
            actions = self.possible_actions(state)
            def X(other_state):
                aps = [(a, self.world.transition_probability(state, a, other_state)[0]) for a in actions]
                if max([p for (a, p) in aps]) > 0:  # other_state is possible successor 
                    next_agency = self.agency_state(other_state)
                    return max([math.sqrt(p) + p * next_agency for (a, p) in aps])
                else:
                    return 0
            res = self.default_transition(state).expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"| | | | ╰ agency_state", prettyState(state), ":", res)
        return res

    # Based on the admissibility information computed above, we can now construct the policy,
    # which is a mapping taking a state and an aspiration interval as input and returning
    # a categorical distribution over (action, aleph4action) pairs.

    def localPolicy(self, state, aleph, plot=False): # recursive
        """return a categorical distribution over (action, aleph4action) pairs"""

        self.plot=plot

#        aleph = Interval(aleph)
        d = self.localPolicyData(state, aleph)
        support = [(a, #Interval(
                    al
                    #)
                    ) for a, al in d[0]]
        ps = d[1]

        if self.debug or self.verbose:
            print(pad(state),"| ╰ localPolicy", prettyState(state), aleph, d)

        return distribution.categorical(support, ps)

    #@cache
    def localPolicyData(self, state, aleph):
        if self.verbose or self.debug:
            print(pad(state), "| localPolicyData, state",prettyState(state),"aleph",aleph,"...")

        # memorize that we encountered this state, aleph:
        self.seen_state_alephs.add((state, aleph))

        actions = self.possible_actions(state)
        assert actions != []

        # Estimate losses based on this estimated aspiration intervals
        # and use it to construct softmin propensities (probability weights) for choosing actions.
        # since we don't know the actual probabilities of actions yet (we will determine those only later),
        # but the loss estimation requires an estimate of the probability of the chosen action,
        # we estimate the probability at 1 / number of actions:
        def getPropensities(actions, estAlephs):
            p = 1 / len(actions)
            losses = [self.combinedLoss(state, action, aleph, estAlephs[i], p) for i, action in enumerate(actions)] # bottleneck
            min_loss = min(losses)
            return [max(math.exp(-(loss - min_loss) / self["lossTemperature"]), 1e-100) for loss in losses]

        p_effective = {}

        if self.dim == 1:
            # Clip aspiration interval to admissibility interval of state:
            alephLo, alephHi = aleph4state = self.aspiration4state(state, aleph)

            # Estimate aspiration intervals for all possible actions in a way
            # independent from the local policy that we are about to construct,
            alephs = [self.aspiration4action(state, action, aleph4state) for action in actions]

            indices = list(range(len(actions)))
            candidate_probs = getPropensities([actions[i] for i in indices], 
                                           [alephs[i] for i in indices])

            if self.debug:
                print(pad(state),"| localPolicyData", prettyState(state), aleph, actions, candidate_probs)

            for i1, p1 in distribution.categorical(indices, candidate_probs).categories():
                # Get admissibility interval for the first action.
                a1 = actions[i1]
                adm1 = self.admissibility4action(state, a1)

                # If a1's admissibility interval is completely contained in aleph4state, we are done:
                if Interval(adm1) <= Interval(aleph4state):
                    if self.verbose or self.debug:
                        print(pad(state),"| localPolicyData, state",prettyState(state),"aleph4state",aleph4state,": a1",a1,"adm1",adm1,"(subset of aleph4state)")
                    probability_add(p_effective, (a1, adm1), p1)
                else:
                    # For the second action, restrict actions so that the the midpoint of aleph4state can be mixed from
                    # those of aleph4action of the first and second action:
                    midTarget = midpoint(aleph4state)
                    aleph1 = alephs[i1]
                    mid1 = midpoint(aleph1)
                    indices2 = [index for index in indices if between(midTarget, midpoint(alephs[index]), mid1)]
                    if len(indices2) == 0:
                        print("OOPS: indices2 is empty", a1, adm1, aleph4state, midTarget, aleph1, mid1, alephs)
                        indices2 = indices
                    propensities2 = getPropensities([actions[i] for i in indices2], 
                                                    [alephs[i] for i in indices2])

                    for i2, p2 in distribution.categorical(indices2, propensities2).categories():
                        # Get admissibility interval for the second action.
                        a2 = actions[i2]
                        adm2 = self.admissibility4action(state, a2)
                        aleph2 = alephs[i2]
                        mid2 = midpoint(aleph2)
                        w = relativePosition(mid1, midTarget, mid2)
                        if w < 0 or w > 1:
                            print("OOPS: p", w)
                            w = clip(0, w, 1)

                        if self.verbose or self.debug:
                            print(pad(state),"| localPolicyData, state",prettyState(state),"aleph4state",aleph4state,": a1,p,a2",a1,w,a2,"adm12",adm1,adm2,"aleph12",aleph1,aleph2)

                        probability_add(p_effective, (a1, aleph1), (1 - w) * p1 * p2)
                        probability_add(p_effective, (a2, aleph2), w * p1 * p2)

        else: # use algorithm from ADT24 paper

            if self.plot: 
                plt.cla()
                plt.xlim(-7, 7)
                plt.ylim(-7, 7)

            if self.ref_dirs is None:
                self.find_ref_dirs(aleph)  # TODO!

            d = self.dim
            Ais = []
            zais = []
            mean_zais = []
            Eais = []
            candidate_probs = []

            # calculate directional actions sets Ai, shifting vectors zai and action aspirations Eai:
            for dir_index in range(d+2):

                Ai = []
                this_zais = []
                this_Eais = []
                for action in actions:
                    zai, Eai = self.aspiration4action(state, action, aleph, dir_index=dir_index)
                    if zai is not None:
                        Ai.append(action)
                        this_zais.append(zai)
                        this_Eais.append(Eai)

                if len(Ai) == 0 and self.plot:
                    plt.title("no action!")
                    plt.show()
                assert len(Ai) > 0, "no action with dir_index " + str(dir_index) + " for state " + str(state) + " and aleph " + str(aleph)

                Ais.append(Ai)
                zais.append(this_zais)
                Eais.append(this_Eais)
                w = np.array(getPropensities(Ai, this_Eais))
                w /= sum(w)
                candidate_probs.append(w)
                mean_zais.append(sum([w[i] * np.array(this_zais[i]) for i in range(len(Ai))]))

            # TODO: adjust the following so that instead of matching the center, only the subset conditions are met:

            # find mix that satisfies state aspiration and has largest probability for freely chosen action:
            c = [-1] + [0] * (d+1)  # maximize the probability of the freely chosen action
            A_eq = np.concatenate([np.array(mean_zais).T,np.repeat([[1]], d+2, axis=1)],axis=0)
            b_eq = [0] * d + [1]
            if self.debug: print("Es",aleph, "A",Ais, "E",Eais, "p",candidate_probs, "z",mean_zais, "c",c, "A",A_eq)
            linprogres = linprog(c, A_ub=None, b_ub=None, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
            if self.debug: print("linprog for p:", linprogres)
            if not linprogres.success:
                plt.title("linprog for p failed "+str(linprogres))
                #plt.show()
                raise ValueError("linprog for p failed", linprogres)
            #print("p",linprogres.x)
            direction_probs = np.maximum(0,linprogres.x)
            direction_probs /= direction_probs.sum()

            E_effective = np.array(Eais[0][0])*0
            # register action,aspiration probabilities:
            for dir_index in range(d+2):
                dp = direction_probs[dir_index]
                cps = candidate_probs[dir_index]
                this_Eais = Eais[dir_index]
                for index, action in enumerate(Ais[dir_index]):
                    try:
                        this_p = dp * cps[index]
                        probability_add(p_effective, (action, this_Eais[index]), this_p)
                        #print(E_effective,this_p,this_Eais[index])
                        E_effective += this_p * np.array(this_Eais[index])
                    except:
                        print(state, aleph, action, this_Eais[index], p_effective, dp * cps[index])
                        raise
            #print(p_effective,E_effective,aleph)
            if self.plot:
                plt.title(f"{state}") 
                #plt.show()
                plt.pause(0.1)
                #plt.close()
            #assert ((E_effective - np.array(aleph))**2).sum()<1e-10


        # now we can construct the local policy as a distribution object:
        locPol = distribution.categorical(p_effective)

        support = locPol.support()
        ps = [max(1e-100, locPol.probability(item)) for item in support] # 1e-100 prevents normalization problems

        if self.verbose or self.debug:
            print(pad(state),"| localPolicy, state",prettyState(state),"aleph",aleph,":")
            #_W.printPolicy(pad(state), support, ps)
        return [support, ps]

    # Propagate aspiration from state-action to successor state, potentially taking into account received expected delta:

    # caching this easy to compute function would only clutter the cache due to its many arguments
    def propagateAspiration(self, state, action, aleph4action, Edel, nextState):
        if self.debug:
            print(pad(state),"| | | | | | propagateAspiration, state",prettyState(state),"action",action,"aleph4action",aleph4action,"Edel",Edel,"nextState",prettyState(nextState),"...")

        if self.dim == 1:
            # compute the relative position of aleph4action in the expectation that we had of 
            #    delta + next admissibility interval 
            # before we knew which state we would land in:
            lam = relativePosition2(self.minAdmissibleQ(state, action), aleph4action, self.maxAdmissibleQ(state, action)) # TODO didn't we calculate the admissible Q when we chose the action?
            # (this is two numbers between 0 and 1.)
            # use it to rescale aleph4action to the admissibility interval of the state that we landed in:
            rescaledAleph4nextState = interpolate2(self.minAdmissibleV(nextState), lam, self.maxAdmissibleV(nextState))
            # (only this part preserves aspiration in expectation)
            res = rescaledAleph4nextState # WAS: interpolate(steadfastAleph4nextState, rescaling4Successors, rescaledAleph4nextState)

            """ Note on influence of Edel: 
            It might seem that the (expected) delta received when taking action a in state s should occur
            explicitly in some form in this formula, similar to how it occurred in the steadfast formula above.
            This is not so, however. The expected delta is taken account of *implicitly* in the rescaling formula
            via the use of min/maxAdmissibleQ(s,a) to compute lam but using min/maxAdmissibleV(s') in interpolating.
            More precisely, one can prove that aspirations are kept in expectation. We want

                aleph(s,a) = E(delta(s,a)) + E(aleph(s') | s'~(s,a)).

            This can be shown to be true as follows:

                min/maxAdmissibleQ(s,a) = E(delta(s,a)) + E(min/maxAdmissibleV(s') | s'~(s,a)),

                lamdba = (aleph(s,a) - minAdmissibleQ(s,a)) / (maxAdmissibleQ(s,a) - minAdmissibleQ(s,a)),

                rescaledAleph(s') = minAdmissibleV(s') + lambda * (maxAdmissibleV(s') - minAdmissibleV(s')),

                E(delta(s,a)) + E(rescaledAleph(s') | s'~(s,a)) 
                = E(delta(s,a)) + E(minAdmissibleV(s') | s'~(s,a)) 
                    + lambda * (E(maxAdmissibleV(s') | s'~(s,a)) - E(minAdmissibleV(s') | s'~(s,a)))
                = minAdmissibleQ(s,a) + lambda * (maxAdmissibleQ(s,a) - minAdmissibleQ(s,a))
                = minAdmissibleQ(s,a) + (aleph(s,a) - minAdmissibleQ(s,a))
                = aleph(s,a).

            So the above rescaling formula is correctly taking account of received delta even without explicitly
            including Edel in the formula.
            """

        else:
            # Compute the center of Ea = aleph4action and its representation as a 
            # convex combinations of the vertices of QRa = simplex4action(state, action).
            # That is, solve the matric equation
            #   (QRa,1).T @ P = (Ea,1).T
            # where 1 is a column vector of 1s, using numpy.linalg.solve:
            Ea_center, Ea_shape = center_and_shape(aleph4action)
            Ea_n = Ea_shape.shape[0]
            flat_Ea_shape = Ea_shape.flatten()
            EaT = Ea_center.reshape((-1,1))
            QRa = self.simplex4action(state, action)[0]
            QRaT = np.transpose(QRa)
            VR_vertices, VR_poly = self.simplex4state(nextState)
            print("Ea:", Ea_center, "\n", Ea_shape, "\nQRa:", QRa, "\nVR:", VR_vertices)
            A = np.concatenate([QRaT, np.ones((1, QRaT.shape[1]))], axis=0)
            B = np.concatenate([EaT, [[1]]], axis=0)
            P = np.linalg.lstsq(A, B)[0]  # TODO: verify this is correct if A is singular!
            # Compute the new center as the corresponding convex combination 
            # of the vertices of VR = simplex4state(nextState):
            VR_A, VR_b = VR_poly.A, VR_poly.b
            new_center = np.dot(P.T, VR_vertices).flatten()

            # We are now shifting the Es towards new_center and then shrink it:
            if VR_A.size > 0:
                # Find the largest r <= 1 so that
                # shifting_vector + Ea_center + r*Ea_shape is a subset of VR.
                # The inequality constraints are thus
                #    VR_A @ row.T <= VR_b
                # for all rows in new_center + r*Es_shape. 
                # In terms of the variable r, the (d+1)² many constraints are:
                #    VR_A @ Ea_shape[i] * r <= VR_b - VR_A @ new_center
                # for i=0...d.
                # The arguments for linprog are thus:
                c = [-1]  # minimize -r
                #print(VR_b, VR_A, Ea_center, new_center)
                b_ub = np.vstack([VR_b - VR_A @ new_center for i in range(Ea_n)]).flatten()  # d+1 repetitions of the vector
                A_ub = (Ea_shape @ VR_A.T).flatten().reshape(-1,1)
                #print("A",A_ub,"b",b_ub,"c",c)
                linprogres = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0,1), #options=linprog_options
                                     )
                if self.debug: print("linprog for r:", linprogres)
                if not linprogres.success:
                    #raise ValueError("linprog for r failed", linprogres)
                    r = 0
                else:
                    r = linprogres.x            
            else:
                # VR is a singleton v, so we have r=0:
                r = 0

            res = nested_tuple(new_center + r * Ea_shape)

        if self.verbose or self.debug:
            print(pad(state),"| | | | | | ╰ propagateAspiration, state",prettyState(state),"action",action,"aleph4action",aleph4action,"Edel",Edel,"nextState",prettyState(nextState),":",res)
        return res

    @cache
    def V(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"V", prettyState(state), aleph4state, "...")

        def X(actionAndAleph):
            return self.Q(state, actionAndAleph[0], actionAndAleph[1]) # recursion
        v = self.localPolicy(state, aleph4state).expectation(X)

        if self.debug or self.verbose:
            print(pad(state),"╰ V", prettyState(state), aleph4state, ":", v)
        return v

    @cache
    def V2(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"V2", prettyState(state), aleph4state, "...")

        def X(actionAndAleph):
            return self.Q2(state, actionAndAleph[0], actionAndAleph[1]) # recursion
        v2 =self.localPolicy(state, aleph4state).expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ V2", prettyState(state), aleph4state, ":", v2)
        return v2

    @cache
    def V3(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"V3", prettyState(state), aleph4state, "...")

        def X(actionAndAleph):
            return self.Q3(state, actionAndAleph[0], actionAndAleph[1]) # recursion
        v3 = self.localPolicy(state, aleph4state).expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ V3", prettyState(state), aleph4state, v3)
        return v3

    @cache
    def V4(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"V4", prettyState(state), aleph4state, "...")

        def X(actionAndAleph):
            return self.Q4(state, actionAndAleph[0], actionAndAleph[1]) # recursion
        v4 = self.localPolicy(state, aleph4state).expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ V4", prettyState(state), aleph4state, v4)
        return v4

    @cache
    def V5(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"V5", prettyState(state), aleph4state, "...")

        def X(actionAndAleph):
            return self.Q5(state, actionAndAleph[0], actionAndAleph[1]) # recursion
        v5 = self.localPolicy(state, aleph4state).expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ V5", prettyState(state), aleph4state, v5)
        return v5

    @cache
    def V6(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"V6", prettyState(state), aleph4state, "...")

        def X(actionAndAleph):
            return self.Q6(state, actionAndAleph[0], actionAndAleph[1]) # recursion
        v6 = self.localPolicy(state, aleph4state).expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ V6", prettyState(state), aleph4state, v6)
        return v6

    # Expected powers of difference between total and some target value v,
    # needed for estimating moments of probabilistic policies in loss function,
    # where v will be an estimate of V(state):

    #@cache
    def relativeQ2(self, s, a, al, v): # aleph4action
        if self.debug:
            print(pad(s),"| | | Q2", prettyState(s), a, al, v, "...")

        res = self.Q2(s,a,al) \
            - 2*self.Q(s,a,al)*v \
            + v ** 2
        if self.debug or self.verbose:
            print(pad(s),"| | | ╰ relativeQ2", prettyState(s), a, al, v, res)
        return res

    #@cache
    def relativeQ4(self, s, a, al, v):
        if self.debug:
            print(pad(s),"| | | relativeQ4", prettyState(s), a, al, v, "...")
        res = self.Q4(s,a,al) \
            - 4*self.Q3(s,a,al)*v \
            + 6*self.Q2(s,a,al)*(v ** 2) \
            - 4*self.Q(s,a,al)*(v ** 3) \
            + v ** 4
        if self.debug or self.verbose:
            print(pad(s),"| | | ╰ relativeQ4", prettyState(s), a, al, v, res)
        return res

    #@cache
    def relativeQ6(self, s, a, al, v):
        if self.debug:
            print(pad(s),"| | | relativeQ6", prettyState(s), a, al, v, "...")
        res = self.Q6(s,a,al) \
            - 6*self.Q5(s,a,al)*v \
            + 15*self.Q4(s,a,al)*(v ** 2) \
            - 20*self.Q3(s,a,al)*(v ** 3) \
            + 15*self.Q2(s,a,al)*(v ** 4) \
            - 6*self.Q(s,a,al)*(v ** 5) \
            + v ** 6
        if self.debug or self.verbose:
            print(pad(s),"| | | ╰ relativeQ6", prettyState(s), a, al, v, res)
        return res

    # TODO: the following should maybe better be w.r.t. the initial aspiration interval, not the current state's:

    # loss based on a "cup" shaped potential centered at the mid-point of the aspiration interval
    # that is almost completely flat in the middle half of the interval 
    # (https://www.wolframalpha.com/input?i=plot+%28x-.5%29%5E6+from+0+to+1):

    @cache
    def cupLoss_action(self, state, action, aleph4state, aleph4action):
        if self.debug:
            print(pad(state),"| | | cupLoss_action", prettyState(state), action, aleph4state, "...")
        res = self.relativeQ6(state, action, aleph4action, midpoint(aleph4state))
        if self.debug or self.verbose:
            print(pad(state),"| | | ╰ cupLoss_action", prettyState(state), action, aleph4state, ":", res)
        return res

    @cache
    def cupLoss_state(self, state, unclippedAleph): # recursive
        if self.debug:
            print(pad(state),"cupLoss_state", prettyState(state), aleph4state, "...")
        aleph4state = self.aspiration4state(state, unclippedAleph)
        def X(actionAndAleph):
            return self.cupLoss_action(state, actionAndAleph[0], aleph4state, actionAndAleph[1])
        res = self.localPolicy(state, aleph4state).expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ cupLoss_state", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def LRAdev_state(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"LRAdev_state", prettyState(state), aleph4state, "...")
        def X(actionAndAleph):
            return self.LRAdev_action(state, actionAndAleph[0], actionAndAleph[1]) # recursion
        res = self.localPolicy(state, aleph4state).expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ LRAdev_state", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def V_ones(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"V_ones", prettyState(state), aleph4state, "...")
        def X(actionAndAleph):
            return self.Q_ones(state, actionAndAleph[0], actionAndAleph[1]) # recursion
        res = self.localPolicy(state, aleph4state).expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ V_ones", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def V_DeltaSquare(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"V_DeltaSquare", prettyState(state), aleph4state, "...")
        def X(actionAndAleph):
            return self.Q_DeltaSquare(state, actionAndAleph[0], actionAndAleph[1]) # recursion
        vDsq = self.localPolicy(state, aleph4state).expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ V_DeltaSquare", prettyState(state), aleph4state, vDsq)
        return vDsq

    @cache
    def ETerminalState_state(self, state, aleph4state, policy="actual"): # recursive
        """expected value of (vector-embedded) terminal state"""
        if self.debug:
            print(pad(state),"ETerminalState_state", prettyState(state), aleph4state, "...")

        if self.world.is_terminal(state):
            res = self.world.state_embedding(state)
        elif policy=="actual":
            def X(actionAndAleph):
                return self.ETerminalState_action(state, actionAndAleph[0], actionAndAleph[1], policy) # recursion
            res = self.localPolicy(state, aleph4state).expectation(X)
        else:
            def X(action):
                return self.ETerminalState_action(state, action, None, policy) # recursion
            res = self["defaultPolicy"](state).expectation(X)

        if self.debug or self.verbose:
            print(pad(state),"╰ ETerminalState_state", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def ETerminalState2_state(self, state, aleph4state, policy="actual"): # recursive
        """expected value of entrywise squared (vector-embedded) terminal state"""
        if self.debug:
            print(pad(state),"ETerminalState2_state", prettyState(state), aleph4state, "...")

        if self.world.is_terminal(state):
            res = self.world.state_embedding(state)**2
        elif policy=="actual":
            def X(actionAndAleph):
                return self.ETerminalState2_action(state, actionAndAleph[0], actionAndAleph[1], policy) # recursion
            res = self.localPolicy(state, aleph4state).expectation(X)
        else:
            def X(action):
                return self.ETerminalState2_action(state, action, None, policy) # recursion
            res = self["defaultPolicy"](state).expectation(X)

        if self.debug or self.verbose:
            print(pad(state),"╰ ETerminalState2_state", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def behaviorEntropy_state(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"behaviorEntropy_state", prettyState(state), aleph4state, "...")
        locPol = self.localPolicy(state, aleph4state)
        def X(actionAndAleph):
            return self.behaviorEntropy_action(state, locPol.probability(actionAndAleph), actionAndAleph[0], actionAndAleph[1]) # recursion
        res = locPol.expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ behaviorEntropy_state", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def behaviorKLdiv_state(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"behaviorKLdiv_state", prettyState(state), aleph4state, "...")
        locPol = self.localPolicy(state, aleph4state)
        def X(actionAndAleph):
            return self.behaviorKLdiv_action(state, locPol.probability(actionAndAleph), actionAndAleph[0], actionAndAleph[1]) # recursion
        res = locPol.expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ behaviorKLdiv_state", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def trajectoryEntropy_state(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"trajectoryEntropy_state", prettyState(state), aleph4state, "...")
        locPol = self.localPolicy(state, aleph4state)
        def X(actionAndAleph):
            return self.trajectoryEntropy_action(state, locPol.probability(actionAndAleph), actionAndAleph[0], actionAndAleph[1]) # recursion
        res = locPol.expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ trajectoryEntropy_state", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def stateDistance_state(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"stateDistance_state", prettyState(state), aleph4state, "...")
        locPol = self.localPolicy(state, aleph4state)
        def X(actionAndAleph):
            return self.stateDistance_action(state, actionAndAleph[0], actionAndAleph[1]) # recursion
        res = locPol.expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ stateDistance_state", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def causation_state(self, state, aleph4state): # recursive
        """Directed information from action sequence to state sequence"""
        if self.debug:
            print(pad(state),"causation_state", prettyState(state), aleph4state, "...")
        locPol = self.localPolicy(state, aleph4state)
        def Y(nextState, action):
            p = self.world.transition_probability(state, action, nextState)[0]
            if p == 0: return float("-inf")
            return math.log(p / locPol.expectation(lambda otherActionAndAleph: 
                                   self.world.transition_probability(state, otherActionAndAleph[0], nextState)[0]))
        def X(actionAndAleph):
            action, aleph4action = actionAndAleph
            return self.world.expectation(state, action, Y, (action,)) + self.causation_action(state, aleph4state, action, aleph4action) # recursion
        res = locPol.expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ causation_state", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def causationPotential_state(self, state, aleph4state): # recursive
        """Maximal directed information from action sequence to state sequence over all possible policies"""
        raise NotImplementedError("causationPotential_state is not yet implemented correctly")
        if self.debug:
            print(pad(state),"causationPotential_state", prettyState(state), aleph4state, "...")
        locPol = self.localPolicy(state, aleph4state)
        def Y(nextState, action):
            p = self.world.transition_probability(state, action, nextState)[0]
            if p == 0: return float("-inf")
            return math.log(p / locPol.expectation(lambda otherActionAndAleph: 
                                   self.world.transition_probability(state, otherActionAndAleph[0], nextState)[0]))
        res = max([self.world.expectation(state, action, Y, (action,)) 
                    + self.causationPotential_action(state, aleph4state, action, self.aspiration4action(state, action, aleph4state))
                    for action in self.world.possible_actions(state)]) 
        if self.debug or self.verbose:
            print(pad(state),"╰ causationPotential_state", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def otherLoss_state(self, state, aleph4state): # recursive
        if self.debug:
            print(pad(state),"otherLoss_state", prettyState(state), aleph4state, "...")
        def X(actionAndAleph):
            return self.otherLoss_action(state, actionAndAleph[0], actionAndAleph[1]) # recursion
        res = self.localPolicy(state, aleph4state).expectation(X)
        if self.debug or self.verbose:
            print(pad(state),"╰ otherLoss_state", prettyState(state), aleph4state, ":", res)
        return res

    @cache
    def randomTieBreaker(self, state, action):
        return random.random()

    # now we can combine all of the above quantities to a combined (safety) loss function:

    # state, action, aleph4state, aleph4action, estActionProbability
    @cache
    def combinedLoss(self, s, a, al4s, al4a, p): # recursive

        if self.debug:
            print(pad(s),"| | combinedLoss, state",prettyState(s),"action",a,"aleph4state",al4s,"aleph4action",al4a,"estActionProbability",p,"...")

        res = tuple((n, k * v(s, a, al4s, al4a, p)) for n, (k, v) in self.losses.items())

        if self.verbose or self.debug:
            losses = ", ".join(f"{n}={l}" for n, l in res )
            print(pad(s),"| | combinedLoss, state",prettyState(s),"action",a,"aleph4state",al4s,"aleph4action",al4a,"estActionProbability",p,":",res,"\n"+pad(s),"| | losses: ", losses)

        return sum(l for (_, l) in res)

    def getData(self): # FIXME: still needed?
        return {
            "stateActionPairs": list(self.stateActionPairsSet),
            "states": list({pair[0] for pair in self.stateActionPairsSet}),
            "locs": [state.loc for state in states],
        }

    @abstractmethod
    def maxAdmissibleQ(self, state, action, coeffs=None): pass
    @abstractmethod
    def minAdmissibleQ(self, state, action): pass
    @abstractmethod
    def disorderingPotential_action(self, state, action): pass
    @abstractmethod
    def agencyChange_action(self, state, action): pass

    @abstractmethod
    def LRAdev_action(self, state, action, aleph4action, myopic=False): pass
    @abstractmethod
    def behaviorEntropy_action(self, state, actionProbability, action, aleph4action): pass
    @abstractmethod
    def behaviorKLdiv_action(self, state, actionProbability, action, aleph4action): pass
    @abstractmethod
    def trajectoryEntropy_action(self, state, actionProbability, action, aleph4action): pass
    @abstractmethod
    def stateDistance_action(self, state, action, aleph4action): pass
    @abstractmethod
    def causation_action(self, state, action, aleph4action): pass
    @abstractmethod
    def causationPotential_action(self, state, action, aleph4action): pass
    @abstractmethod
    def otherLoss_action(self, state, action, aleph4action): pass

    @abstractmethod
    def Q(self, state, action, aleph4action): pass
    @abstractmethod
    def Q2(self, state, action, aleph4action): pass
    @abstractmethod
    def Q3(self, state, action, aleph4action): pass
    @abstractmethod
    def Q4(self, state, action, aleph4action): pass
    @abstractmethod
    def Q5(self, state, action, aleph4action): pass
    @abstractmethod
    def Q6(self, state, action, aleph4action): pass

    @abstractmethod
    def Q_ones(self, state, action, aleph4action): pass
    @abstractmethod
    def Q_DeltaSquare(self, state, action, aleph4action): pass

    @abstractmethod
    def ETerminalState_action(self, state, action, aleph4action, policy="actual"): pass
    @abstractmethod
    def ETerminalState2_action(self, state, action, aleph4action, policy="actual"): pass

    @abstractmethod
    def possible_actions(self, state, action): pass

class AgentMDPLearning(AspirationAgent):
    def __init__(self, params, maxAdmissibleQ=None, minAdmissibleQ=None, 
            disorderingPotential_action=None,
            agencyChange_action=None,
            LRAdev_action=None, Q_ones=None, Q_DeltaSquare=None, 
            behaviorEntropy_action=None, behaviorKLdiv_action=None, 
            trajectoryEntropy_action=None, stateDistance_action=None,
            causation_action=None,
            causationPotential_action=None,
            otherLoss_action=None,
            Q=None, Q2=None, Q3=None, Q4=None, Q5=None, Q6=None,
            ETerminalState_action=None, ETerminalState2_action=None,
            possible_actions=None):
        super().__init__(params)

        self.maxAdmissibleQ = maxAdmissibleQ
        self.minAdmissibleQ = minAdmissibleQ
        self.disorderingPotential_action = disorderingPotential_action
        self.agencyChange_action = agencyChange_action

        self.LRAdev_action = LRAdev_action
        self.behaviorEntropy_action = behaviorEntropy_action
        self.behaviorKLdiv_action = behaviorKLdiv_action
        self.trajectoryEntropy_action = trajectoryEntropy_action
        self.stateDistance_action = stateDistance_action
        self.causation_action = causation_action
        self.causationPotential_action = causationPotential_action
        self.otherLoss_action = otherLoss_action

        self.Q = Q
        self.Q2 = Q2
        self.Q3 = Q3
        self.Q4 = Q4
        self.Q5 = Q5
        self.Q6 = Q6

        self.Q_ones = Q_ones
        self.Q_DeltaSquare = Q_DeltaSquare

        self.ETerminalState_action = ETerminalState_action
        self.ETerminalState2_action = ETerminalState2_action

        self.possible_actions = possible_actions

class AgentMDPPlanning(AspirationAgent):
    def __init__(self, params, world=None):
        self.world = world
        super().__init__(params)
        self.history = []
        self.offset = None

    def possible_actions(self, state):
        if self.world.is_terminal(state):
            return []
        return self.world.possible_actions(state)

    # Compute upper and lower admissibility bounds for Q and V that are allowed in view of maxLambda and minLambda:

    # Compute the Q and V functions of the classical maximization problem (if maxLambda==1)
    # or of the LRA-based problem (if maxLambda<1):

    @cache
    def maxAdmissibleQ(self, state, action, coeffs=None): # recursive
        """maximal admissible Q-value for a given state-action pair (if dim==1 and coeffs not given),
        or admissible Q-value for state and action when maximizing the given linear combination of the components of the expected Total (if coeffs given)"""
        if self.verbose or self.debug:
            print(pad(state), "| | | | maxAdmissibleQ, state", state, "action", action, "coeffs", coeffs, "...")

        # register (state, action) in global store (could be anywhere, but here is just as fine as anywhere else)
        self.stateActionPairsSet.add((state, action))

        Edel = self.world.raw_moment_of_delta(state, action)
        # Bellman equation
        q = Edel + self.world.expectation(state, action, self.maxAdmissibleV, (coeffs,)) # recursion

        if self.verbose or self.debug:
            print(pad(state), "| | | | ╰ maxAdmissibleQ, state", state, "action", action, "coeffs", coeffs, ":", q)

        return q

    @cache
    def minAdmissibleQ(self, state, action): # recursive
        if self.verbose or self.debug:
            print(pad(state), "| | | | minAdmissibleQ, state", state, "action", action, "...")

        # register (state, action) in global store (could be anywhere, but here is just as fine as anywhere else)
        self.stateActionPairsSet.add((state, action))

        Edel = self.world.raw_moment_of_delta(state, action)
        # Bellman equation
        q = Edel + self.world.expectation(state, action, self.minAdmissibleV) # recursion

        if self.verbose or self.debug:
            print(pad(state), "| | | | ╰ minAdmissibleQ, state", state, "action", action, ":", q)

        return q

    # TODO: Consider two other alternatives:
    # 1. Only rescale the width and not the location of the aspiration interval,
    # and move it as close as possible to the state aspiration interval
    # (but maybe keeping a minimal safety distance from the bounds of the admissibility interval of the action).
    # In both cases, if the admissibility interval of the action is larger than that of the state,
    # the final action aspiration interval might need to be shrinked to fit into the aspiration interval of the state
    # once the mixture is know.
    # 2. This could be avoided by a further modification, where we rescale only downwards, never upwards:
    # - If phi(a) contains aleph(s), then aleph(a) = aleph(s)
    # - If aleph(s) contains phi(a), then aleph(a) = phiMid(a) +- alephW(s)*phiW(a)/phiW(s) / 2
    # - If phiLo(a) < alephLo(s) and phiHi(a) < alephHi(s), then aleph(a) = phiHi(a) - [0, alephW(s)*min(1,phiW(a)/phiW(s))]
    # - If phiHi(a) > alephHi(s) and phiLo(a) > alephLo(s), then aleph(a) = phiLo(a) + [0, alephW(s)*min(1,phiW(a)/phiW(s))]

    # Some safety metrics do not depend on aspiration and can thus also be computed upfront,
    # like min/maxAdmissibleQ, min/maxAdmissibleV:


    # TODO: IMPLEMENT A LEARNING VERSION OF THIS FUNCTION:

    # Disordering potential (maximal entropy (relative to some defaultTransition) 
    # over trajectories any agent could produce from here (see overleaf for details)):
    @cache
    def disorderingPotential_action(self, state, action): # recursive
        if self.debug:
            print(pad(state),"| | | disorderingPotential_action", prettyState(state), action, '...')
        if not self.default_transition:
            self._compute_default_transition(state)
        def f(nextState, probability):
            if self.world.is_terminal(nextState):
                return 0
            else:
                nextMP = self.disorderingPotential_state(nextState) # recursion
                defaultScore = self.default_transition(state).score(nextState)
                internalEntropy = self["internalTransitionEntropy"](state, action, nextState) if self["internalTransitionEntropy"] else 0
                return nextMP + defaultScore - math.log(probability) + internalEntropy

        # Note for ANN approximation: disorderingPotential_action can be positive or negative. 
        res = self.world.expectation_of_fct_of_probability(state, action, f)
        if self.debug:
            print(pad(state),"| | | ╰ disorderingPotential_action", prettyState(state), action, ':', res)
        return res

    @cache
    def agencyChange_action(self, state, action): # recursive
        """the expected absolute change in log agency (to be independent of scale)"""
        if self.debug:
            print(pad(state),"| | | agencyChange_action", prettyState(state), action, '...')
        # Note for ANN approximation: agency_action can only be non-negative. 
        state_agency = self.agency_state(state)
        def f(successor):
            return 0 if self.world.is_terminal(successor) else abs(math.log(state_agency) - math.log(self.agency_state(successor)))
        res = self.world.expectation(state, action, f)
        if self.debug:
            print(pad(state),"| | | ╰ agencyChange_action", prettyState(state), action, ':', res)
        return res


    # TODO: IMPLEMENT A LEARNING VERSION OF THIS FUNCTION:

    # Based on the policy, we can compute many resulting quantities of interest useful in assessing safety
    # better than with the above myopic safety metrics. All of them satisfy Bellman-style equations:

    # Actual Q and V functions of resulting policy (always returning scalars):
    @cache
    def Q(self, state, action, aleph4action): # recursive
        if self.debug:
            print(pad(state),"| | | | Q", prettyState(state), action, aleph4action, '...')

        Edel = self.world.raw_moment_of_delta(state, action)
        def total(nextState):
            if self.world.is_terminal(nextState):
                return Edel
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return Edel + self.V(nextState, nextAleph4state) # recursion
        q = self.world.expectation(state, action, total)

        if self.debug or self.verbose:
            print(pad(state),"| | | | ╰ Q", prettyState(state), action, aleph4action, ":", q)
        return q

    # Expected squared total, for computing the variance of total:
    @cache
    def Q2(self, state, action, aleph4action): # recursive
        if self.debug:
            print(pad(state),"| | | | | Q2", prettyState(state), action, aleph4action, '...')

        Edel = self.world.raw_moment_of_delta(state, action)
        Edel2 = self.world.raw_moment_of_delta(state, action, 2)

        def total(nextState):
            if self.world.is_terminal(nextState):
                return Edel2
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                # TODO: verify formula:
                return Edel2 \
                    + 2*Edel*self.V(nextState, nextAleph4state) \
                    + self.V2(nextState, nextAleph4state) # recursion

        q2 = self.world.expectation(state, action, total)

        if self.debug or self.verbose:
            print(pad(state),"| | | | | ╰ Q2", prettyState(state), action, aleph4action, ":", q2)
        return q2

    # Similarly: Expected third and fourth powers of total, for computing the 3rd and 4th centralized moment of total:
    @cache
    def Q3(self, state, action, aleph4action): # recursive
        if self.debug:
            print(pad(state),"| | | | | Q3", prettyState(state), action, aleph4action, '...')

        Edel = self.world.raw_moment_of_delta(state, action)
        Edel2 = self.world.raw_moment_of_delta(state, action, 2)
        Edel3 = self.world.raw_moment_of_delta(state, action, 3)

        def total(nextState):
            if self.world.is_terminal(nextState):
                return Edel3
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                # TODO: verify formula:
                return Edel3 \
                    + 3*Edel2*self.V(nextState, nextAleph4state) \
                    + 3*Edel*self.V2(nextState, nextAleph4state) \
                    + self.V3(nextState, nextAleph4state) # recursion
        q3 = self.world.expectation(state, action, total)

        if self.debug or self.verbose:
            print(pad(state),"| | | | | ╰ Q3", prettyState(state), action, aleph4action, ":", q3)
        return q3

    # Expected fourth power of total, for computing the expected fourth power of deviation of total from expected total (= fourth centralized moment of total):
    @cache
    def Q4(self, state, action, aleph4action): # recursive
        if self.debug:
            print(pad(state),"| | | | | Q4", prettyState(state), action, aleph4action, '...')

        Edel = self.world.raw_moment_of_delta(state, action)
        Edel2 = self.world.raw_moment_of_delta(state, action, 2)
        Edel3 = self.world.raw_moment_of_delta(state, action, 3)
        Edel4 = self.world.raw_moment_of_delta(state, action, 4)

        def total(nextState):
            if self.world.is_terminal(nextState):
                return Edel4
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                # TODO: verify formula:
                return Edel4 \
                    + 4*Edel3*self.V(nextState, nextAleph4state) \
                    + 6*Edel2*self.V2(nextState, nextAleph4state) \
                    + 4*Edel*self.V3(nextState, nextAleph4state) \
                    + self.V4(nextState, nextAleph4state) # recursion
        q4 = self.world.expectation(state, action, total)

        if self.debug or self.verbose:
            print(pad(state),"| | | | | ╰ Q4", prettyState(state), action, aleph4action, ":", q4)
        return q4

    # Expected fifth power of total, for computing the bed-and-banks loss component based on a 6th order polynomial potential of this shape: https://www.wolframalpha.com/input?i=plot+%28x%2B1%29%C2%B3%28x-1%29%C2%B3+ :
    @cache
    def Q5(self, state, action, aleph4action): # recursive
        if self.debug:
            print(pad(state),"| | | | | Q5", prettyState(state), action, aleph4action, '...')

        Edel = self.world.raw_moment_of_delta(state, action)
        Edel2 = self.world.raw_moment_of_delta(state, action, 2)
        Edel3 = self.world.raw_moment_of_delta(state, action, 3)
        Edel4 = self.world.raw_moment_of_delta(state, action, 4)
        Edel5 = self.world.raw_moment_of_delta(state, action, 5)

        def total(nextState):
            if self.world.is_terminal(nextState):
                return Edel5
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                # TODO: verify formula:
                return Edel5 \
                    + 5*Edel4*self.V(nextState, nextAleph4state) \
                    + 10*Edel3*self.V2(nextState, nextAleph4state) \
                    + 10*Edel2*self.V3(nextState, nextAleph4state) \
                    + 5*Edel*self.V4(nextState, nextAleph4state) \
                    + self.V5(nextState, nextAleph4state) # recursion
        q5 = self.world.expectation(state, action, total)

        if self.debug or self.verbose:
            print(pad(state),"| | | | | ╰ Q5", prettyState(state), action, aleph4action, ":", q5)
        return q5

    # Expected sixth power of total, for computing the bed-and-banks loss component based on a 6th order polynomial potential of this shape: https://www.wolframalpha.com/input?i=plot+%28x%2B1%29%C2%B3%28x-1%29%C2%B3+ :
    @cache
    def Q6(self, state, action, aleph4action): # recursive
        if self.debug:
            print(pad(state),"| | | | | Q6", prettyState(state), action, aleph4action, '...')

        Edel = self.world.raw_moment_of_delta(state, action)
        Edel2 = self.world.raw_moment_of_delta(state, action, 2)
        Edel3 = self.world.raw_moment_of_delta(state, action, 3)
        Edel4 = self.world.raw_moment_of_delta(state, action, 4)
        Edel5 = self.world.raw_moment_of_delta(state, action, 5)
        Edel6 = self.world.raw_moment_of_delta(state, action, 6)

        def total(nextState):
            if self.world.is_terminal(nextState):
                return Edel6
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                # TODO: verify formula:
                return Edel6 \
                    + 6*Edel5*self.V(nextState, nextAleph4state) \
                    + 15*Edel4*self.V2(nextState, nextAleph4state) \
                    + 20*Edel3*self.V3(nextState, nextAleph4state) \
                    + 15*Edel2*self.V4(nextState, nextAleph4state) \
                    + 6*Edel*self.V5(nextState, nextAleph4state) \
                    + self.V6(nextState, nextAleph4state) # recursion
        q6 = self.world.expectation(state, action, total)

        if self.debug or self.verbose:
            print(pad(state),"| | | | | ╰ Q6", prettyState(state), action, aleph4action, ":", q6)
        return q6

    # Squared deviation of local relative aspiration (midpoint of interval) from 0.5:
    @cache
    def LRAdev_action(self, state, action, aleph4action, myopic=False): # recursive
        if self.debug:
            print(pad(state),"| | | LRAdev_action", prettyState(state), action, aleph4action, myopic, '...')

        # Note for ANN approximation: LRAdev_action must be between 0 and 0.25 
        Edel = self.world.raw_moment_of_delta(state, action)

        def dev(nextState):
            localLRAdev = (0.5 - relativePosition(self.minAdmissibleQ(state, action), midpoint(aleph4action), self.maxAdmissibleQ(state, action))) ** 2
            if self.world.is_terminal(nextState) or myopic:
                return localLRAdev
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return localLRAdev + self.LRAdev_state(nextState, nextAleph4state) # recursion
        res = self.world.expectation(state, action, dev)

        if self.debug or self.verbose:
            print(pad(state),"| | | ╰ LRAdev_action", prettyState(state), action, aleph4action, ":", res)
        return res

    # TODO: verify the following two formulas for expected Delta variation along a trajectory:

    # Expected total of ones (= expected length of trajectory), for computing the expected Delta variation along a trajectory:
    @cache
    def Q_ones(self, state, action, aleph4action=None): # recursive
        if self.debug:
            print(pad(state),"| | | | | Q_ones", prettyState(state), action, aleph4action, "...")
        Edel = self.world.raw_moment_of_delta(state, action)

        # Note for ANN approximation: Q_ones must be nonnegative. 
        def one(nextState):
            if self.world.is_terminal(nextState):
                return 1
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return 1 + self.V_ones(nextState, nextAleph4state) # recursion
        q_ones = self.world.expectation(state, action, one)

        if self.debug or self.verbose:
            print(pad(state),"| | | | | ╰ Q_ones", prettyState(state), action, aleph4action, ":", q_ones)
        return q_ones

    # Expected total of squared Deltas, for computing the expected Delta variation along a trajectory:
    @cache
    def Q_DeltaSquare(self, state, action, aleph4action=None): # recursive
        if self.debug:
            print(pad(state),"| | | | | Q_DeltaSquare", prettyState(state), action, aleph4action, "...")
        Edel = self.world.raw_moment_of_delta(state, action)
        EdelSq = Edel**2 + self["varianceOfDelta"](state, action)

        # Note for ANN approximation: Q_DeltaSquare must be nonnegative. 
        def d(nextState):
            if self.world.is_terminal(nextState) or aleph4action is None:
                return EdelSq
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return EdelSq + self.V_DeltaSquare(nextState, nextAleph4state) # recursion
        if self.debug or self.verbose:
            print(pad(state),"| | | | | ╰ Q_DeltaSquare", prettyState(state), action, aleph4action, ":", qDsq)
        qDsq = self.world.expectation(state, action, d)
        return qDsq


	# Methods to calculate the approximate Wasserstein distance (in state embedding space) between policy-induced and default distribution of terminal states, both starting at the current state:

    @cache
    def ETerminalState_action(self, state, action, aleph4action, policy="actual"): # recursive
        if self.debug:
            print(pad(state),"| | | | | ETerminalState_action", prettyState(state), action, aleph4action, policy, '...')

        Edel = self.world.raw_moment_of_delta(state, action)
        if policy=="actual":
            def X(nextState):
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return self.ETerminalState_state(nextState, nextAleph4state, policy) # recursion
            res = self.world.expectation(state, action, X)
        else:
            def X(nextState):
                return self.ETerminalState_state(nextState, None, policy) # recursion
            res = self.world.expectation(state, action, X)

        if self.debug or self.verbose:
            print(pad(state),"| | | | | ╰ ETerminalState_action", prettyState(state), action, aleph4action, policy, ":", res)
        return res

    @cache
    def ETerminalState2_action(self, state, action, aleph4action, policy="actual"): # recursive
        if self.debug:
            print(pad(state),"| | | | | ETerminalState2_action", prettyState(state), action, aleph4action, policy, '...')

        Edel = self.world.raw_moment_of_delta(state, action)
        if policy=="actual":
            def X(nextState):
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return self.ETerminalState2_state(nextState, nextAleph4state, policy) # recursion
            res = self.world.expectation(state, action, X)
        else:
            def X(nextState):
                return self.ETerminalState2_state(nextState, None, policy) # recursion
            res = self.world.expectation(state, action, X)

        if self.debug or self.verbose:
            print(pad(state),"| | | | | ╰ ETerminalState2_action", prettyState(state), action, aleph4action, policy, ":", res)
        return res

    @cache
    def wassersteinTerminalState_action(self, state, action, aleph4action):
        if self.debug:
            print(pad(state),"| | | | wassersteinTerminalState_action", prettyState(state), action, aleph4action, '...')
        mu0 = self.ETerminalState_state(state, None, "default")
        mu20 = self.ETerminalState2_state(state, None, "default")
        muPi = self.ETerminalState_action(state, action, aleph4action, "actual")
        mu2Pi = self.ETerminalState2_action(state, action, aleph4action, "actual")
        sigma0 = np.maximum(mu20 - mu0**2, 0)**0.5
        sigmaPi = np.maximum(mu2Pi - muPi**2, 0)**0.5
        res = ((mu0 - muPi)**2).sum() + ((sigma0 - sigmaPi)**2).sum()
        if self.debug or self.verbose:
            print(pad(state),"| | | | ╰ wassersteinTerminalState_action", prettyState(state), action, aleph4action, ":", res)
        return res

    # Other safety criteria:

    # Shannon entropy of behavior
    # (actually, negative KL divergence relative to uninformedPolicy (e.g., a uniform distribution),
    # to be consistent under action cloning or action refinement):
    #@cache
    def behaviorEntropy_action(self, state, actionProbability, action, aleph4action=None): # recursive
        # Note for ANN approximation: behaviorEntropy_action must be <= 0 (!) 
        # because it is the negative (!) of a KL divergence. 
        if self.debug:
            print(pad(state),"| | | behaviorEntropy_action", prettyState(state), action, aleph4action, '...')
        Edel = self.world.raw_moment_of_delta(state, action)
        def entropy(nextState):
            uninfPolScore = self["uninformedPolicy"](state).score(action) if ("uninformedPolicy" in self.params) else 0
            localEntropy = uninfPolScore \
                            - math.log(actionProbability) \
                            + (self["internalActionEntropy"](state, action) if ("internalActionEntropy" in self.params) else 0)
            if self.world.is_terminal(nextState) or aleph4action is None:
                return localEntropy
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return localEntropy + self.behaviorEntropy_state(nextState, nextAleph4state) # recursion
        res = self.world.expectation(state, action, entropy)

        if self.debug or self.verbose:
            print(pad(state),"| | | ╰ behaviorEntropy_action", prettyState(state), action, aleph4action, ":", res)
        return res

    # KL divergence of behavior relative to refPolicy (or uninformedPolicy if refPolicy is not set):
    #@cache
    def behaviorKLdiv_action(self, state, actionProbability, action, aleph4action=None): # recursive
        # Note for ANN approximation: behaviorKLdiv_action must be nonnegative. 
        if self.debug:
            print(pad(state),"| | | behaviorKLdiv_action", prettyState(state), action, aleph4action, '...')
        refPol = None
        if "referencePolicy" in self.params:
            refPol = self["referencePolicy"]
        elif "uninformedPolicy" in self.params:
            refPol = self["uninformedPolicy"]
        else:
            return None # TODO this should remain None after math operations

        Edel = self.world.raw_moment_of_delta(state, action)
        def div(nextState):
            localDivergence = math.log(actionProbability) - refPol(state).score(action)
            if self.world.is_terminal(nextState) or aleph4action is None:
                return localDivergence
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return localDivergence + self.behaviorKLdiv_state(nextState, nextAleph4state) # recursion
        res = self.world.expectation(state, action, div)

        if self.debug or self.verbose:
            print(pad(state),"| | | ╰ behaviorKLdiv_action", prettyState(state), action, aleph4action, ":", res)
        return res

    # Shannon entropy of trajectory
    # (actually, negative KL divergence relative to defaultTransition (e.g., a uniform distribution),
    # to be consistent under state cloning or state refinement):
    #@cache
    def trajectoryEntropy_action(self, state, actionProbability, action, aleph4action=None): # recursive
        # Note for ANN approximation: trajectoryEntropy_action must be <= 0 (!) 
        # because it is the negative (!) of a KL divergence. 
        if self.debug:
            print(pad(state),"| | | trajectoryEntropy_action", prettyState(state), actionProbability, action, aleph4action, '...')
        if not self.default_transition:
            self._compute_default_transition(state)
        Edel = self.world.raw_moment_of_delta(state, action)
        def entropy(nextState, transitionProbability):
            priorScore = self["uninformedStatePriorScore"](nextState)
            localEntropy = priorScore \
                            - math.log(actionProbability) \
                            - math.log(transitionProbability) \
                            + (self["internalTrajectoryEntropy"](state, action) if ("internalTrajectoryEntropy" in self.params) else 0)
            # TODO: decide whether the priorScore should really be used as it leads to completely opposite behavior in GW25: with the priorScore in place, penalizing trajectoryEntropy makes the agent *avoid* destroying the moving object, which should be considered a *non-reduction* in entropy, while destroying it should be considered a reduction in entropy...
            if self.world.is_terminal(nextState) or aleph4action is None:
                return localEntropy
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return localEntropy + self.trajectoryEntropy_state(nextState, nextAleph4state) # recursion
        res = self.world.expectation_of_fct_of_probability(state, action, entropy)

        if self.debug or self.verbose:
            print(pad(state),"| | | ╰ trajectoryEntropy_action", prettyState(state), actionProbability, action, aleph4action, ":", res)
        return res

    # Expected squared distance of terminal state from reference state:
    #@cache
    def stateDistance_action(self, state, action, aleph4action=None): # recursive
        # Note for ANN approximation: stateDistance_action must be >= 0
        if self.debug:
            print(pad(state),"| | | stateDistance_action", prettyState(state), action, aleph4action, '...')
        Edel = self.world.raw_moment_of_delta(state, action)
        def X(nextState):
            if self.world.is_terminal(nextState) or aleph4action is None:
                return self.world.state_distance(nextState, self.params["referenceState"]) ** 2
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return self.stateDistance_state(nextState, nextAleph4state) # recursion
        res = self.world.expectation(state, action, X)

        if self.debug or self.verbose:
            print(pad(state),"| | | ╰ stateDistance_action", prettyState(state), action, aleph4action, ":", res)
        return res

    # Causation (=directed information) from actions to states:
    #@cache
    def causation_action(self, state, aleph4state, action, aleph4action=None): # recursive
        # Note for ANN approximation: causation_action must be >= 0
        if self.debug:
            print(pad(state),"| | | causation_action", prettyState(state), action, aleph4action, '...')
        Edel = self.world.raw_moment_of_delta(state, action)
        def X(nextState):
            if self.world.is_terminal(nextState) or aleph4action is None:
                return 0
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return self.causation_state(nextState, nextAleph4state) # recursion
        res = self.world.expectation(state, action, X)

        if self.debug or self.verbose:
            print(pad(state),"| | | ╰ causation_action", prettyState(state), action, aleph4action, ":", res)
        return res

    # Causation Potential (= maximal directed information) from actions to states:
    #@cache
    def causationPotential_action(self, state, aleph4state, action, aleph4action=None): # recursive
        # Note for ANN approximation: causationPotential_action must be >= 0
        if self.debug:
            print(pad(state),"| | | causationPotential_action", prettyState(state), action, aleph4action, '...')
        Edel = self.world.raw_moment_of_delta(state, action)
        def X(nextState):
            if self.world.is_terminal(nextState) or aleph4action is None:
                return 0
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return self.causationPotential_state(nextState, nextAleph4state) # recursion
        res = self.world.expectation(state, action, X)

        if self.debug or self.verbose:
            print(pad(state),"| | | ╰ causationPotential_action", prettyState(state), action, aleph4action, ":", res)
        return res

    # other loss:
    #@cache
    def otherLoss_action(self, state, action, aleph4action=None): # recursive
        if self.debug:
            print(pad(state),"| | | otherLoss_action", prettyState(state), action, aleph4action, '...')
        Edel = self.world.raw_moment_of_delta(state, action)
        def loss(nextState):
            #localLoss = self["otherLocalLoss"](state, action) # TODO this variable may not exist in params
            localLoss = 0 # TODO this variable may not exist in params
            if self.world.is_terminal(nextState) or aleph4action is None:
                return localLoss
            else:
                nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
                return localLoss + self.otherLoss_state(nextState, nextAleph4state) # recursion
        res =  self.world.expectation(state, action, loss)

        if self.debug or self.verbose:
            print(pad(state),"| | | ╰ otherLoss_action", prettyState(state), action, aleph4action, ":", res)
        return res
