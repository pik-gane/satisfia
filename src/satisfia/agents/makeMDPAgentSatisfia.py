#!/usr/bin/env python3

import math
from functools import lru_cache

from util import distribution

VERBOSE = False
DEBUG = False

"""class Interval():
	def __init__(self, left, right=None):
		self.left = left
		self.right = right if (right != None) else left"""

def interpolate(x, l, y):
	# denoted  x : l : y in formulas

	""" TODO implement intervals
	if (_.isArray(x) || _.isArray(lam) || _.isArray(y)) {
		// one argument is an interval, so everything becomes an interval:
		var xx = asInterval(x), lamlam = asInterval(lam), yy = asInterval(y);
		return [xx[0] + lamlam[0] * (yy[0] - xx[0]),
				xx[1] + lamlam[1] * (yy[1] - xx[1])];
	}"""

	return x + l * (y - x);

class AgentMDP():
	def __init__(self, params, world):
		defaults = {
			# admissibility parameters:
			"maxLambda": 1, # upper bound on local relative aspiration in each step (must be minLambda...1)  # TODO: rename to lambdaHi
			"minLambda": 0, # lower bound on local relative aspiration in each step (must be 0...maxLambda)  # TODO: rename to lambdaLo
			# policy parameters:
			"lossTemperature": 0.1, # temperature of softmin mixture of actions w.r.t. loss, must be > 0
			# "rescaling4Actions": 0, # degree (0...1) of aspiration rescaling from state to action. (larger implies larger variance) # TODO: disable this because a value of >0 can't be taken into account in a consistent way easily
			# "rescaling4Successors": 1, # degree (0...1) of aspiration rescaling from action to successor state. (expectation is only preserved if this is 1.0) # TODO: disable also this since a value <1 leads to violation of the expectation guarantee 
			# coefficients for cheap to compute loss functions:
			"lossCoeff4Random": 0, # weight of random tie breaker in loss function, must be >= 0
			"lossCoeff4FeasibilityPower": 1, # weight of power of squared admissibility interval width in loss function, must be >= 0
			"lossCoeff4MP": 1, # weight of messing potential in loss function, must be >= 0
			"lossCoeff4LRA1": 1, # weight of current-state deviation of LRA from 0.5 in loss function, must be >= 0
			"lossCoeff4Time1": 1, # weight of not terminating in loss function, must be >= 0
			"lossCoeff4Entropy1": 1, # weight of current-state action entropy in loss function, must be >= 0
			"lossCoeff4KLdiv1": 1, # weight of current-state KL divergence in loss function, must be >= 0
			# coefficients for expensive to compute loss functions (all zero by default except for variance):
			"lossCoeff4Variance": 1, # weight of variance of total in loss function, must be >= 0
			"lossCoeff4Fourth": 0, # weight of centralized fourth moment of total in loss function, must be >= 0
			"lossCoeff4Cup": 0, # weight of "cup" loss component, based on sixth moment of total, must be >= 0
			"lossCoeff4LRA": 0, # weight of deviation of LRA from 0.5 in loss function, must be >= 0
			"lossCoeff4Time": 0, # weight of time in loss function, must be >= 0
			"lossCoeff4DeltaVariation": 0, # weight of variation of Delta in loss function, must be >= 0
			"lossCoeff4Entropy": 0, # weight of action entropy in loss function, must be >= 0
			"lossCoeff4KLdiv": 0, # weight of KL divergence in loss function, must be >= 0
			"lossCoeff4OtherLoss": 0, # weight of other loss components specified by otherLossIncrement, must be >= 0
			"allowNegativeCoeffs": False, # if true, allow negative loss coefficients

			"varianceOfDelta": (lambda state, action: 0),
			"skewnessOfDelta": (lambda state, action: 0),
			"excessKurtosisOfDelta": (lambda state, action: 0),
			"fifthMomentOfDelta": (lambda state, action: 8 * self.params["varianceOfDelta"](state, action) ** 2.5) # assumes a Gaussian distribution
			"sixthMomentOfDelta": (lambda state, action: 15 * self.params["varianceOfDelta"](state, action) ** 3) # assumes a Gaussian distribution
		}

		self.params = defaults | params
		# TODO do I need to add params_.options

		self.world = world

		self.stateActionPairsSet = set()

		assert lossTemperature > 0, "lossTemperature must be > 0"
		#assert 0 <= rescaling4Actions <= 1, "rescaling4Actions must be in 0...1"
		#assert 0 <= rescaling4Successors <= 1, "rescaling4Successors must be in 0...1"
		assert allowNegativeCoeffs || lossCoeff4Random >= 0, "lossCoeff4random must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4FeasibilityPower >= 0, "lossCoeff4FeasibilityPower must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4MP >= 0, "lossCoeff4MP must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4LRA1 >= 0, "lossCoeff4LRA1 must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4Time1 >= 0, "lossCoeff4Time1 must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4Entropy1 >= 0, "lossCoeff4Entropy1 must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4KLdiv1 >= 0, "lossCoeff4KLdiv1 must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4Variance >= 0, "lossCoeff4variance must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4Fourth >= 0, "lossCoeff4Fourth must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4Cup >= 0, "lossCoeff4Cup must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4LRA >= 0, "lossCoeff4LRA must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4Time >= 0, "lossCoeff4time must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4DeltaVariation >= 0, "lossCoeff4DeltaVariation must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4Entropy >= 0, "lossCoeff4entropy must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4KLdiv >= 0, "lossCoeff4KLdiv must be >= 0"
		assert allowNegativeCoeffs || lossCoeff4OtherLoss >= 0, "lossCoeff4OtherLoss must be >= 0"
		assert lossCoeff4Entropy == 0 || lossCoeff4MP == 0 || (uninformedPolicy in self.params), "uninformedPolicy must be provided if lossCoeff4MP > 0 or lossCoeff4Entropy > 0"

		if verbose or debug:
			print("makeMDPAgentSatisfia with parameters", self.params);

	def __getitem__(self, name):
		return self.params[name]

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
			→ Q, .., Q6, Q_DeltaSquare, Q_ones
			  → propagateAspiration (see below)
			  → E(V), ..., E(V6), E(V_DeltaSquare), E(V_ones)
				→ localPolicy (RECURSION)
				→ E(Q), ..., E(Q6), E(Q_DeltaSquare), E(Q_ones) (RECURSION)
			→ otherLoss_action
			  → propagateAspiration (see below)
			  → E(otherLoss_state)
				→ localPolicy (RECURSION)
				→ E(otherLoss_action) (RECURSION)
			→ similarly for other loss components (RECURSION)
		→ expectedDelta, varianceOfDelta, transition
		→ propagateAspiration
		  → aspiration4state
		→ simulate (RECURSION)"""

	# Utility function for deriving transition probabilities from the transition function:
	# Remark: later the following should actually be provided by the env/world:

	@lru_cache(maxsize=None)
	def transitionDistribution(self, state, action):
		return distribution.infer(lambda: self.world.transition(state, action))

	# Compute upper and lower admissibility bounds for Q and V that are allowed in view of maxLambda and minLambda:

	# Compute the Q and V functions of the classical maximization problem (if maxLambda==1)
	# or of the LRA-based problem (if maxLambda<1):

	@lru_cache(maxsize=None)
	def maxAdmissibleQ(self, state, action): # recursive
		if verbose or debug:
			print("maxAdmissibleQ, state", state, "action", action, "...")

		# register (state, action) in global store (could be anywhere, but here is just as fine as anywhere else)
		self.stateActionPairsSet.insert((state, action))

		Edel = expectedDelta(state, action)
		if state.terminateAfterAction:
			q = Edel
		else:
			# Bellman equation
			q = Edel + self.world.expectation(state, action, self.maxAdmissibleV) # recursion

		if verbose or debug:
			print("maxAdmissibleQ, state", state, "action", action, ":", q)

		return q

	@lru_cache(maxsize=None)
	def minAdmissibleQ(self, state, action): # recursive
		if verbose or debug:
			print("minAdmissibleQ, state", state, "action", action, "...")

		# register (state, action) in global store (could be anywhere, but here is just as fine as anywhere else)
		stateActionPairsSet.insert((state, action))

		Edel = expectedDelta(state, action)
		if state.terminateAfterAction:
			q = Edel
		else:
			# Bellman equation
			q = distribution.infer(lambda: self.minAdmissibleV(self.world.transition(state, action))).E() # recursion

		if verbose or debug:
			print("minAdmissibleQ, state", state, "action", action, ":", q)

		return q

	@lru_cache(maxsize=None)
	def maxAdmissibleV(self, state): # recursive
		if verbose or debug:
			print("maxAdmissibleV, state", state, "...")

		actions = self.world.stateToActions(state)
		qs = [self.maxAdmissibleQ(state, a) for a in actions] # recursion
		v = max(qs) if maxLambda == 1 else interpolate(min(qs), maxLambda, max(qs))

		if verbose or debug:
			print("maxAdmissibleV, state", state, ":", v)

		return v

	@lru_cache(maxsize=None)
	def minAdmissibleV(self, state): # recursive
		if verbose or debug:
			print("minAdmissibleV, state", state, "...")

		actions = self.world.stateToActions(state)
		qs = [self.minAdmissibleQ(state, a) for a in actions] # recursion
		v = min(qs) if minLambda == 0 else interpolate(min(qs), minLambda, max(qs))

		if verbose or debug:
			print("minAdmissibleV, state", state, ":", v)

		return v

	# The resulting admissibility interval for states.
	def admissibility4state(self, state):
		return self.minAdmissibleV(state), self.maxAdmissibleV(state)

	# The resulting admissibility interval for actions.
	def admissibility4action(self, state, action):
		return self.minAdmissibleQ(state, action), self.maxAdmissibleQ(state, action)




	# Some safety metrics do not depend on aspiration and can thus also be computed upfront,
	# like min/maxAdmissibleQ, min/maxAdmissibleV:

	# Messing potential (maximal entropy (relative to some uninformedStatePrior) 
	# over trajectories any agent could produce from here (see overleaf for details)):
	@lru_cache(maxsize=None)
	def messingPotential_action(self, state, action): # recursive
		def sample(nextState, probability):
			if state.terminateAfterAction:
				return 0
			else:
				nextMP = self.messingPotential_state(nextState) # recursion
				priorScore = self["uninformedStatePriorScore"](nextState) if self["uninformedStatePriorScore"] else 0
				internalEntropy = self["internalTransitionEntropy"](state, action, nextState) if self["internalTransitionEntropy"] else 0
				return nextMP + priorScore - np.log(probability) + internalEntropy

		# Note for ANN approximation: messingPotential_action can be positive or negative. 
		return self.world.expectation_of_fct_of_probability([state], action, sample)

	@lru_cache(maxsize=None)
	def messingPotential_state(self, state): # recursive
		actions = self.world.possible_actions(state)
		maxMPpolicyWeights = [math.exp(self.messingPotential_action(state, a)) for a in actions]
		return math.log(sum(maxMPpolicyWeights))




