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

def between(item, a, b):
	return (a <= item <= b) or (b <= item <= a)

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
			"fifthMomentOfDelta": (lambda state, action: 8 * self.params["varianceOfDelta"](state, action) ** 2.5), # assumes a Gaussian distribution
			"sixthMomentOfDelta": (lambda state, action: 15 * self.params["varianceOfDelta"](state, action) ** 3) # assumes a Gaussian distribution
		}

		self.params = defaults | params
		# TODO do I need to add params_.options

		self.world = world

		self.stateActionPairsSet = set()

		assert lossTemperature > 0, "lossTemperature must be > 0"
		#assert 0 <= rescaling4Actions <= 1, "rescaling4Actions must be in 0...1"
		#assert 0 <= rescaling4Successors <= 1, "rescaling4Successors must be in 0...1"
		assert allowNegativeCoeffs or lossCoeff4Random >= 0, "lossCoeff4random must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4FeasibilityPower >= 0, "lossCoeff4FeasibilityPower must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4MP >= 0, "lossCoeff4MP must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4LRA1 >= 0, "lossCoeff4LRA1 must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4Time1 >= 0, "lossCoeff4Time1 must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4Entropy1 >= 0, "lossCoeff4Entropy1 must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4KLdiv1 >= 0, "lossCoeff4KLdiv1 must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4Variance >= 0, "lossCoeff4variance must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4Fourth >= 0, "lossCoeff4Fourth must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4Cup >= 0, "lossCoeff4Cup must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4LRA >= 0, "lossCoeff4LRA must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4Time >= 0, "lossCoeff4time must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4DeltaVariation >= 0, "lossCoeff4DeltaVariation must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4Entropy >= 0, "lossCoeff4entropy must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4KLdiv >= 0, "lossCoeff4KLdiv must be >= 0"
		assert allowNegativeCoeffs or lossCoeff4OtherLoss >= 0, "lossCoeff4OtherLoss must be >= 0"
		assert lossCoeff4Entropy == 0 or lossCoeff4MP == 0 or (uninformedPolicy in self.params), "uninformedPolicy must be provided if lossCoeff4MP > 0 or lossCoeff4Entropy > 0"

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
			print(pad(state), "maxAdmissibleQ, state", state, "action", action, "...")

		# register (state, action) in global store (could be anywhere, but here is just as fine as anywhere else)
		self.stateActionPairsSet.insert((state, action))

		Edel = expectedDelta(state, action)
		if state.terminateAfterAction:
			q = Edel
		else:
			# Bellman equation
			q = Edel + self.world.expectation(state, action, self.maxAdmissibleV) # recursion

		if verbose or debug:
			print(pad(state), "maxAdmissibleQ, state", state, "action", action, ":", q)

		return q

	@lru_cache(maxsize=None)
	def minAdmissibleQ(self, state, action): # recursive
		if verbose or debug:
			print(pad(state), "minAdmissibleQ, state", state, "action", action, "...")

		# register (state, action) in global store (could be anywhere, but here is just as fine as anywhere else)
		stateActionPairsSet.insert((state, action))

		Edel = expectedDelta(state, action)
		if state.terminateAfterAction:
			q = Edel
		else:
			# Bellman equation
			q = distribution.infer(lambda: self.minAdmissibleV(self.world.transition(state, action))).E() # recursion

		if verbose or debug:
			print(pad(state), "minAdmissibleQ, state", state, "action", action, ":", q)

		return q

	@lru_cache(maxsize=None)
	def maxAdmissibleV(self, state): # recursive
		if verbose or debug:
			print(pad(state), "maxAdmissibleV, state", state, "...")

		actions = self.world.stateToActions(state)
		qs = [self.maxAdmissibleQ(state, a) for a in actions] # recursion
		v = max(qs) if maxLambda == 1 else interpolate(min(qs), maxLambda, max(qs))

		if verbose or debug:
			print(pad(state), "maxAdmissibleV, state", state, ":", v)

		return v

	@lru_cache(maxsize=None)
	def minAdmissibleV(self, state): # recursive
		if verbose or debug:
			print(pad(state), "minAdmissibleV, state", state, "...")

		actions = self.world.stateToActions(state)
		qs = [self.minAdmissibleQ(state, a) for a in actions] # recursion
		v = min(qs) if minLambda == 0 else interpolate(min(qs), minLambda, max(qs))

		if verbose or debug:
			print(pad(state), "minAdmissibleV, state", state, ":", v)

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

	# Based on the admissibility information computed above, we can now construct the policy,
	# which is a mapping taking a state and an aspiration interval as input and returning
	# a categorical distribution over (action, aleph4action) pairs.

	def localPolicy(self, state, aleph): # recursive
		"""return a categorical distribution over (action, aleph4action) pairs"""

		d = localPolicyData(state, aleph)
		support = d[0]
		ps = d[1]

		if debug:
			print(pad(state), " localPolicy", prettyState(state), aleph, d)

		return distribution.categorical(support, ps)

	@lru_cache(maxsize=None)
	def localPolicyData(self, state, aleph):
		if verbose or debug:
			print(pad(state), "| localPolicy, state",prettyState(state),"aleph",aleph,"...")
		
		# Clip aspiration interval to admissibility interval of state:
		alephLo, alephHi = aleph4state = aspiration4state(state, aleph)

		# Estimate aspiration intervals for all possible actions in a way 
		# independent from the local policy that we are about to construct,
		actions = stateToActions(state)
		estAlephs1 = [estAspiration4action(state, action, aleph4state) for action in actions]

		# Estimate losses based on this estimated aspiration intervals
		# and use it to construct softmin propensities (probability weights) for choosing actions.
		# since we don't know the actual probabilities of actions yet (we will determine those only later),
		# but the loss estimation requires an estimate of the probability of the chosen action,
		# we estimate the probability at 1 / number of actions:
		def propensity(index, indices, estAlephs1):
			action = actions[index]
			loss = combinedLoss(state, action, aleph4state, estAlephs[index], 1 / indices.length)
			return min(1e100, max(math.exp(-loss / lossTemperature), 1e-100))

		indices = actions.keys()
		propensities = [propensity(index, indices, estAlephs1) for index in indices] # bottleneck

		if debug:
			print("| localPolicyData", prettyState(state), aleph, actions, {propensities})

		# now we can construct the local policy as a WebPPL distribution object:
		def sample():
			# Draw a first action a1 using the calculated softmin propensities,
			# and get its admissibility interval:
			####### TODO needs changes
			i1 = distribution.categorical(indices, propensities).sample()
			a1 = actions[i1]
			adm1Lo, adm1Hi = adm1 = admissibility4action(state, a1)

			# If a1's admissibility interval is completely contained in aleph4state, we are done:
			if set(adm1) < set(aleph4state):
				if verbose or debug:
					print(pad(state),"| | locPol, state",prettyState(state),"aleph4state",aleph4state,": a1",a1,"adm1",adm1,"(subset of aleph4state)")
				return [a1, adm1]
			else:
				# For drawing the second action, restrict actions so that the the midpoint of aleph4state can be mixed from
				# those of estAlephs4action of the first and second action:
				midTarget = midpoint(aleph4state)
				estAleph1 = estAlephs1[i1]
				mid1 = midpoint(estAleph1)
				indices2 = [index for index in indices if between(midTarget, midpoint(estAlephs1[index]), mid1)]

				# Since we are already set on giving a1 a considerable weight, we no longer aim to have aleph(a2)
				# as close as possible to aleph(s), but to a target aleph that would allow mixing a1 and a2 
				# in roughly equal proportions, i.e., we aim to have aleph(a2) as close as possible to
				# aleph(s) + (aleph(s) - aleph(a1)):
				aleph2target = interpolate(estAleph1, 2.0, aleph4state)
				# Due to the new target aleph, we have to recompute the estimated alephs and resulting losses and propensities:
				estAlephs2 = [estAspiration4action(state, actions[index], aleph2target) for index in indices]
				propensities2 = [propensity(index, indices2, estAlephs2) for index in indices2]

				if debug:
					print("| localPolicyData", prettyState(state), aleph4state, {"a1": a1, "midTarget": midTarget, "estAleph1": estAleph1, "mid1": mid1, "indices2": indices2, "aleph2target": aleph2target, "estAlephs2": estAlephs2, "propensities2": propensities2})

				# Like for a1, we now draw a2 using a softmin mixture of these actions, based on the new propentities,
				# and get its admissibility interval:
				####### TODO needs changes
				i2 = distribution.categorical(indices2, propensities2).sample()
				a2 = actions[i2]
				adm2Lo, adm2Hi = adm2 = admissibility4action(state, a2)

				# Now we need to find two aspiration intervals aleph1 in adm1 and aleph2 in adm2, 
				# and a probability p such that
				#   aleph1:p:aleph2 is contained in aleph4state 
				# and aleph1, aleph2 are close to the estimates we used above in estimating loss.
				# Instead of optimizing this, we use the following heuristic:
				# We first choose p so that the midpoints mix exactly:
				estAleph2 = estAlephs2[i2]
				mid2 = midpoint(estAleph2)
				p = relativePosition(mid1, midTarget, mid2)

				# Now we find the largest relative size of aleph1 and aleph2 
				# so that their mixture is still contained in aleph4state:
				# we want aleph1Lo:p:aleph2Lo >= alephLo and aleph1Hi:p:aleph2Hi <= alephHi
				# where aleph1Lo = mid1 - x * w1, aleph1Hi = mid1 + x * w1, 
				#		 aleph2Lo = mid2 - x * w2, aleph2Hi = mid2 + x * w2,
				# hence midTarget - x * w1:p:w2 >= alephLo and midTarget + x * w1:p:w2 <= alephHi,
				# i.e., x <= (midTarget - alephLo) / (w1:p:w2) and x <= (alephHi - midTarget) / (w1:p:w2): 
				w1 = estAleph1[1] - estAleph1[0]
				w2 = estAleph2[1] - estAleph2[0]
				w = interpolate(w1, p, w2)
				x = min((midTarget - alephLo) / w, (alephHi - midTarget) / w) if (w > 0) else 0
				aleph1 = [mid1 - x * w1, mid1 + x * w1]
				aleph2 = [mid2 - x * w2, mid2 + x * w2]
						
				if debug:
					print("| localPolicyData",prettyState(state), aleph4state, {"a1": a1, "estAleph1": estAleph1, "adm1": adm1, "w1": w1, "a2": a2, "estAleph2": estAleph2, "adm2": adm2, "w2": w2, "p": p, "w": w, "x": x, "aleph1": aleph1, "aleph2": aleph2})

				if verbose or debug:
					print(pad(state),"| | locPol, state",prettyState(state),"aleph4state",aleph4state,": a1,p,a2",a1,p,a2,"adm12",adm1,adm2,"aleph12",aleph1,aleph2)

				return distribution.categorical([(a1, aleph1), (a2, aleph2)], [1-p, p]).sample()

				# TODO: understand why in GW3 we go right with 50% probability rather than always going left.

			# TODO: there is an optimization but it doesn't work yet

		locPol = distribution.infer(sample)

		support = locPol.support()
		ps = [max(1e-100, math.exp(locPol.score(item))) for item in support] # 1e-100 prevents normalization problems

		if verbose or debug:
			print(pad(state),"| localPolicy, state",prettyState(state),"aleph",aleph,":");
			_W.printPolicy(pad(state), support, ps)
		return [support, ps]

	# Propagate aspiration from state-action to successor state, potentially taking into account received expected delta:

	# caching this easy to compute function would only clutter the cache due to its many arguments
	def propagateAspiration(state, action, aleph4action, Edel, nextState):
		if debug:
			print(pad(state),"| | | propagateAspiration, state",prettyState(state),"action",action,"aleph4action",aleph4action,"Edel",Edel,"nextState",prettyState(nextState),"...")

		# compute the relative position of aleph4action in the expectation that we had of 
		#	delta + next admissibility interval 
		# before we knew which state we would land in:
		lam = relativePosition(minAdmissibleQ(state, action), aleph4action, maxAdmissibleQ(state, action));
		# (this is two numbers between 0 and 1.)
		# use it to rescale aleph4action to the admissibility interval of the state that we landed in:
		rescaledAleph4nextState = interpolate(minAdmissibleV(nextState), lam, maxAdmissibleV(nextState))
		# (only this part preserves aspiration in expectation)
		res = rescaledAleph4nextState # WAS: interpolate(steadfastAleph4nextState, rescaling4Successors, rescaledAleph4nextState)
		if verbose or debug:
			print(pad(state),"| | | propagateAspiration, state",prettyState(state),"action",action,"aleph4action",aleph4action,"Edel",Edel,"nextState",prettyState(nextState),":",res)
		return res

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

	# Based on the policy, we can compute many resulting quantities of interest useful in assessing safety
	# better than with the above myopic safety metrics. All of them satisfy Bellman-style equations:

	# Actual Q and V functions of resulting policy (always returning scalars):
	@lru_cache(maxsize=None)
	def Q(self, state, action, aleph4action): # recursive
		if debug:
			print("| Q", prettyState(state), action, aleph4action)

		def sample():
			Edel = expectedDelta(state, action)
			if state.terminateAfterAction:
				return Edel;
			else:
				nextState = transition(state, action)
				nextAleph4state = propagateAspiration(state, action, aleph4action, Edel, nextState)
				return Edel + V(nextState, nextAleph4state) # recursion
		q = distribution.infer(sample).E()

		if debug:
			print("| Q", prettyState(state), action, aleph4action, q)
		return q

	@lru_cache(maxsize=None)
	def V(self, state, aleph4state): # recursive
		if debug:
			print("| V", prettyState(state), aleph4state)

		locPol = localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample() # recursion
			return Q(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v = distribution.infer(sample).E()

		if debug:
			print("| V", prettyState(state), aleph4state, ":", v)
		return v;

	# Raw moments of delta: 
	@lru_cache(maxsize=None)
	def expectedDeltaSquared(self, state, action):
		Edel = expectedDelta(state, action)
		return varianceOfDelta(state, action) + squared(Edel)
	@lru_cache(maxsize=None)
	def expectedDeltaCubed(self, state, action):
		Edel = expectedDelta(state, action)
		varDel = varianceOfDelta(state, action)
		return (varDel ** 1.5)*skewnessOfDelta(state, action) \
			+ 3*Edel*varDel \
			+ (Edel ** 3)
	@lru_cache(maxsize=None)
	def expectedDeltaFourth(self, s, a):
		Edel = expectedDelta(s, a)
		return squared(varianceOfDelta(s, a)) * (3 + excessKurtosisOfDelta(s, a)) \
			+ 4*expectedDeltaCubed(s, a)*Edel \
			- 6*expectedDeltaSquared(s, a)*squared(Edel) \
			+ 3*(Edel ** 4)
	@lru_cache(maxsize=None)
	def expectedDeltaFifth(self, s, a):
		Edel = expectedDelta(s, a)
		return fifthMomentOfDelta(s, a) \
			+ 5*expectedDeltaFourth(s, a)*Edel \
			- 10*expectedDeltaCubed(s, a)*squared(Edel) \
			+ 10*expectedDeltaSquared(s, a)*cubed(Edel) \
			- 4*(Edel ** 5)
	@lru_cache(maxsize=None)
	def expectedDeltaSixth(self, s, a):
		Edel = expectedDelta(s, a)
		return sixthMomentOfDelta(s, a) \
			+ 6*expectedDeltaFifth(s, a)*Edel \
			- 15*expectedDeltaFourth(s, a)*squared(Edel) \
			+ 20*expectedDeltaCubed(s, a)*cubed(Edel) \
			- 15*expectedDeltaSquared(s, a)*(Edel ** 4) \
			+ 5*(Edel ** 6)

	# Expected squared total, for computing the variance of total:
	@lru_cache(maxsize=None)
	def Q2(self, state, action, aleph4action): # recursive
		def sample():
			Edel = expectedDelta(state, action)
			Edel2 = expectedDeltaSquared(state, action)
			if state.terminateAfterAction:
				return Edel2;
			else:
				nextState = transition(state, action)
				nextAleph4state = propagateAspiration(state, action, aleph4action, Edel, nextState)
				# TODO: verify formula:
				return Edel2 \
					+ 2*Edel*V(nextState, nextAleph4state) \
					+ V2(nextState, nextAleph4state) # recursion

		q2 = distribution.infer(sample).E()
		if debug:
			print("| Q2", prettyState(state), action, aleph4action, q2)
		return q2
	@lru_cache(maxsize=None)
	def V2(self, state, aleph4state): # recursive
		locPol = localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample() # recursion
			return Q2(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v2 = distribution.infer(sample).E()
		if debug:
			print("| V2", prettyState(state), aleph4state, v2)
		return v2

	# Similarly: Expected third and fourth powers of total, for computing the 3rd and 4th centralized moment of total:
	@lru_cache(maxsize=None)
	def Q3(self, state, action, aleph4action): # recursive
		def sample():
			Edel = expectedDelta(state, action)
			Edel2 = expectedDeltaSquared(state, action)
			Edel3 = expectedDeltaCubed(state, action)
			if state.terminateAfterAction:
				return Edel3
			else:
				nextState = transition(state, action),
				nextAleph4state = propagateAspiration(state, action, aleph4action, Edel, nextState)
				# TODO: verify formula:
				return Edel3 \
					+ 3*Edel2*V(nextState, nextAleph4state) \
					+ 3*Edel*V2(nextState, nextAleph4state) \
					+ V3(nextState, nextAleph4state) # recursion
		q3 = distribution.infer(sample).E()
		if debug:
			print("| Q3", prettyState(state), action, aleph4action, q3)
		return q3
	@lru_cache(maxsize=None)
	def V3(self, state, aleph4state): # recursive
		locPol = localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample() # recursion
			return Q3(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v3 = distribution.infer(sample).E()
		if debug:
			print("| V3", prettyState(state), aleph4state, v3)
		return v3

	# Expected fourth power of total, for computing the expected fourth power of deviation of total from expected total (= fourth centralized moment of total):
	@lru_cache(maxsize=None)
	def Q4(self, state, action, aleph4action): # recursive
		def sample():
			Edel = expectedDelta(state, action),
			Edel2 = expectedDeltaSquared(state, action)
			Edel3 = expectedDeltaCubed(state, action)
			Edel4 = expectedDeltaFourth(state, action)
			if state.terminateAfterAction:
				return Edel4;
			else:
				nextState = transition(state, action),
				nextAleph4state = propagateAspiration(state, action, aleph4action, Edel, nextState)
				# TODO: verify formula:
				return Edel4 \
					+ 4*Edel3*V(nextState, nextAleph4state) \
					+ 6*Edel2*V2(nextState, nextAleph4state) \
					+ 4*Edel*V3(nextState, nextAleph4state) \
					+ V4(nextState, nextAleph4state) # recursion
		q4 = distribution.infer(sample).E()
		if debug:
			print("| Q4", prettyState(state), action, aleph4action, q4)
		return q4
	def V4(self, state, aleph4state): # recursive
		locPol = localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample() # recursion
			return Q4(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v4 = distribution.infer(sample).E()
		if debug:
			print("| V4", prettyState(state), aleph4state, v4)
		return v4

	# Expected fifth power of total, for computing the bed-and-banks loss component based on a 6th order polynomial potential of this shape: https://www.wolframalpha.com/input?i=plot+%28x%2B1%29%C2%B3%28x-1%29%C2%B3+ :
	def Q5(self, state, action, aleph4action): # recursive
		def sample():
			Edel = expectedDelta(state, action)
			Edel2 = expectedDeltaSquared(state, action)
			Edel3 = expectedDeltaCubed(state, action)
			Edel4 = expectedDeltaFourth(state, action)
			Edel5 = expectedDeltaFifth(state, action)
			if state.terminateAfterAction:
				return Edel5;
			else:
				nextState = transition(state, action)
				nextAleph4state = propagateAspiration(state, action, aleph4action, Edel, nextState)
				# TODO: verify formula:
				return Edel5 \
					+ 5*Edel4*V(nextState, nextAleph4state) \
					+ 10*Edel3*V2(nextState, nextAleph4state) \
					+ 10*Edel2*V3(nextState, nextAleph4state) \
					+ 5*Edel*V4(nextState, nextAleph4state) \
					+ V5(nextState, nextAleph4state) # recursion
		q5 = distribution.infer(sample).E()
		if debug:
			print("| Q5", prettyState(state), action, aleph4action, q5)
		return q5
	@lru_cache(maxsize=None)
	def V5(self, state, aleph4state): # recursive
		locPol = localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample() # recursion
			return Q5(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v5 = distribution.infer(sample).E()
		if debug:
			print("| V5", prettyState(state), aleph4state, v5)
		return v5

	# Expected sixth power of total, for computing the bed-and-banks loss component based on a 6th order polynomial potential of this shape: https://www.wolframalpha.com/input?i=plot+%28x%2B1%29%C2%B3%28x-1%29%C2%B3+ :
	@lru_cache(maxsize=None)
	def Q6(self, state, action, aleph4action): # recursive
		def sample():
			Edel = expectedDelta(state, action)
			Edel2 = expectedDeltaSquared(state, action)
			Edel3 = expectedDeltaCubed(state, action)
			Edel4 = expectedDeltaFourth(state, action)
			Edel5 = expectedDeltaFifth(state, action)
			Edel6 = expectedDeltaSixth(state, action)
			if state.terminateAfterAction:
				return Edel6;
			else:
				nextState = transition(state, action)
				nextAleph4state = propagateAspiration(state, action, aleph4action, Edel, nextState)
				# TODO: verify formula:
				return Edel6 \
					+ 6*Edel5*V(nextState, nextAleph4state) \
					+ 15*Edel4*V2(nextState, nextAleph4state) \
					+ 20*Edel3*V3(nextState, nextAleph4state) \
					+ 15*Edel2*V4(nextState, nextAleph4state) \
					+ 6*Edel*V5(nextState, nextAleph4state) \
					+ V6(nextState, nextAleph4state) # recursion
		q6 = distribution.infer(sample).E()
		if debug:
			print("| Q6", prettyState(state), action, aleph4action, q6)
		return q6
	@lru_cache(maxsize=None)
	def V6(self, state, aleph4state): # recursive
		locPol = localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample() # recursion
			return Q6(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v6 = distribution.infer(sample).E()
		if debug:
			print("| V6", prettyState(state), aleph4state, v6)
		return v6


