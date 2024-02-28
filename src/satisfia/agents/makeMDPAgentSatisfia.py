#!/usr/bin/env python3

import math
from functools import cache, lru_cache

from util import distribution
from util.helper import *

from abc import ABC, abstractmethod

VERBOSE = False
DEBUG = False

pad = str
prettyState = str

class AspirationAgent(ABC):
	def __init__(self, params):
		"""
		If world is provided, maxAdmissibleQ, minAdmissibleQ, Q, Q2, ..., Q6 are not needed because they are computed from the world. Otherwise, these functions must be provided, e.g. as learned using some reinforcement learning algorithm. Their signature is
		- maxAdmissibleQ|minAdmissibleQ: (state, action) -> float
		- Q,Q2,...,Q6: (state, action, action, aleph4action) -> float

		disorderingPotential_action, LRAdev_action, Q_ones, Q_DeltaSquare, behaviorEntropy_action, and behaviorKLdiv_action are only needed if their respective loss coefficients 
		(lossCoeff4DP, lossCoeff4LRA, lossCoeff4Time, lossCoeff4Entropy, lossCoeff4KLdiv)
		are nonzero and no world model is provided. Their signature is
		- disorderingPotential_action: (state, action) -> float
		- LRAdev_action|Q_ones|Q_DeltaSquare: (state, action, aleph4action) -> float
		- behaviorEntropy_action|behaviorKLdiv_action: (state, actionProbability, action, aleph4action) -> float

		"""
		defaults = {
			# admissibility parameters:
			"maxLambda": 1, # upper bound on local relative aspiration in each step (must be minLambda...1)	# TODO: rename to lambdaHi
			"minLambda": 0, # lower bound on local relative aspiration in each step (must be 0...maxLambda)	# TODO: rename to lambdaLo
			# policy parameters:
			"lossTemperature": 0.1, # temperature of softmin mixture of actions w.r.t. loss, must be > 0
			# "rescaling4Actions": 0, # degree (0...1) of aspiration rescaling from state to action. (larger implies larger variance) # TODO: disable this because a value of >0 can't be taken into account in a consistent way easily
			# "rescaling4Successors": 1, # degree (0...1) of aspiration rescaling from action to successor state. (expectation is only preserved if this is 1.0) # TODO: disable also this since a value <1 leads to violation of the expectation guarantee 

			# THESE LOSS COMPONENTS DO NOT USE THE WORLD MODEL:

			# coefficients for cheap to compute loss functions:
			"lossCoeff4Random": 0, # weight of random tie breaker in loss function, must be >= 0
			"lossCoeff4FeasibilityPower": 1, # weight of power of squared admissibility interval width in loss function, must be >= 0
			"lossCoeff4LRA1": 1, # weight of current-state deviation of LRA from 0.5 in loss function, must be >= 0
			"lossCoeff4Time1": 1, # weight of not terminating in loss function, must be >= 0
			"lossCoeff4Entropy1": 1, # weight of current-state action entropy in loss function, must be >= 0
			"lossCoeff4KLdiv1": 1, # weight of current-state KL divergence in loss function, must be >= 0

			# THE FOLLOWING CAN IN PRINCIPLE ALSO COMPUTED OR LEARNED UPFRONT:

			"lossCoeff4DP": 1, # weight of disordering potential in loss function, must be >= 0

			"uninformedStatePriorScore": 0,
			"internalTransitionEntropy": 0,

			# THESE LOSS COMPONENTS USE THE WORLD MODEL BECAUSE THEY DEPEND ON THE TRANSITION FUNCTION AND THE POLICY:

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

		self.stateActionPairsSet = set()

		assert self.params["lossTemperature"] > 0, "lossTemperature must be > 0"
		#assert 0 <= rescaling4Actions <= 1, "rescaling4Actions must be in 0...1"
		#assert 0 <= rescaling4Successors <= 1, "rescaling4Successors must be in 0...1"
		assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4Random"] >= 0, "lossCoeff4random must be >= 0"
		assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4FeasibilityPower"] >= 0, "lossCoeff4FeasibilityPower must be >= 0"
		assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4DP"] >= 0, "lossCoeff4DP must be >= 0"
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
		assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4Entropy"] >= 0, "lossCoeff4entropy must be >= 0"
		assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4KLdiv"] >= 0, "lossCoeff4KLdiv must be >= 0"
		assert self.params["allowNegativeCoeffs"] or self.params["lossCoeff4OtherLoss"]	 >= 0, "lossCoeff4OtherLoss must be >= 0"
		assert self.params["lossCoeff4Entropy"] == 0 or lossCoeff4DP == 0 or ("uninformedPolicy" in self.params), "uninformedPolicy must be provided if lossCoeff4DP > 0 or lossCoeff4Entropy > 0"

		if VERBOSE or DEBUG:
			print("makeMDPAgentSatisfia with parameters", self.params)

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
		→ world.raw_moment_of_delta, varianceOfDelta, transition
		→ propagateAspiration
			→ aspiration4state
		→ simulate (RECURSION)"""

	def __getitem__(self, name):
		return self.params[name]

	#@lru_cache(maxsize=None)
	def maxAdmissibleV(self, state): # recursive
		if VERBOSE or DEBUG:
			print(pad(state), "maxAdmissibleV, state", state, "...")

		v = 0
		actions = self.possible_actions(state)
		if actions != []:
			qs = [self.maxAdmissibleQ(state, a) for a in actions] # recursion
			v = max(qs) if self["maxLambda"] == 1 else interpolate(min(qs), self["maxLambda"], max(qs))

		if VERBOSE or DEBUG:
			print(pad(state), "maxAdmissibleV, state", state, ":", v)

		return v

	#@lru_cache(maxsize=None)
	def minAdmissibleV(self, state): # recursive
		if VERBOSE or DEBUG:
			print(pad(state), "minAdmissibleV, state", state, "...")

		v = 0
		actions = self.possible_actions(state)
		if actions != []:
			qs = [self.minAdmissibleQ(state, a) for a in actions] # recursion
			v = min(qs) if self["minLambda"] == 0 else interpolate(min(qs), self["minLambda"], max(qs))

		if VERBOSE or DEBUG:
			print(pad(state), "minAdmissibleV, state", state, ":", v)

		return v

	# The resulting admissibility interval for states.
	def admissibility4state(self, state):
		return self.minAdmissibleV(state), self.maxAdmissibleV(state)

	# The resulting admissibility interval for actions.
	def admissibility4action(self, state, action):
		return self.minAdmissibleQ(state, action), self.maxAdmissibleQ(state, action)

	# When in state, we can get any expected total in the interval
	# [minAdmissibleV(state), maxAdmissibleV(state)].
	# So when having aspiration aleph, we can still fulfill it in expectation if it lies in the interval.
	# Therefore, when in state at incoming aspiration aleph,
	# we adjust our aspiration to aleph clipped to that interval:
	#@lru_cache(maxsize=None)
	def aspiration4state(self, state, unclippedAleph):
		if VERBOSE or DEBUG:
			print(pad(state),"| | aspiration4state, state",prettyState(state),"unclippedAleph",unclippedAleph,"...")
		res = clip(self.minAdmissibleV(state), Interval(unclippedAleph), self.maxAdmissibleV(state))
		if VERBOSE or DEBUG:
			print(pad(state),"| | aspiration4state, state",prettyState(state),"unclippedAleph",unclippedAleph,":",res)
		return res

	# When constructing the local policy, we first use an estimated action aspiration interval
	# that does not depend on the local policy but is simply based on the state's aspiration interval,
	# moved from the admissibility interval of the state to the admissibility interval of the action.
	#@lru_cache(maxsize=None)
	def estAspiration4action(self, state, action, aleph4state):
		if DEBUG:
			print("| | estAspiration4action, state",prettyState(state),"action",action,"aleph4state",aleph4state,"...")
		phi = self.admissibility4action(state, action)
		if isSubsetOf(phi, aleph4state):
			if VERBOSE or DEBUG:
				print(pad(state),"| | estAspiration4action, state",prettyState(state),"action",action,"aleph4state",aleph4state,":",phi,"(subset of aleph4state)")
			return phi
		else:
			""" DISBLED:
				// if rescaling4Actions == 1, we use a completely rescaled version:
				var rescaled = interpolate(minAdmissibleQ(state, action),
																	 relativePosition(minAdmissibleV(state), aleph4state, maxAdmissibleV(state)),
																	 maxAdmissibleQ(state, action));
				// if rescaling4Actions == 0, we
			"""

			# We use a steadfast version that does make sure that aleph(a) is no wider than aleph(s):
			# - If phi(a) contains aleph(s), then aleph(a) = aleph(s)
			# - If aleph(s) contains phi(a), then aleph(a) = phi(a)
			# - If phiLo(a) < alephLo(s) and phiHi(a) < alephHi(s), then aleph(a) = [max(phiLo(a), phiHi(a) - alephW(s)), phiHi(a)]
			# - If phiHi(a) > alephHi(s) and phiLo(a) > alephLo(s), then aleph(a) = [phiLo(a), min(phiHi(a), phiLo(a) + alephW(s))]
			phiLo, phiHi = phi
			alephLo, alephHi = aleph4state
			w = alephHi - alephLo
			steadfast = phi
			if isSubsetOf(aleph4state, phi):
				steadfast = aleph4state
			elif phiLo < alephLo and phiHi < alephHi:
				steadfast = [max(phiLo, phiHi - w), phiHi]
			elif phiHi > alephHi and phiLo > alephLo:
				steadfast = [phiLo, min(phiHi, phiLo + w)]
			# DISABLED: We interpolate between the two versions according to rescaling4Actions:
			res = steadfast # WAS: interpolate(steadfast, rescaling4Actions, rescaled);

			if VERBOSE or DEBUG:
				print(pad(state),"| | estAspiration4action, state",prettyState(state),"action",action,"aleph4state",aleph4state,":",res,"(steadfast)") # WAS: "(steadfast/rescaled)")
			return res

	#@lru_cache(maxsize=None)
	def disorderingPotential_state(self, state): # recursive
		actions = self.possible_actions(state)
		maxMPpolicyWeights = [math.exp(self.disorderingPotential_action(state, a)) for a in actions]
		# TODO what if the sum is 0
		return math.log(sum(maxMPpolicyWeights))

	# Based on the admissibility information computed above, we can now construct the policy,
	# which is a mapping taking a state and an aspiration interval as input and returning
	# a categorical distribution over (action, aleph4action) pairs.

	def localPolicy(self, state, aleph): # recursive
		"""return a categorical distribution over (action, aleph4action) pairs"""

		d = self.localPolicyData(state, aleph)
		support = d[0]
		ps = d[1]

		if DEBUG:
			print(pad(state), " localPolicy", prettyState(state), aleph, d)

		return distribution.categorical(support, ps)

	#@lru_cache(maxsize=None)
	def localPolicyData(self, state, aleph):
		if VERBOSE or DEBUG:
			print(pad(state), "| localPolicy, state",prettyState(state),"aleph",aleph,"...")

		# Clip aspiration interval to admissibility interval of state:
		alephLo, alephHi = aleph4state = self.aspiration4state(state, aleph)

		# Estimate aspiration intervals for all possible actions in a way
		# independent from the local policy that we are about to construct,
		actions = self.possible_actions(state)
		assert actions != []
		estAlephs1 = [self.estAspiration4action(state, action, aleph4state) for action in actions]

		# Estimate losses based on this estimated aspiration intervals
		# and use it to construct softmin propensities (probability weights) for choosing actions.
		# since we don't know the actual probabilities of actions yet (we will determine those only later),
		# but the loss estimation requires an estimate of the probability of the chosen action,
		# we estimate the probability at 1 / number of actions:
		def propensity(index, indices, estAlephs):
			action = actions[index]
			loss = self.combinedLoss(state, action, aleph4state, estAlephs[index], 1 / len(indices))
			return min(1e100, max(math.exp(-loss / self["lossTemperature"]), 1e-100))

		p_effective = {}

		def probability_add(p, key, weight):
			if weight < 0:
				raise ValueError("invalid weight")

			if weight == 0:
				if key in p:
					del p[key]
			elif key in p:
				p[key] += weight
			else:
				p[key] = weight

		indices = list(range(len(actions)))
		propensities = [propensity(index, indices, estAlephs1) for index in indices] # bottleneck

		if DEBUG:
			print("| localPolicyData", prettyState(state), aleph, actions, {propensities})

		for i1, p1 in distribution.categorical(indices, propensities).categories():
			# Get admissibility interval for the first action.
			a1 = actions[i1]
			adm1Lo, adm1Hi = adm1 = self.admissibility4action(state, a1)

			# If a1's admissibility interval is completely contained in aleph4state, we are done:
			if Interval(adm1) <= Interval(aleph4state):
				if VERBOSE or DEBUG:
					print(pad(state),"| | locPol, state",prettyState(state),"aleph4state",aleph4state,": a1",a1,"adm1",adm1,"(subset of aleph4state)")
				probability_add(p_effective, (a1, adm1), p1)
			else:
				# For the second action, restrict actions so that the the midpoint of aleph4state can be mixed from
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
				estAlephs2 = [self.estAspiration4action(state, actions[index], aleph2target) for index in indices]
				propensities2 = [propensity(index, indices2, estAlephs2) for index in indices2]

				if DEBUG:
					print("| localPolicyData", prettyState(state), aleph4state, {"a1": a1, "midTarget": midTarget, "estAleph1": estAleph1, "mid1": mid1, "indices2": indices2, "aleph2target": aleph2target, "estAlephs2": estAlephs2, "propensities2": propensities2})

				for i2, p2 in distribution.categorical(indices2, propensities2).categories():
					# Get admissibility interval for the second action.
					a2 = actions[i2]
					adm2Lo, adm2Hi = adm2 = self.admissibility4action(state, a2)

					# Now we need to find two aspiration intervals aleph1 in adm1 and aleph2 in adm2,
					# and a probability p such that
					#	 aleph1:p:aleph2 is contained in aleph4state
					# and aleph1, aleph2 are close to the estimates we used above in estimating loss.
					# Instead of optimizing this, we use the following heuristic:
					# We first choose p so that the midpoints mix exactly:
					estAleph2 = estAlephs2[i2]
					mid2 = midpoint(estAleph2)
					p = relativePosition(mid1, midTarget, mid2)
					# TODO try clipping

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
					aleph1 = Interval(mid1 - x * w1, mid1 + x * w1)
					aleph2 = Interval(mid2 - x * w2, mid2 + x * w2)

					if DEBUG:
						print("| localPolicyData",prettyState(state), aleph4state, {"a1": a1, "estAleph1": estAleph1, "adm1": adm1, "w1": w1, "a2": a2, "estAleph2": estAleph2, "adm2": adm2, "w2": w2, "p": p, "w": w, "x": x, "aleph1": aleph1, "aleph2": aleph2})

					if VERBOSE or DEBUG:
						print(pad(state),"| | locPol, state",prettyState(state),"aleph4state",aleph4state,": a1,p,a2",a1,p,a2,"adm12",adm1,adm2,"aleph12",aleph1,aleph2)

					probability_add(p_effective, (a1, aleph1), (1 - p) * p1 * p2)
					probability_add(p_effective, (a2, aleph2), p * p1 * p2)

		# now we can construct the local policy as a WebPPL distribution object:
		locPol = distribution.categorical(p_effective)

		support = locPol.support()
		ps = [max(1e-100, math.exp(locPol.score(item))) for item in support] # 1e-100 prevents normalization problems

		if VERBOSE or DEBUG:
			print(pad(state),"| localPolicy, state",prettyState(state),"aleph",aleph,":")
			_W.printPolicy(pad(state), support, ps)
		return [support, ps]

	# Propagate aspiration from state-action to successor state, potentially taking into account received expected delta:

	# caching this easy to compute function would only clutter the cache due to its many arguments
	def propagateAspiration(self, state, action, aleph4action, Edel, nextState):
		if DEBUG:
			print(pad(state),"| | | propagateAspiration, state",prettyState(state),"action",action,"aleph4action",aleph4action,"Edel",Edel,"nextState",prettyState(nextState),"...")

		# compute the relative position of aleph4action in the expectation that we had of 
		#	delta + next admissibility interval 
		# before we knew which state we would land in:
		lam = relativePosition(self.minAdmissibleQ(state, action), aleph4action, self.maxAdmissibleQ(state, action)) # TODO didn't we calculate the admissible Q when we chose the action?
		# (this is two numbers between 0 and 1.)
		# use it to rescale aleph4action to the admissibility interval of the state that we landed in:
		rescaledAleph4nextState = interpolate(self.minAdmissibleV(nextState), lam, self.maxAdmissibleV(nextState))
		# (only this part preserves aspiration in expectation)
		res = rescaledAleph4nextState # WAS: interpolate(steadfastAleph4nextState, rescaling4Successors, rescaledAleph4nextState)
		if VERBOSE or DEBUG:
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

	#@lru_cache(maxsize=None)
	def V(self, state, aleph4state): # recursive
		if DEBUG:
			print("| V", prettyState(state), aleph4state)

		locPol = self.localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample()[0] # recursion
			return self.Q(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v = distribution.infer(sample).E()

		if DEBUG:
			print("| V", prettyState(state), aleph4state, ":", v)
		return v

	#@lru_cache(maxsize=None)
	def V2(self, state, aleph4state): # recursive
		locPol = self.localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample()[0] # recursion
			return self.Q2(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v2 = distribution.infer(sample).E()
		if DEBUG:
			print("| V2", prettyState(state), aleph4state, v2)
		return v2

	#@lru_cache(maxsize=None)
	def V3(self, state, aleph4state): # recursive
		locPol = self.localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample()[0] # recursion
			return self.Q3(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v3 = distribution.infer(sample).E()
		if DEBUG:
			print("| V3", prettyState(state), aleph4state, v3)
		return v3

	#@lru_cache(maxsize=None)
	def V4(self, state, aleph4state): # recursive
		locPol = self.localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample()[0] # recursion
			return self.Q4(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v4 = distribution.infer(sample).E()
		if DEBUG:
			print("| V4", prettyState(state), aleph4state, v4)
		return v4

	#@lru_cache(maxsize=None)
	def V5(self, state, aleph4state): # recursive
		locPol = self.localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample()[0] # recursion
			return self.Q5(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v5 = distribution.infer(sample).E()
		if DEBUG:
			print("| V5", prettyState(state), aleph4state, v5)
		return v5

	#@lru_cache(maxsize=None)
	def V6(self, state, aleph4state): # recursive
		locPol = self.localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample()[0] # recursion
			return self.Q6(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		v6 = distribution.infer(sample).E()
		if DEBUG:
			print("| V6", prettyState(state), aleph4state, v6)
		return v6

	# Expected powers of difference between total and some target value v,
	# needed for estimating moments of probabilistic policies in loss function,
	# where v will be an estimate of V(state):

	#@lru_cache(maxsize=None)
	def relativeQ2(self, s, a, al, v): # aleph4action
		res = self.Q2(s,a,al) \
			- 2*self.Q(s,a,al)*v \
			+ v ** 2
		if DEBUG:
			print("| Q2v", prettyState(s), a, al, v, res)
		return res

	#@lru_cache(maxsize=None)
	def relativeQ4(self, s, a, al, v):
		return self.Q4(s,a,al) \
			- 4*self.Q3(s,a,al)*v \
			+ 6*self.Q2(s,a,al)*(v ** 2) \
			- 4*self.Q(s,a,al)*(v ** 3) \
			+ v ** 4

	#@lru_cache(maxsize=None)
	def relativeQ6(self, s, a, al, v):
		return self.Q6(s,a,al) \
			- 6*self.Q5(s,a,al)*v \
			+ 15*self.Q4(s,a,al)*(v ** 2) \
			- 20*self.Q3(s,a,al)*(v ** 3) \
			+ 15*self.Q2(s,a,al)*(v ** 4) \
			- 6*self.Q(s,a,al)*(v ** 5) \
			+ v ** 6

	# TODO: the following should maybe better be w.r.t. the initial aspiration interval, not the current state's:

	# loss based on a "cup" shaped potential centered at the mid-point of the aspiration interval
	# that is almost completely flat in the middle half of the interval 
	# (https://www.wolframalpha.com/input?i=plot+%28x-.5%29%5E6+from+0+to+1):

	#@lru_cache(maxsize=None)
	def cupLoss_action(self, state, action, aleph4state, aleph4action):
		res = self.relativeQ6(state, action, aleph4action, midpoint(aleph4state))
		if DEBUG:
			print("| cupLoss_action", prettyState(state), action, aleph4state, res)
		return res
	#@lru_cache(maxsize=None)
	def cupLoss_state(self, state, unclippedAleph): # recursive
		aleph4state = self.aspiration4state(state, unclippedAleph)
		locPol = self.localPolicy(state, aleph4state) # recursion
		def sample():
			actionAndAleph = locPol.sample()[0]
			return self.cupLoss_action(state, actionAndAleph[0], aleph4state, actionAndAleph[1])
		return distribution.infer(sample).E()

	#@lru_cache(maxsize=None)
	def LRAdev_state(self, state, aleph4state): # recursive
		locPol = self.localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample()[0] # recursion
			return self.LRAdev_action(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		return distribution.infer(sample).E()

	#@lru_cache(maxsize=None)
	def V_ones(self, state, aleph4state): # recursive
		locPol = self.localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample()[0] # recursion
			return self.Q_ones(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		if DEBUG:
			print("| V_ones", prettyState(state), aleph4state, v_ones)
		v_ones = distribution.infer(sample).E()
		return v_ones

	#@lru_cache(maxsize=None)
	def V_DeltaSquare(self, state, aleph4state): # recursive
		locPol = self.localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample()[0] # recursion
			return self.Q_DeltaSquare(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		if DEBUG:
			print("| V_DeltaSquare", prettyState(state), aleph4state, vDsq)
		vDsq = distribution.infer(sample).E()
		return vDsq

	#@lru_cache(maxsize=None)
	def behaviorEntropy_state(self, state, aleph4state): # recursive
		locPol = self.localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample()[0] # recursion
			return self.behaviorEntropy_action(state, math.exp(locPol.score(actionAndAleph)), actionAndAleph[0], actionAndAleph[1]) # recursion
		return distribution.infer(sample).E()

	#@lru_cache(maxsize=None)
	def behaviorKLdiv_state(self, state, aleph4state): # recursive
		locPol = self.localPolicy(state, aleph4state)
		def sample():
			actionAndAleph = locPol.sample()[0] # recursion
			return self.behaviorKLdiv_action(state, math.exp(locPol.score(actionAndAleph)), actionAndAleph[0], actionAndAleph[1]) # recursion
		return distribution.infer(sample).E()

	#@lru_cache(maxsize=None)
	def otherLoss_state(self, state, aleph4state): # recursive
		locPol = self.localPolicy(state, aleph4state) # recursion
		def sample():
			actionAndAleph = locPol.sample()[0]
			return self.otherLoss_action(state, actionAndAleph[0], actionAndAleph[1]) # recursion
		return distribution.infer(sample).E()

	@cache
	def randomTieBreaker(self, state, action):
		return random.random()

	# now we can combine all of the above quantities to a combined (safety) loss function:

	# state, action, aleph4state, aleph4action, estActionProbability
	#@lru_cache(maxsize=None)
	def combinedLoss(self, s, a, al4s, al4a, p): # recursive
		def expr_params(expr, *params, default=0):
			args = [self.params[param] for param in params]
			return expr(*args) if any(args) else default

		if VERBOSE or DEBUG:
			print(pad(s),"| | combinedLoss, state",prettyState(s),"action",a,"aleph4state",al4s,"aleph4action",al4a,"estActionProbability",p,"...")

		# cheap criteria, including some myopic versions of the more expensive ones:
		lRandom = expr_params(lambda l: l * self.randomTieBreaker(s, a), "lossCoeff4Random")
		lFeasibilityPower = expr_params(lambda l: l * (self.maxAdmissibleQ(s, a) - self.minAdmissibleQ(s, a)) ** 2, "lossCoeff4FeasibilityPower")
		lMP = expr_params(lambda l: l * self.disorderingPotential_action(s, a), "lossCoeff4DP")
		lLRA1 = expr_params(lambda l: l * self.LRAdev_action(s, a, al4a, True), "lossCoeff4LRA1")
		lEntropy1 = expr_params(lambda l: l * self.behaviorEntropy_action(s, p, a), "lossCoeff4Entropy1")
		lKLdiv1 = expr_params(lambda l: l * self.behaviorKLdiv_action(s, p, a), "lossCoeff4KLdiv1")

		# moment-based criteria:
		# (To compute expected powers of deviation from V(s), we cannot use the actual V(s) 
		# because we don't know the local policy at s yet. Hence we use a simple estimate based on aleph4state)
		estVs = midpoint(al4s)
		lVariance = expr_params(lambda l: l * self.relativeQ2(s, a, al4a, estVs), "lossCoeff4Variance") # recursion
		lFourth = expr_params(lambda l: l * self.relativeQ4(s, a, al4a, estVs), "lossCoeff4Fourth") # recursion
		lCup = expr_params(lambda l: l * self.cupLoss_action(s, a, al4s, al4a), "lossCoeff4Cup") # recursion
		lLRA = expr_params(lambda l: l * self.LRAdev_action(s, a, al4a), "lossCoeff4LRA") # recursion

		# timing-related criteria:
		q_ones = expr_params(lambda x, y: Q_ones(s, a, al4a), "lossCoeff4DeltaVariation", "lossCoeff4Time")
		lTime = expr_params(lambda l: l * q_ones, "lossCoeff4Time")
		lDeltaVariation = 0
		if q_ones != 0:
			lDeltaVariation = expr_params(lambda l: l * (self.Q_DeltaSquare(s, a, al4a) / q_ones - self.Q2(s, a, al4a) / (q_ones ** 2)), "lossCoeff4DeltaVariation") # recursion

		# randomization-related criteria:
		lEntropy = expr_params(lambda l: l * self.behaviorEntropy_action(s, p, a, al4a), "lossCoeff4Entropy") # recursion
		lKLdiv = expr_params(lambda l: l * self.behaviorKLdiv_action(s, p, a, al4a), "lossCoeff4KLdiv") # recursion

		lOther = 0
		if "otherLocalLoss" in self.params:
			lOther = expr_params(lambda l: l * self.otherLoss_action(s, a, al4a), "lossCoeff4OtherLoss") # recursion

		res = lRandom + lFeasibilityPower + lMP + lLRA1 + self["lossCoeff4Time1"] + lEntropy1 + lKLdiv1 \
							+ lVariance + lFourth + lCup + lLRA \
							+ lTime + lDeltaVariation \
							+ lEntropy + lKLdiv \
							+ lOther
		if VERBOSE or DEBUG:
			print(pad(s),"| | combinedLoss, state",prettyState(s),"action",a,"aleph4state",al4s,"aleph4action",al4a,"estActionProbability",p,":",res,"\n"+pad(s),"| |	", json.dumps({
				"lRandom": lRandom,
				"lFeasibilityPower": lFeasibilityPower,
				"lMP": lMP,
				"lLRA1": lLRA1,
				"lTime1": self["lossCoeff4Time1"],
				"lEntropy1": lEntropy1,
				"lKLdiv1": lKLdiv1,
				"lVariance": lVariance,
				"lFourth": lFourth,
				"lCup": lCup,
				"lLRA": lLRA,
				"lTime": lTime,
				"lDeltaVariation": lDeltaVariation,
				"lEntropy": lEntropy,
				"lKLdiv": lKLdiv,
				"lOther": lOther
			}))
		return res

	def getData():
		return {
			"stateActionPairs": list(self.stateActionPairsSet),
			"states": list({pair[0] for pair in self.stateActionPairsSet}),
			"locs": [state.loc for state in states],
		}

	@abstractmethod
	def maxAdmissibleQ(self, state, action): pass
	@abstractmethod
	def minAdmissibleQ(self, state, action): pass
	@abstractmethod
	def disorderingPotential_action(self, state, action): pass

	@abstractmethod
	def LRAdev_action(self, state, action, aleph4action, myopic): pass
	@abstractmethod
	def behaviorEntropy_action(self, state, actionProbability, action, aleph4action): pass
	@abstractmethod
	def behaviorKLdiv_action(self, state, actionProbability, action, aleph4action): pass
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
	def possible_actions(self, state, action): pass

class AgentMDPLearning(AspirationAgent):
	def __init__(self, params, maxAdmissibleQ=None, minAdmissibleQ=None, 
			disorderingPotential_action=None,
			LRAdev_action=None, Q_ones=None, Q_DeltaSquare=None, behaviorEntropy_action=None, behaviorKLdiv_action=None, otherLoss_action=None,
			Q=None, Q2=None, Q3=None, Q4=None, Q5=None, Q6=None,
			possible_actions=None):
		super().__init__(params)

		self.maxAdmissibleQ = maxAdmissibleQ
		self.minAdmissibleQ = minAdmissibleQ
		self.disorderingPotential_action = disorderingPotential_action

		self.LRAdev_action = LRAdev_action
		self.behaviorEntropy_action = behaviorEntropy_action
		self.behaviorKLdiv_action = behaviorKLdiv_action
		self.otherLoss_action = otherLoss_action

		self.Q = Q
		self.Q2 = Q2
		self.Q3 = Q3
		self.Q4 = Q4
		self.Q5 = Q5
		self.Q6 = Q6

		self.Q_ones = Q_ones
		self.Q_DeltaSquare = Q_DeltaSquare

		self.possible_actions = possible_actions

class AgentMDPPlanning(AspirationAgent):
	def __init__(self, params, world=None):
		self.world = world
		super().__init__(params)

	def possible_actions(self, state):
		if self.world.is_terminal(state):
			return []
		return self.world.possible_actions(state)

	# Compute upper and lower admissibility bounds for Q and V that are allowed in view of maxLambda and minLambda:

	# Compute the Q and V functions of the classical maximization problem (if maxLambda==1)
	# or of the LRA-based problem (if maxLambda<1):

	#@lru_cache(maxsize=None)
	def maxAdmissibleQ(self, state, action): # recursive
		if VERBOSE or DEBUG:
			print(pad(state), "maxAdmissibleQ, state", state, "action", action, "...")

		# register (state, action) in global store (could be anywhere, but here is just as fine as anywhere else)
		self.stateActionPairsSet.add((state, action))

		Edel = self.world.raw_moment_of_delta(state, action)
		# Bellman equation
		q = Edel + self.world.expectation(state, action, self.maxAdmissibleV) # recursion

		if VERBOSE or DEBUG:
			print(pad(state), "maxAdmissibleQ, state", state, "action", action, ":", q)

		return q

	#@lru_cache(maxsize=None)
	def minAdmissibleQ(self, state, action): # recursive
		if VERBOSE or DEBUG:
			print(pad(state), "minAdmissibleQ, state", state, "action", action, "...")

		# register (state, action) in global store (could be anywhere, but here is just as fine as anywhere else)
		self.stateActionPairsSet.add((state, action))

		Edel = self.world.raw_moment_of_delta(state, action)
		# Bellman equation
		q = Edel + self.world.expectation(state, action, self.minAdmissibleV) # recursion

		if VERBOSE or DEBUG:
			print(pad(state), "minAdmissibleQ, state", state, "action", action, ":", q)

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

	# Disordering potential (maximal entropy (relative to some uninformedStatePrior) 
	# over trajectories any agent could produce from here (see overleaf for details)):
	#@lru_cache(maxsize=None)
	def disorderingPotential_action(self, state, action): # recursive
		def sample(nextState, probability):
			if self.world.is_terminal(nextState):
				return 0
			else:
				nextMP = self.disorderingPotential_state(nextState) # recursion
				priorScore = self["uninformedStatePriorScore"](nextState) if self["uninformedStatePriorScore"] else 0
				internalEntropy = self["internalTransitionEntropy"](state, action, nextState) if self["internalTransitionEntropy"] else 0
				return nextMP + priorScore - math.log(probability) + internalEntropy

		# Note for ANN approximation: disorderingPotential_action can be positive or negative. 
		return self.world.expectation_of_fct_of_probability(state, action, sample)


	# TODO: IMPLEMENT A LEARNING VERSION OF THIS FUNCTION:

	# Based on the policy, we can compute many resulting quantities of interest useful in assessing safety
	# better than with the above myopic safety metrics. All of them satisfy Bellman-style equations:

	# Actual Q and V functions of resulting policy (always returning scalars):
	#@lru_cache(maxsize=None)
	def Q(self, state, action, aleph4action): # recursive
		if DEBUG:
			print("| Q", prettyState(state), action, aleph4action)

		Edel = self.world.raw_moment_of_delta(state, action)
		def total(nextState):
			if self.world.is_terminal(nextState):
				return Edel
			else:
				nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
				return Edel + self.V(nextState, nextAleph4state) # recursion
		q = self.world.expectation(state, action, total)

		if DEBUG:
			print("| Q", prettyState(state), action, aleph4action, q)
		return q

	# Expected squared total, for computing the variance of total:
	#@lru_cache(maxsize=None)
	def Q2(self, state, action, aleph4action): # recursive
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

		if DEBUG:
			print("| Q2", prettyState(state), action, aleph4action, q2)
		return q2

	# Similarly: Expected third and fourth powers of total, for computing the 3rd and 4th centralized moment of total:
	#@lru_cache(maxsize=None)
	def Q3(self, state, action, aleph4action): # recursive
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

		if DEBUG:
			print("| Q3", prettyState(state), action, aleph4action, q3)
		return q3

	# Expected fourth power of total, for computing the expected fourth power of deviation of total from expected total (= fourth centralized moment of total):
	#@lru_cache(maxsize=None)
	def Q4(self, state, action, aleph4action): # recursive
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

		if DEBUG:
			print("| Q4", prettyState(state), action, aleph4action, q4)
		return q4

	# Expected fifth power of total, for computing the bed-and-banks loss component based on a 6th order polynomial potential of this shape: https://www.wolframalpha.com/input?i=plot+%28x%2B1%29%C2%B3%28x-1%29%C2%B3+ :
	def Q5(self, state, action, aleph4action): # recursive
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

		if DEBUG:
			print("| Q5", prettyState(state), action, aleph4action, q5)
		return q5

	# Expected sixth power of total, for computing the bed-and-banks loss component based on a 6th order polynomial potential of this shape: https://www.wolframalpha.com/input?i=plot+%28x%2B1%29%C2%B3%28x-1%29%C2%B3+ :
	#@lru_cache(maxsize=None)
	def Q6(self, state, action, aleph4action): # recursive
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

		if DEBUG:
			print("| Q6", prettyState(state), action, aleph4action, q6)
		return q6

	# Squared deviation of local relative aspiration (midpoint of interval) from 0.5:
	#@lru_cache(maxsize=None)
	def LRAdev_action(self, state, action, aleph4action, myopic): # recursive
		# Note for ANN approximation: LRAdev_action must be between 0 and 0.25 
		Edel = self.world.raw_moment_of_delta(state, action)

		def dev(nextState):
			localLRAdev = (0.5 - relativePosition(self.minAdmissibleQ(state, action), midpoint(aleph4action), self.maxAdmissibleQ(state, action))) ** 2
			if self.world.is_terminal(nextState) or myopic:
				return localLRAdev
			else:
				nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
				return localLRAdev + self.LRAdev_state(nextState, nextAleph4state) # recursion
		return self.world.expectation(state, action, dev)

	# TODO: verify the following two formulas for expected Delta variation along a trajectory:

	# Expected total of ones (= expected length of trajectory), for computing the expected Delta variation along a trajectory:
	#@lru_cache(maxsize=None)
	def Q_ones(self, state, action, aleph4action=None): # recursive
		Edel = self.world.raw_moment_of_delta(state, action)

		# Note for ANN approximation: Q_ones must be nonnegative. 
		def one(nextState):
			if self.world.is_terminal(nextState) or aleph4action == None:
				return 1
			else:
				nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
				return 1 + self.V_ones(nextState, nextAleph4state) # recursion
		q_ones = self.world.expectation(state, action, one)

		if DEBUG:
			print("| Q_ones", prettyState(state), action, aleph4action, q_ones)
		return q_ones

	# Expected total of squared Deltas, for computing the expected Delta variation along a trajectory:
	#@lru_cache(maxsize=None)
	def Q_DeltaSquare(self, state, action, aleph4action=None): # recursive
		Edel = self.world.raw_moment_of_delta(state, action)
		EdelSq = Edel**2 + self["varianceOfDelta"](state, action)

		# Note for ANN approximation: Q_DeltaSquare must be nonnegative. 
		def d(nextState):
			if self.world.is_terminal(nextState) or aleph4action == None:
				return EdelSq
			else:
				nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
				return EdelSq + self.V_DeltaSquare(nextState, nextAleph4state) # recursion
		if DEBUG:
			print("| Q_DeltaSquare", prettyState(state), action, aleph4action, qDsq)
		qDsq = self.world.expectation(state, action, d)
		return qDsq

	# Other safety criteria:

	# Shannon entropy of behavior
	# (actually, negative KL divergence relative to uninformedPolicy (e.g., a uniform distribution),
	# to be consistent under action cloning or action refinement):
	#@lru_cache(maxsize=None)
	def behaviorEntropy_action(self, state, actionProbability, action, aleph4action=None): # recursive
		# Note for ANN approximation: behaviorEntropy_action must be <= 0 (!) 
		# because it is the negative (!) of a KL divergence. 
		Edel = self.world.raw_moment_of_delta(state, action)
		def entropy(nextState):
			uninfPolScore = self["uninformedPolicy"](state).score(action) if ("uninformedPolicy" in self.params) else 0
			localEntropy = uninfPolScore \
							- math.log(actionProbability) \
							+ self["internalActionEntropy"](state, action) if ("internalActionEntropy" in self.params) else 0
			if self.world.is_terminal(nextState) or aleph4action == None:
				return localEntropy
			else:
				nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
				return localEntropy + self.behaviorEntropy_state(nextState, nextAleph4state) # recursion
		return self.world.expectation(state, action, entropy)

	# KL divergence of behavior relative to refPolicy (or uninformedPolicy if refPolicy is not set):
	#@lru_cache(maxsize=None)
	def behaviorKLdiv_action(self, state, actionProbability, action, aleph4action=None): # recursive
		# Note for ANN approximation: behaviorKLdiv_action must be nonnegative. 
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
			if self.world.is_terminal(nextState) or aleph4action == None:
				return localDivergence
			else:
				nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
				return localDivergence + self.behaviorKLdiv_state(nextState, nextAleph4state) # recursion
		return self.world.expectation(state, action, div)

	# other loss:
	#@lru_cache(maxsize=None)
	def otherLoss_action(self, state, action, aleph4action=None): # recursive
		Edel = self.world.raw_moment_of_delta(state, action)
		def loss(nextState):
			#localLoss = self["otherLocalLoss"](state, action) # TODO this variable may not exist in params
			localLoss = 0 # TODO this variable may not exist in params
			if self.world.is_terminal(nextState) or aleph4action == None:
				return localLoss
			else:
				nextAleph4state = self.propagateAspiration(state, action, aleph4action, Edel, nextState)
				return localLoss + self.otherLoss_state(nextState, nextAleph4state) # recursion
		return self.world.expectation(state, action, loss)
