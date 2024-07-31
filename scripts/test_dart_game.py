#!/usr/bin/env python3

""" TODO
what are the policies in behaviorKLdiv_action
what should localLoss be
rounding errors when calculating aspiration4state break inequalities in localPolicy




environment vs world model
"""

import sys
sys.path.insert(0,'./src/')

import time
from numpy import random
import numpy as np

from environments.dart_game import DartGame
import pylab as plt
from satisfia.agents.makeMDPAgentSatisfia import AgentMDPPlanning

def move_randomly(env):
	state, info = env.reset()
	delta = terminated = 0
	for t in range(1000):
		actions = env.possible_actions(state)
		action = random.choice(actions)
		print(t, state, delta, terminated, info, actions, action)
		state, delta, terminated, _, info = env.step(action)
		if terminated:
			print(t, state, delta, terminated)
			print("Goal reached!")
			break

def simulate_episode(env, aleph):
	class uniform_random_policy():
		def __init__(self):
			pass

		def __call__(self, state):
			return self

		def score(self, action):
			return 1

	state, info = env.reset()
	agent = AgentMDPPlanning({
		# admissibility parameters:
		"maxLambda": 1, # upper bound on local relative aspiration in each step (must be minLambda...1)	# TODO: rename to lambdaHi
		"minLambda": 0, # lower bound on local relative aspiration in each step (must be 0...maxLambda)	# TODO: rename to lambdaLo
		# policy parameters:
		"lossTemperature": 1e-10, # temperature of softmin mixture of actions w.r.t. loss, must be > 0
		"lossCoeff4Random": 0, # weight of random tie breaker in loss function, must be >= 0
		"lossCoeff4FeasibilityPower": 0, # weight of power of squared admissibility interval width in loss function, must be >= 0
		"lossCoeff4LRA1": 0, # weight of current-state deviation of LRA from 0.5 in loss function, must be >= 0
		"lossCoeff4Time1": 0, # weight of not terminating in loss function, must be >= 0
		"lossCoeff4Entropy1": 0, # weight of current-state action entropy in loss function, must be >= 0
		"lossCoeff4KLdiv1": 0, # weight of current-state KL divergence in loss function, must be >= 0
		"lossCoeff4DP": 0, # weight of disordering potential in loss function, must be >= 0
		"uninformedStatePriorScore": 0,
		"internalTransitionEntropy": 0,
		# coefficients for expensive to compute loss functions (all zero by default except for variance):
		"lossCoeff4Variance": 0, # weight of variance of total in loss function, must be >= 0
		"lossCoeff4Fourth": 0, # weight of centralized fourth moment of total in loss function, must be >= 0
		"lossCoeff4Cup": 0, # weight of "cup" loss component, based on sixth moment of total, must be >= 0
		"lossCoeff4LRA": 0, # weight of deviation of LRA from 0.5 in loss function, must be >= 0
		"lossCoeff4Time": 0, # weight of time in loss function, must be >= 0
		"lossCoeff4DeltaVariation": 0, # weight of variation of Delta in loss function, must be >= 0
		"lossCoeff4Entropy": 0, # weight of action entropy in loss function, must be >= 0
		"lossCoeff4KLdiv": 0, # weight of KL divergence in loss function, must be >= 0
		"lossCoeff4OtherLoss": 0, # weight of other loss components specified by otherLossIncrement, must be >= 0
		"uninformedPolicy": uniform_random_policy()
		}, world=env)
	total = delta = np.array([0, 0])
	for t in range(1000):
		action, aleph4action = agent.localPolicy(state, aleph).sample()[0]
		print("t:",t, ", last delta:",delta, ", total:", total, ", s:",state, ", aleph4s:", aleph, ", a:", action, ", aleph4a:", aleph4action)
		nextState, delta, terminated, _, info = env.step(action)
		total += delta
		aleph = agent.propagateAspiration(state, action, aleph4action, delta, nextState)
		state = nextState
		if terminated:
			print("t:",t, ", last delta:",delta, ", final total:", total, ", final s:",state, ", aleph4s:", aleph)
			print("Terminated.")
			break


env = DartGame()
aleph0 = (0, 0)
#move_randomly(env)
simulate_episode(env, aleph0)
#env.render()
#time.sleep(1)
env.close()

exit()