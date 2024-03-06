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

from world_model import SimpleGridworld
from environments.very_simple_gridworlds import make_simple_gridworld
import pylab as plt
from satisfia.agents.makeMDPAgentSatisfia import AgentMDPPlanning

def move_randomly(env):
	state, delta, terminated, _, info = env.reset()
	for t in range(1000):
		actions = env.possible_actions(state)
		action = random.choice(actions)
		print(t, state, delta, terminated, _, info, actions, action)
		state, delta, terminated, _, info = env.step(action)
		if terminated:
			print(t, state, delta, terminated)
			print("Goal reached!")
			break

def move_agent(env, aleph):
	class policy():
		def __init__(self):
			pass

		def __call__(self, state):
			return self

		def score(self, action):
			return 1

	state, delta, terminated, _, info = env.reset()
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
		"lossCoeff4Time": 100, # weight of time in loss function, must be >= 0
		"lossCoeff4DeltaVariation": 0, # weight of variation of Delta in loss function, must be >= 0
		"lossCoeff4Entropy": 0, # weight of action entropy in loss function, must be >= 0
		"lossCoeff4KLdiv": 0, # weight of KL divergence in loss function, must be >= 0
		"lossCoeff4OtherLoss": 0, # weight of other loss components specified by otherLossIncrement, must be >= 0
		"uninformedPolicy": policy()
		}, world=env)
	total = delta
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

for gw in ["GW1", "GW2", "GW3", "GW4", "GW5", "GW6"]:
	print("\nRUNNING AROUND",gw,":")
	env, aleph0 = make_simple_gridworld(gw = gw, render_mode = "human", fps = 1)
	env.reset()
	#move_randomly(env)
	move_agent(env, aleph0)
	env.render()
	#time.sleep(5)
	env.close()


print("\nPUSHING A BOX THROUGH A GOAL:")
env, aleph0 = make_simple_gridworld(gw = "test_box", render_mode = "human", fps = 1)
env.reset()
#move_randomly(env)
move_agent(env, aleph0)
env.render()
time.sleep(1)
env.close()


print("\nRUNNING AROUND AISG2:")
env, aleph0 = make_simple_gridworld(gw = "AISG2", render_mode = "human", fps = 1)
env.reset()
#move_randomly(env)
move_agent(env, aleph0)
env.render()
time.sleep(1)
env.close()

exit()

print("\nRUNNING AROUND A RANDOM GRID:")
grid = [
	[   
		random.choice([' ', ' ', ' ', '#', '#', ',', '^', '~','X'])
		for x in range(11)
	]
	for y in range(11)
]
grid[2][2] = "A"
grid[8][8] = "G"
delta_grid = [
	[
		' ' if grid[y][x] == '#' else random.choice([' ','M','P'], p=[0.4,0.3,0.3])
		for x in range(11)
	]
	for y in range(11)
]
print(grid)
print(delta_grid)
env = SimpleGridworld(grid = grid, delta_grid = delta_grid, cell_code2delta = {'M':-1, 'P':1}, render_mode = "human", fps = 1)
move_randomly(env)
env.close()
