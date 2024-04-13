#!/usr/bin/env python3

import sys
from time import perf_counter



sys.path.insert(0,'./src/')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--default', default=0, type=float)
parser.add_argument('-w', '--world', default="GW26", type=str)
parser.add_argument('--human', default=False, action='store_true')
args = parser.parse_args()

default = args.default 

from environments.very_simple_gridworlds import make_simple_gridworld
from satisfia.agents.makeMDPAgentSatisfia import AgentMDPPlanning, AspirationAgent
from world_model.simple_gridworld import SimpleGridworld, MDPWorldModel
from world_model.world_model import WorldModel

def move_agent(env, aleph, default=0):
	class policy():
		def __init__(self):
			pass

		def __call__(self, state):
			return self

		def score(self, action):
			return 1

	state, _ = env.reset()
	agent = AgentMDPPlanning({
		# admissibility parameters:
		"maxLambda": 1, # upper bound on local relative aspiration in each step (must be minLambda...1)	# TODO: rename to lambdaHi
		"minLambda": 0, # lower bound on local relative aspiration in each step (must be 0...maxLambda)	# TODO: rename to lambdaLo
		# policy parameters:
		"lossTemperature": 1e-10, # temperature of softmin mixture of actions w.r.t. loss, must be > 0
		"uninformedStatePriorScore": 0,
		"internalTransitionEntropy": 0,
		"lossCoeff4Random": default, # weight of random tie breaker in loss function, must be >= 0
		"lossCoeff4FeasibilityPower": default, # weight of power of squared admissibility interval width in loss function, must be >= 0
		"lossCoeff4LRA1": default, # weight of current-state deviation of LRA from 0.5 in loss function, must be >= 0
		"lossCoeff4Time1": default, # weight of not terminating in loss function, must be >= 0
		"lossCoeff4Entropy1": default, # weight of current-state action entropy in loss function, must be >= 0
		"lossCoeff4KLdiv1": default, # weight of current-state KL divergence in loss function, must be >= 0
		"lossCoeff4DP": default, # weight of disordering potential in loss function, must be >= 0
		"lossCoeff4Variance": default, # weight of variance of total in loss function, must be >= 0
		"lossCoeff4Fourth": default, # weight of centralized fourth moment of total in loss function, must be >= 0
		"lossCoeff4Cup": default, # weight of "cup" loss component, based on sixth moment of total, must be >= 0
		"lossCoeff4LRA": default, # weight of deviation of LRA from 0.5 in loss function, must be >= 0
		"lossCoeff4Time": default, # weight of time in loss function, must be >= 0
		"lossCoeff4DeltaVariation": default, # weight of variation of Delta in loss function, must be >= 0
		"lossCoeff4Entropy": default, # weight of action entropy in loss function, must be >= 0
		"lossCoeff4KLdiv": default, # weight of KL divergence in loss function, must be >= 0
		"lossCoeff4OtherLoss": default, # weight of other loss components specified by otherLossIncrement, must be >= 0
        "lossCoeff4WassersteinTerminalState": 100, # weight of Wasserstein distance to default terminal state distribution in loss function, must be >= 0
		"uninformedPolicy": policy()
		}, world=env)
	total = delta = 0
	for t in range(1000):
		action, aleph4action = agent.localPolicy(state, aleph).sample()[0]
		nextState, delta, terminated, _, _ = env.step(action)
		total += delta
		aleph = agent.propagateAspiration(state, action, aleph4action, delta, nextState)
		state = nextState
		if terminated:
			break


class timeit:
    def __init__(self, msg) -> None:
        self.msg = msg

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        time = perf_counter() - self.start
        print(f'{self.msg} took {time:.3f} seconds')

with timeit(f"{args.world} with default = {default}"):
    # pr.disable()
    env, aleph0 = make_simple_gridworld(gw = args.world, render_mode = 'human' if args.human else None, fps = 1)
    env.reset()
    move_agent(env, aleph0, default)
    env.render()
    env.close()
    # pr.dump_stats("perf_eval.prof")

import functools
for c in (AspirationAgent, AgentMDPPlanning, SimpleGridworld, MDPWorldModel, WorldModel):
    for m in c.__dict__.values():
        if isinstance(m, functools._lru_cache_wrapper):
            ci = m.cache_info()
            if ci.misses:
                print(f"{c.__name__}.{m.__name__}, hits={ci.hits}, "
                      f"misses={ci.misses} ({(ci.hits/ci.misses*100):.0f}%)")

