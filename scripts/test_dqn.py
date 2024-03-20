import sys
sys.path.insert(0, "./src/")

from satisfia.agents.makeMDPAgentSatisfia import AspirationAgent
from satisfia.agents.learning.dqn import train_dqn, DQNConfig, MinMaxLinear, MinMaxUnbiasLinear, PiecewiseLinearScheduler
import simplified_checkers_against_random_player
import multi_armed_bandit
import gymnasium as gym
import torch
from torch import tensor
from torch.nn import Linear, Sequential, ReLU
from os.path import isfile
import pickle
from plotly.graph_objects import Figure, Scatter, Layout
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm
from statistics import mean, stdev
from copy import deepcopy
from itertools import count
from more_itertools import chunked
import random
import numpy as np
from math import prod
from joblib import Parallel, delayed

# env = gym.make("SimplifiedCheckersAgainstRandomPlayer-v0", board_height=4, board_width=4, num_rows_with_pieces_initially=1)
# env = TimeLimit(env, 500)
# env = gym.make("CartPole-v1")
env = gym.make("MultiArmedBandit-v0")
# save_to = "temp-checkers-model.pickle"
save_to = "multi-armed-bandit-model-lambda-1.pickle"
if not isfile(save_to):
    model = Sequential(Linear(prod(env.observation_space.shape), 128), ReLU(), Linear(128, 128), ReLU(), MinMaxUnbiasLinear(128, env.action_space.n))
    stats = train_dqn( model,
                       env,
                       DQNConfig( lambda_high = 1.,
                                  lambda_low = 0.,
                                  double_q_learning=True,
                                  unbias=True,
                                  total_timesteps=500_000,
                                  eps_scheduler=PiecewiseLinearScheduler([(0., 1.), (0.2, 0.05), (0.7, 0.), (1., 0.)])) )

    opacity = 0.5
    def smoothen(xs):
        return [mean(chunk) for chunk in chunked(xs, 100)]
    for max_or_min in ["max", "min"]:
        fig = Figure(layout=Layout( title  = f"{max_or_min}imizer training statistics",
                                    yaxis  = dict(title="episodic length"),
                                    yaxis2 = dict(title="epsiodic length", anchor="free", autoshift=True, overlaying="y"),
                                    yaxis3 = dict(title="td loss",         anchor="free", autoshift=True, overlaying="y"),
                                    yaxis4 = dict(title="predictor loss",  anchor="free", autoshift=True, overlaying="y") ))
        fig.add_trace(Scatter(x=smoothen(stats.episodic_lengths[max_or_min].keys()), y=smoothen(stats.episodic_lengths[max_or_min].values()), name="episodic length", mode="markers", marker=dict(opacity=opacity)))
        fig.add_trace(Scatter(x=smoothen(stats.episodic_returns[max_or_min].keys()), y=smoothen(stats.episodic_returns[max_or_min].values()), name="episodic return", mode="markers", marker=dict(opacity=opacity), yaxis="y2"))
        fig.add_trace(Scatter(x=smoothen(stats.td_losses       [max_or_min].keys()), y=smoothen(stats.td_losses       [max_or_min].values()), name="td loss",         mode="markers", marker=dict(opacity=opacity), yaxis="y3"))
        fig.add_trace(Scatter(x=smoothen(stats.predictor_losses[max_or_min].keys()), y=smoothen(stats.predictor_losses[max_or_min].values()), name="predictor loss",  mode="markers", marker=dict(opacity=opacity), yaxis="y4"))
        fig.show()

    with open(save_to, "wb") as f:
        pickle.dump(model, f)

with open(save_to, "rb") as f:
    model = pickle.load(f)

class AgentMDPDQN(AspirationAgent):
    def __init__(self, params, model, action_space, unbias=True):
        super().__init__(params)
        self.model = model
        self.action_space = action_space
        self.unbias = unbias

    def maxAdmissibleQ(self, state: torch.tensor, action: int):
        key = "max_unbiased" if self.unbias else "max"
        q = self.model(torch.tensor(state).float())
        assert (q["min"] < q["max"]).all() and (q["min_unbiased"] < q["max_unbiased"]).all()
        return q[key][action].item()
    
    def minAdmissibleQ(self, state: torch.tensor, action: int):
        key = "min_unbiased" if self.unbias else "min"
        q = self.model(torch.tensor(state).float())
        assert (q["min"] < q["max"]).all() and (q["min_unbiased"] < q["max_unbiased"]).all()
        return q[key][action].item()
    
    def disorderingPotential_action(self, *args, **kwargs):
        assert False
    
    def LRAdev_action(self, *args, **kwargs):
        assert False
    
    def behaviorEntropy_action(self, *args, **kwargs):
        assert False
    
    def behaviorKLdiv_action(self, *args, **kwargs):
        assert False
    
    def otherLoss_action(self, *args, **kwargs):
        assert False
    
    def Q(self, *args, **kwargs):
        assert False
    
    def Q2(self, *args, **kwargs):
        assert False
    
    def Q3(self, *args, **kwargs):
        assert False
    
    def Q4(self, *args, **kwargs):
        assert False
    
    def Q5(self, *args, **kwargs):
        assert False
    
    def Q6(self, *args, **kwargs):
        assert False
    
    def Q_ones(self, *args, **kwargs):
        assert False
    
    def Q_DeltaSquare(self, *args, **kwargs):
        assert False
    
    def possible_actions(self, *args, **kwargs):
        return list(range(self.action_space.n))
    
    def agencyChange_action(self, *args, **kwargs):
        assert False

def test_agent(agent, env, aleph):
    observation, _ = env.reset()
    total = 0.
    while True:
        action = agent.localPolicy(observation, aleph=aleph).sample()
        action = action[0][0]
        observation, reward, done, truncated, _ = env.step(np.array(action))
        total += reward
        if done or truncated:
            return total

params = { "maxLambda": 1.,
           "minLambda": 0.,
           "lossCoeff4DP": 0,
           "lossCoeff4LRA": 0,
           "lossCoeff4LRA1": 0,
           "lossCoeff4Time": 0,
           "lossCoeff4Time1": 0,
           "lossCoeff4Entropy": 0,
           "lossCoeff4Entropy1": 0,
           "lossCoeff4KLdiv": 0,
           "lossCoeff4KLdiv1": 0,
           "lossCoeff4Variance": 0,
           "lossCoeff4Fourth": 0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
           "lossCoeff4Cup": 0,
           "lossCoeff4LRA": 0,
           "lossCoeff4OtherLoss": 0,
           "lossCoeff4AgencyChange": 0,
           "debug": False }

for unbias in [False]: # [True, False]
    agent = AgentMDPDQN(params, model, env.action_space, unbias=unbias)

    # alephs = [-4., -2., -1., 0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.]
    # alephs = [-8., -4., -2., -1., -0.5, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.2, 1.4, 1.6, 1.8, 1.9, 2.25, 2.5, 3, 4., 8.]
    # alephs = [-8., -4., -2., -1., -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1., 2., 4., 8.]
    alephs = [-2., -1., -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1., 2.]
    # alephs = [1.]
    total_means = []
    total_stdevs = []
    for aleph in alephs:
        totals = [test_agent(agent, env, aleph=aleph) for _ in tqdm(range(5_000), desc=f"{aleph=}")]
        print(f"mean aspiration agent total for aspiration {aleph}: mean:", mean(totals), "stdev:", stdev(totals))
        total_means.append(mean(totals))
        total_stdevs.append(stdev(totals))
    print(f"{totals=}")
    fig = Figure(layout=Layout(xaxis=dict(title="aspiration"), yaxis=dict(title="total"), title=f"{unbias=}"))
    fig.add_trace(Scatter(x=alephs, y=total_means, name="got with standard deviations", error_y=dict(type="data", array=total_stdevs, visible=True)))
    fig.add_trace(Scatter(x=alephs, y=alephs, name="expected"))
    fig.show()