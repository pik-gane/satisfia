import sys
sys.path.insert(0, "./src/")

from satisfia.agents.makeMDPAgentSatisfia import AspirationAgent
from satisfia.agents.learning.dqn import train_dqn, DQNConfig, MinMaxLinear, MinMaxUnbiasLinear, PiecewiseLinearScheduler
import simplified_checkers_against_random_player
import multi_armed_bandit
import gymnasium as gym
import torch
from torch import tensor
from torch.nn import Linear, Sequential, ReLU, ModuleList, Embedding, Module
from os.path import isfile
import pickle
from plotly.graph_objects import Figure, Scatter, Layout
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm
from statistics import mean, stdev
from copy import deepcopy
from itertools import count, pairwise
from more_itertools import chunked
import random
import scipy
import numpy as np
from math import prod
from joblib import Parallel, delayed
from environments.very_simple_gridworlds import make_simple_gridworld

def confidence_interval(xs, confidence):
    return scipy.stats.t.interval( confidence=confidence,
                                   df=len(xs)-1,
                                   loc=mean(xs),
                                   scale=scipy.stats.sem(xs) )

def error_bars(xs, confidence):
    lower, upper = confidence_interval(xs, confidence=confidence)
    mean = mean(xs)
    return mean - lower, upper - mean

class AgentMDPDQN(AspirationAgent):
    def __init__(self, params, model, action_space, unbias=False):
        super().__init__(params)
        self.model = model
        self.action_space = action_space
        self.unbias = unbias

    def maxAdmissibleQ(self, state: torch.tensor, action: int):
        key = "max_unbiased" if self.unbias else "max"
        q = self.model(torch.tensor(state))
        assert (q["min"] < q["max"]).all()
        if "min_unbiased" in q:
            assert (q["min_unbiased"] < q["max_unbiased"]).all()
        return q[key][action].item()
    
    def minAdmissibleQ(self, state: torch.tensor, action: int):
        key = "min_unbiased" if self.unbias else "min"
        q = self.model(torch.tensor(state))
        assert (q["min"] < q["max"]).all()
        if "min_unbiased" in q:
            (q["min_unbiased"] < q["max_unbiased"]).all()
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

    def stateDistance_action(self, *args, **kwargs):
        assert False
    
    def trajectoryEntropy_action(self, *args, **kwargs):
        assert False

    def ETerminalState_action(self, *args, **kwargs):
        assert False

    def ETerminalState2_action(self, *args, **kwargs):
        assert False

    def causationPotential_action(self, *args, **kwargs):
        assert False

    def causation_action(self, *args, **kwargs):
        assert False

def test_agent(agent, env, aleph):
    observation, _ = env.reset()
    total = 0.
    while True:
        action, aleph4action = agent.localPolicy(observation, aleph=aleph).sample()[0]
        next_observation, reward, done, truncated, _ = env.step(action)
        total += reward
        agent.propagateAspiration(observation, action, aleph4action, reward, next_observation)
        observation = next_observation
        if done or truncated:
            return total

def test_optimizer(model, env, max_or_min):
    torch_argmax_or_argmin = {"max": torch.argmax, "min": torch.argmin}
    observation, _ = env.reset()
    total = 0.
    while True:
        action = torch_argmax_or_argmin[max_or_min](model(torch.tensor(observation).float())[max_or_min]).item()
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
           "defaultPolicy": None,
           "debug": False }

class MultiEmbedding(Module):
    def __init__(self, vocab_sizes, dim, vocab_starts=None):
        super().__init__()
        self.vocab_starts = vocab_starts if vocab_starts is not None else np.zeros_like(vocab_sizes)
        self.embeddings = ModuleList(Embedding(n, dim) for n in vocab_sizes)

    def forward(self, x):
        if isinstance(x, tuple):
            x = np.array(list(x))
        return sum(embedding(x[..., i] - self.vocab_starts[i]) for i, embedding in enumerate(self.embeddings))
        
class ToNumpyArrayWrapper(gym.Env):
    def __init__(self, env: gym.Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if isinstance(observation, tuple):
            observation = np.array(list(observation))
        if isinstance(observation, list):
            observation = np.array(observation)
        return observation, info
    
    def step(self, action, options=None):
        observation, reward, done, truncated, info = self.env.step(action)
        if isinstance(observation, tuple):
            observation = np.array(list(observation))
        if isinstance(observation, list):
            observation = np.array(observation)
        return observation, reward, done, truncated, info

for gridword_name in ["GW1", "GW2", "GW3", "GW4", "GW5", "GW6", "GW22", "GW23", "GW24", "test_box", "AISG2"]:
    env, aleph0 = make_simple_gridworld(gridword_name)
    print(f"{aleph0=}")
    print(env.action_space)
    print(env.observation_space.nvec)
    print(env.action_space.n)
    print(f"{env.observation_space.__dict__=}")

    print(env.reset())

    save_filename = f"{gridword_name}-model.pickle"
    if not isfile(save_filename):
        model = Sequential( MultiEmbedding( vocab_sizes=env.observation_space.nvec,
                                            vocab_starts=env.observation_space.start,
                                            dim=16 ),
                            ReLU(),
                            MinMaxLinear(16, env.action_space.n) )

        train_dqn( model,
                ToNumpyArrayWrapper(env),
                DQNConfig( lambda_high = 1.,
                            lambda_low = 0.,
                            double_q_learning = True,
                            unbias = False,
                            total_timesteps = 10,
                            eps_scheduler=PiecewiseLinearScheduler([(0., 1.), (0.2, 0.05), (0.7, 0.), (1., 0.)]) ) )

        with open(save_filename, "wb") as f:
            pickle.dump(model, f)

    with open(save_filename, "rb") as f:
        model = pickle.load(f)

    agent = AgentMDPDQN(params, model, env.action_space)

    alephs = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]
    sample_size = 100
    for aleph in alephs:
        print("TOTAL", mean(test_agent(agent, env, aleph0) for _ in tqdm(range(sample_size))))

exit()

alephs = list(np.linspace(-1.5, 1.5, 17))
error_bar_confidence = 0.95
sample_size = 5
totals = dict()
total_confidence_intervals = dict()
optimizer_totals = {"max": None, "min": None}
optimizer_total_confidence_intervals = {"max": None, "min": None}
for lambda_low, lambda_high in [(0., 1.), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5)]:
    env_name = "SimplifiedCheckersAgainstRandomPlayer-v0"
    env = gym.make(env_name, board_height=4, board_width=4, num_rows_with_pieces_initially=1)

    model_filename = f"simplified-checkers-steps-lambda-{lambda_low}-{lambda_high}.pickle"
    if not isfile(model_filename):
        model = Sequential(Linear(prod(env.observation_space.shape), 128), ReLU(), Linear(128, 128), ReLU(), MinMaxUnbiasLinear(128, env.action_space.n))
        stats = train_dqn( model,
                           env,
                           DQNConfig( lambda_high = lambda_high,
                                      lambda_low = lambda_low,
                                      double_q_learning=True,
                                      unbias=False,
                                      total_timesteps=500_000,
                                      eps_scheduler=PiecewiseLinearScheduler([(0., 1.), (0.2, 0.05), (0.7, 0.), (1., 0.)])) )
            
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

    with open(model_filename, "rb") as f:
        model = pickle.load(f)

    agent = AgentMDPDQN(params, model, env.action_space)
    totals[(lambda_low, lambda_high)] = []
    total_confidence_intervals[(lambda_low, lambda_high)] = []
    for aleph in alephs:
        test_agent(agent, env, aleph)
        totals_for_aleph = Parallel(n_jobs=-1)( delayed(test_agent)(agent, env, aleph)
                                                for _ in tqdm(range(sample_size), desc=f" k{aleph=}") )
        totals[(lambda_low, lambda_high)].append(mean(totals_for_aleph))
        total_confidence_intervals[(lambda_low, lambda_high)].append(
            scipy.stats.t.interval( confidence=error_bar_confidence,
                                    df=len(totals_for_aleph)-1,
                                    loc=mean(totals_for_aleph),
                                    scale=scipy.stats.sem(totals_for_aleph) )
        )

    if (lambda_low, lambda_high) == (0., 1.):
        for max_or_min in ["max", "min"]:
            totals_sample = Parallel(n_jobs=-1)( delayed(test_optimizer)(agent.model, env, max_or_min)
                                                 for _ in tqdm(range(sample_size), desc=f"{max_or_min}imizer") )
            optimizer_totals[max_or_min] = mean(totals_sample)
            optimizer_total_confidence_intervals[max_or_min] = \
                scipy.stats.t.interval( confidence=error_bar_confidence,
                                        df=len(totals_sample)-1,
                                        loc=mean(totals_sample),
                                        scale=scipy.stats.sem(totals_sample) )

fig = Figure(layout = Layout( title = f"Totals achieved by AgentMDPDQN on simplified checkers agains a random player.",
                              xaxis=dict(title="aspiration"), yaxis=dict(title="total") ))
fig.add_trace(Scatter(x=alephs, y=alephs, name="expected"))
for (lambda_low, lambda_high) in totals.keys():
    totals_for_lambda = totals[(lambda_low, lambda_high)]
    total_confidence_intervals_for_lambda = total_confidence_intervals[(lambda_low, lambda_high)]
    fig.add_trace(Scatter( x=alephs,
                           y=totals_for_lambda,
                           name=f"total for lambda=({lambda_low}, {lambda_high}) ({error_bar_confidence:.0%} confidenec error bars)",
                           error_y=dict( type="data",
                                         symmetric=False,
                                         array=[upper - total for (lower, upper), total in zip(total_confidence_intervals_for_lambda, totals_for_lambda)],
                                         arrayminus=[total - lower for (lower, upper), total in zip(total_confidence_intervals_for_lambda, totals_for_lambda)],
                                         visible=True ) ))
for max_or_min in ["max", "min"]:
    fig.add_hline(y=optimizer_totals[max_or_min], line_color="grey", annotation_text=f"{max_or_min}imizer total")
    for error_bar in optimizer_total_confidence_intervals[max_or_min]:
        fig.add_hline(y=error_bar, line_color="grey", line_dash="dash")
fig.show()
