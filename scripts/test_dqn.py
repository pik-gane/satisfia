import sys
sys.path.insert(0, "./src/")

from environments.very_simple_gridworlds import make_simple_gridworld, all_worlds
from satisfia.agents.learning.dqn.train import train_dqn
from satisfia.agents.learning.dqn.config import DQNConfig, UniformPointwiseAspirationSampler, \
                                                UniformAspirationSampler, PiecewiseLinearScheduler
from satisfia.agents.learning.dqn.agent_mdp_dqn import AgentMDPDQN, local_policy
from satisfia.agents.learning.dqn.criteria import complete_criteria
from satisfia.agents.learning.models.building_blocks import SatisfiaMLP
from satisfia.agents.learning.environment_wrappers import RestrictToPossibleActionsWrapper
from satisfia.agents.makeMDPAgentSatisfia import AspirationAgent, AgentMDPPlanning
from satisfia.util.interval_tensor import IntervalTensor

import gymnasium as gym
import torch
from torch import tensor
from torch.nn import Module
import numpy as np
import scipy
import pickle
from joblib import Parallel, delayed
from statistics import mean
from functools import partial
from tqdm import tqdm
from typing import Tuple, List, Dict, Iterable, Callable, Generator
from os.path import isfile
import dataclasses
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.graph_objects import Figure, Scatter, Layout

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("using", device)

def multi_tqdm(num_tqdms: int) -> List[Callable[[Iterable], Iterable]]:
    def itr_wrapper(itr: Iterable, progress_bar: tqdm, desc: str | None = None, total: int | None = None) -> Generator:
        progress_bar.desc = desc
        progress_bar.reset()
        if total is not None:
            progress_bar.total = total
        else:
            progress_bar.total = len(itr) if hasattr(itr, "__len__") else None
        progress_bar.refresh()
        for item in itr:
            yield item
            progress_bar.update()
            progress_bar.refresh()

    progress_bars = [tqdm() for _ in range(num_tqdms)]
    return [partial(itr_wrapper, progress_bar=progress_bar) for progress_bar in progress_bars]

def confidence_interval(xs: List[float], confidence: float):
    return scipy.stats.t.interval(confidence, len(xs)-1, loc=np.mean(xs), scale=scipy.stats.sem(xs))

def error_bars(xs: List[float], confidence: float):
    mean_ = mean(xs)
    lower_confidence, upper_confidence = confidence_interval(xs, confidence)
    return mean_ - lower_confidence, upper_confidence - mean_

def run_or_load(filename, function, *args, **kwargs):
    if isfile(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    result = function(*args, **kwargs)
    with open(filename, "wb") as f:
        pickle.dump(result, f)
    return result

def compute_total(agent: AspirationAgent, env: gym.Env, aspiration4state: float | Tuple[float, float]) -> float:
    if isinstance(aspiration4state, (int, float)):
        aspiration4state = (aspiration4state, aspiration4state)

    total = 0.
    observation, _ = env.reset()
    done = False
    while not done:
        action, aspiration4action = agent.localPolicy(observation, aspiration4state).sample()[0]
        next_observation, delta, done, truncated, _ = env.step(action)
        done = done or truncated
        total += delta
        aspiration4state = agent.propagateAspiration(observation, action, aspiration4action, Edel=None, nextState=next_observation)
        observation = next_observation
    return total

def scatter_with_y_error_bars( x: Iterable[float],
                               y: Iterable[Iterable[float]],
                               confidence: float,
                               **plotly_kwargs ) -> Scatter:

    means = [mean(point_ys) for point_ys in y]
    error_bars_ = [error_bars(point_ys, confidence) for point_ys in y]
    return Scatter( x = x,
                    y = means,
                    error_y = dict( type = "data",
                                    symmetric = False,
                                    array = [lower for lower, upper in error_bars_],
                                    arrayminus = [upper for lower, upper in error_bars_] ),
                    **plotly_kwargs )

def plot_totals_vs_aspiration( agents: Iterable[AspirationAgent] | Dict[str, AspirationAgent] | AspirationAgent,
                               env: gym.Env,
                               aspirations: Iterable[int | Tuple[int, int]] | int | Tuple[int, int],
                               sample_size: int,
                               reference_agents: Iterable[AspirationAgent] | Dict[str, AspirationAgent] | AspirationAgent = [],
                               error_bar_confidence: float = 0.95,
                               n_jobs: int = -1,
                               title: str = "Totals for agent(s)",
                               save_to: str | None = None ):

    if not isinstance(agents, Iterable):
        agents = [agents]
    if not isinstance(agents, Dict):
        agents = {f"agent {i}": agent for i, agent in enumerate(agents)}
    if not isinstance(reference_agents, Iterable):
        reference_agents = [reference_agents]
    if not isinstance(reference_agents, Dict):
        reference_agents = {f"agent {i}": agent for i, agent in enumerate(reference_agents)}
    if not isinstance(aspirations, Iterable):
        aspirations = [aspirations]
    
    agent_tqdm, aspiration_tqdm, sample_tqdm = multi_tqdm(3)

    totals = dict()
    reference_totals = dict()
    for is_reference in [False, True]:
        for agent_name, agent in (agent_tqdm(agents.items(), desc="agents")
                                    if not is_reference else agent_tqdm(reference_agents.items(), desc="reference agents")):
            
            for aspiration in aspiration_tqdm(aspirations, desc=agent_name):
                if n_jobs == 1:
                    t = [
                        compute_total(agent, env, aspiration)
                        for _ in sample_tqdm(range(sample_size), desc=f"{agent_name}, {aspiration=}")
                    ]
                else:
                    t = Parallel(n_jobs=n_jobs)(
                        delayed(compute_total)(agent, env, aspiration)
                        for _ in sample_tqdm(range(sample_size), desc=f"{agent_name}, {aspiration=}")
                    )

                if is_reference:
                    reference_totals[agent_name, aspiration] = t
                else:
                    totals[agent_name, aspiration] = t

    fig = Figure(layout=Layout( title       = title + f". {error_bar_confidence:.0%} confidence error bars",
                                xaxis_title = "Aspiration",
                                yaxis_title = "Total" ))

    aspirations_as_points = [ (aspiration if isinstance(aspiration, (float, int)) else mean(aspiration))
                              for aspiration in aspirations ]

    point_aspirations = all(isinstance(aspiration, (int, float)) for aspiration in aspirations)
    for i_lower_or_upper, lower_or_upper in enumerate([None] if point_aspirations else ["lower", "upper"]):
        fig.add_trace(Scatter( x = aspirations_as_points,
                               y = [ aspiration if isinstance(aspiration, (float, int)) else aspiration[i_lower_or_upper]
                                     for aspiration in aspirations ],
                               name = "aspiration" if point_aspirations else f"{lower_or_upper} aspiration" ))

    for is_reference in [False, True]:
        for i_agent, agent_name in enumerate(reference_agents.keys() if is_reference else agents.keys()):
            t = reference_totals if is_reference else totals
            fig.add_trace(scatter_with_y_error_bars( x = aspirations_as_points,
                                                     y = [t[agent_name, aspiration] for aspiration in aspirations],
                                                     confidence = error_bar_confidence,
                                                     line = dict(color = DEFAULT_PLOTLY_COLORS[i_agent], dash = "dash" if is_reference else "solid"),
                                                     name = ("reference " if is_reference else "")
                                                        + (agent_name if not (len(agents) == 1 and agent_name == "agent 0") else "") ))

    fig.show()

    if save_to is not None:
        fig.write_html(save_to)

cfg = DQNConfig( aspiration_sampler = UniformPointwiseAspirationSampler(-20, 5),
                 criterion_coefficients_for_loss = dict( maxAdmissibleQ = 1.,
                                                         minAdmissibleQ = 1.,
                                                         Q = 1. ),
                 exploration_rate_scheduler =
                    PiecewiseLinearScheduler([0., 0.1, 1.], [1., 0.05, 0.05]),
                 noisy_network_exploration = False,
                 # noisy_network_exploration_rate_scheduler =
                 #    PiecewiseLinearScheduler([0., 0.1, 1.], [1., 0.05, 0.05]),
                 num_envs = 10,
                 async_envs = False,
                 discount = 1,
                 total_timesteps = 10_000,
                 training_starts = 100,
                 training_frequency = 1,
                 target_network_update_frequency = 50,
                 satisfia_agent_params = { "lossCoeff4FeasibilityPowers": 0,
                                           "lossCoeff4LRA1": 0,
                                           "lossCoeff4Time1": 0,
                                           "lossCoeff4Entropy1": 0,
                                           "defaultPolicy": None },
                 device = device,
                 plotted_criteria = ["maxAdmissibleQ", "minAdmissibleQ", "Q"],
                 plot_criteria_frequency = 100,
                 states_for_plotting_criteria = [(time, 2, 2) for time in range(10)],
                 state_aspirations_for_plotting_criteria = [(-5, -5), (-1, -1), (1, 1)],
                 actions_for_plotting_criteria = [2, 4] )

def train_and_plot(gridworld_name: str):
    print(gridworld_name)

    def make_env():
        env, _ = make_simple_gridworld(gridworld_name)
        return env

    def make_model(pretrained=None):
        d_observation = len(make_env().observation_space)
        n_actions = make_env().action_space.n
        model = SatisfiaMLP(
            input_size = d_observation,
            output_not_depending_on_agent_parameters_sizes = { "maxAdmissibleQ": n_actions,
                                                               "minAdmissibleQ": n_actions },
            output_depending_on_agent_parameters_sizes = { "Q": n_actions },
            common_hidden_layer_sizes = [64, 64],
            hidden_layer_not_depending_on_agent_parameters_sizes = [64],
            hidden_layer_depending_on_agent_parameters_sizes = [64],
            batch_size = cfg.num_envs,
            layer_norms = True,
            dropout = 0.1
        )
        if pretrained is not None:
            model.load_state_dict(pretrained)
        return model
        # return NoisyMLPReturningDict( [ d_observation + d_aspiration,
        #                                 64,
        #                                 64,
        #                                 { "maxAdmissibleQ": n_actions,
        #                                   "minAdmissibleQ": n_actions,
        #                                   "Q": n_actions } ],
        #                               batch_size=cfg.num_envs ).to(device)

    planning_agent = AgentMDPPlanning( cfg.satisfia_agent_params,
                                       RestrictToPossibleActionsWrapper(make_env()) )

    model = run_or_load( f"dqn-{gridworld_name}-no-discount-with-criteria.pickle",
                         train_dqn,
                         make_env,
                         make_model,
                         dataclasses.replace(
                             cfg,
                             planning_agent_for_plotting_ground_truth=planning_agent
                         ) )
    model = model.to(device)

    learning_agent = AgentMDPDQN( cfg.satisfia_agent_params,
                                  model,
                                  num_actions = make_env().action_space.n,
                                  device = device )

    for state in [(time, 2, 2) for time in range(10)]:
        for action in range(5):
            for state_aspiration in [(0, 0), (1, 1), (2, 2)]:
                action_aspiration = planning_agent.aspiration4action(state, action, state_aspiration)
                print( state,
                       action,
                       action_aspiration,
                         planning_agent.Q(state, action, state_aspiration)
                       - learning_agent.Q(state, action, state_aspiration) )

    first_observation, _ = make_env().reset()
    min_achievable_total = planning_agent.minAdmissibleV(first_observation)
    max_achievable_total = planning_agent.maxAdmissibleV(first_observation)
    plot_totals_vs_aspiration( agents = learning_agent,
                               env = RestrictToPossibleActionsWrapper(make_env()),
                               aspirations = np.linspace( min_achievable_total - 1,
                                                          max_achievable_total + 1,
                                                          20 ),
                               sample_size = 1_000,
                               # reference_agents = planning_agent,
                               title = f"totals for agent with no discount and longer training in {gridworld_name}" )

all_gridworlds = ["GW1", "GW2", "GW3", "GW4", "GW5", "GW6", "GW22", "GW23", "GW24", "GW25", "GW26", "GW27", "GW28", "GW29", "GW30", "GW31", "GW32", "AISG2", "test_return", "test_box"]
require_longer_training = ["GW22", "GW28", "AISG2", "test_box"]

# Parallel(n_jobs=-1)(delayed(train_and_plot)(gridworld_name) for gridworld_name in require_longer_training)
for gridworld_name in ["GW1"]:
    train_and_plot(gridworld_name)
