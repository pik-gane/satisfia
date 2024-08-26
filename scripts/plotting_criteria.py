import sys
sys.path.insert(0, "./src/")
import tf_slim as slim
from satisfia.agents.learning.dqn.train import *

import time
from numpy import random

from world_model import SimpleGridworld
from environments.very_simple_gridworlds import make_simple_gridworld
import pylab as plt
from satisfia.agents.makeMDPAgentSatisfia import AgentMDPPlanning, AspirationAgent

#print("\nPUSHING A BOX THROUGH A GOAL:")
#env, aleph0 = make_simple_gridworld(gw = "test_box", render_mode = "human", fps = 1)
#model = AgentMDPPlanning(AspirationAgent)
#plot_criteria(model, env)


def run_or_load(filename, function, *args, **kwargs):
    if isfile(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    result = function(*args, **kwargs)
    with open(filename, "wb") as f:
        pickle.dump(result, f)
    return result


def train_and_plot( env_name: str,
                    gridworld: bool = True,
                    max_achievable_total: float | None = None,
                    min_achievable_total: float | None = None ):

    print(env_name)

    def make_env():
        if gridworld:
            env, _ = make_simple_gridworld(env_name, time=10)
            env = RestrictToPossibleActionsWrapper(env)
        else:
            env = gym.make(env_name)
            env = TimeLimit(env, 1_000)
            env = RescaleDeltaWrapper(env, from_interval=(-500, 100), to_interval=(-5, 1))
        return env

    def make_model(pretrained=None):
        # to do: compute d_observation properly
        d_observation = len(make_env().observation_space) if gridworld else 8
        n_actions = make_env().action_space.n
        model = SatisfiaMLP(
            input_size = d_observation,
            output_not_depending_on_agent_parameters_sizes = { "maxAdmissibleQ": n_actions,
                                                               "minAdmissibleQ": n_actions },
            #output_depending_on_agent_parameters_sizes = dict(), # { "Q": n_actions },
            common_hidden_layer_sizes = [64, 64],
            hidden_layer_not_depending_on_agent_parameters_sizes = [16],
            #hidden_layer_depending_on_agent_parameters_sizes = [], # [64],
            batch_size = cfg.num_envs,
            layer_norms = False,
            dropout = 0
        )
        if pretrained is not None:
            model.load_state_dict(pretrained)
        return model

    planning_agent = AgentMDPPlanning(cfg.satisfia_agent_params, make_env()) if gridworld else None
    model = run_or_load( f"dqn-{env_name}-no-discount.pickle",
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
