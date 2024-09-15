import sys
sys.path.insert(0, "./src/")

from satisfia.agents.learning.dqn.config import DQNConfig
from satisfia.agents.learning.dqn.variable_parameters import variable_params
from satisfia.agents.learning.dqn.train import train_dqn, DQNTrainingStatistics
import random
from test_dqn import *
import numpy as np
import pandas as pd
import importlib 


hyperparameters= {
"temperature": [i for i in np.linspace(0,10,11)],
"period": [i for i in range(1,40,5)],
"batch_size": [i for i in range(16,256,16)],
"total_timesteps": [i for i in range(100_000, 1_000_000, 100_000)],
"learning_rate_scheduler": [i for i in np.logspace(-5,1,11)]
}
all_gridworlds = [  "GW6"]
# , "GW22", "GW23", "GW24", "GW25", "GW26",
#                    "GW27", "GW28", "GW29", "GW30", "GW31", "GW32", "AISG2", "test_return",
#                    "test_box", "GW1", "GW2", "GW3", "GW4", "GW5" ]

results_df = pd.DataFrame(columns=["temperature", "period", "batch_size", "total_timesteps", 
                                   "learning_rate_scheduler", "gridworld", "mean_average_deviation"])

def Gibbs_sample_parameters(n):
    global results_df
    for parameter in hyperparameters: 
        random_value = hyperparameters[parameter][random.randint(0, len(hyperparameters[parameter]) - 1)]
        setattr(cfg, parameter, \
                random_value)
        print(random_value)
    for i in range(n):
        parameter = random.choice(list(hyperparameters.keys()))
        parameter_value = hyperparameters[parameter][random.randint(0, len(hyperparameters[parameter]) - 1)]
        setattr(cfg, parameter, \
                parameter_value)
        gridworld = random.choice(all_gridworlds)
    def make_env():
        env, _ = make_simple_gridworld(gridworld, time=10)
        env = RestrictToPossibleActionsWrapper(env)
        return env
    def make_model(pretrained=None):
        d_observation = len(make_env().observation_space)
        n_actions = make_env().action_space.n
        model = SatisfiaMLP(
            input_size = d_observation,
            output_not_depending_on_agent_parameters_sizes = { "maxAdmissibleQ": n_actions,
                                                                "minAdmissibleQ": n_actions },
            output_depending_on_agent_parameters_sizes = dict(), 
            common_hidden_layer_sizes = [64, 64],
            hidden_layer_not_depending_on_agent_parameters_sizes = [16],
            hidden_layer_depending_on_agent_parameters_sizes = [], 
            batch_size = cfg.num_envs,
            layer_norms = False,
            dropout = 0
        )
        if pretrained is not None:
            model.load_state_dict(pretrained)
        return model

    planning_agent = AgentMDPPlanning({ "lossCoeff4FeasibilityPowers": 0,
                                           "lossCoeff4LRA1": 0,
                                           "lossCoeff4Time1": 0,
                                           "lossCoeff4Entropy1": 0,
                                           "defaultPolicy": None },
                                        make_env())

    model = run_or_load( f"dqn-{gridworld}-no-discount.pickle",
                            train_dqn,
                            make_env,
                            make_model,
                            dataclasses.replace(
                                cfg,
                                planning_agent_for_plotting_ground_truth=planning_agent
                            ) )
    model = model.to(device)
    learning_agent = AgentMDPDQN( { "lossCoeff4FeasibilityPowers": 0,
                                            "lossCoeff4LRA1": 0,
                                            "lossCoeff4Time1": 0,
                                            "lossCoeff4Entropy1": 0,
                                            "defaultPolicy": None },
                                    model,
                                    num_actions = make_env().action_space.n,
                                    device = device )
    get_ground_truth = DQNTrainingStatistics()
    return get_ground_truth.ground_truth_criteria(model, gridworld)

Gibbs_sample_parameters(1)



        # mean_average_deviation = np.mean(np.abs(ground_truth - predictions))
        # selected_params["mean_average_deviation"] = mean_average_deviation

        # # Append the selected parameters and result to the DataFrame
        # results_df = results_df.append(selected_params, ignore_index=True)


