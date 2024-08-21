from __future__ import annotations

from satisfia.agents.learning.dqn.config import DQNConfig
from satisfia.agents.learning.dqn.bellman_formula import bellman_formula
from satisfia.agents.learning.dqn.criteria import complete_criteria
from satisfia.agents.learning.dqn.replay_buffer import ReplayBuffer
from satisfia.agents.learning.dqn.exploration_strategy import ExplorationStrategy
from satisfia.agents.learning.environment_wrappers import RestrictToPossibleActionsWrapper
from satisfia.agents.learning.dqn.agent_mdp_dqn import AgentMDPDQN
from satisfia.util.interval_tensor import IntervalTensor

from gymnasium import Env
from gymnasium.wrappers import AutoResetWrapper
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import torch
from torch import tensor
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from collections import Counter
from statistics import mean
from more_itertools import chunked
from tqdm import tqdm
from typing import Callable, Tuple, List, Dict
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.graph_objects import Figure

def train_dqn( make_env:   Callable[[], Env],
               make_model: Callable[[], Module],
               cfg: DQNConfig ) -> Module:
    
    stats = DQNTrainingStatistics(cfg)

    q_network = make_model()
    target_network = make_model() 
    target_network.load_state_dict(q_network.state_dict())

    # we set weight decay to zero because we had some mild Q value underestimation problems and were
    # suspecting they were because of the weight decay, but we are not sure at all this is correct
    # and not sure aet all setting weigth decay to zero is helpful
    optimizer = AdamW( q_network.parameters(),
                       lr           = cfg.learning_rate_scheduler(0),
                       weight_decay = 0 )

    make_envs = [ (lambda: AutoResetWrapper(make_env()))
                  for _ in range(cfg.num_envs) ]
    envs = AsyncVectorEnv(make_envs) if cfg.async_envs else SyncVectorEnv(make_envs)

    if cfg.env_type == "mujoco":
        num_actions = envs.action_space.shape[0]
    else:
        num_actions = envs.action_space.nvec[0]
    exploration_strategy = ExplorationStrategy(
        target_network
            if cfg.frozen_model_for_exploration is None
                else cfg.frozen_model_for_exploration,
        cfg,
        num_actions=num_actions
    )
    replay_buffer = ReplayBuffer(cfg.buffer_size, device=cfg.device)

    seen_observations = set()

    observations, _ = envs.reset()
    for timestep in tqdm(range(cfg.total_timesteps), desc="training dqn"):
        for observation in observations:
            seen_observations.add(tuple(observation.tolist()))

        # NOTE: Want to explore using eplison greedy or other non-model related strategies
        actions = exploration_strategy(tensor(observations, device=cfg.device), timestep=timestep)

        next_observations, deltas, dones, truncations, _ = envs.step(actions.cpu().numpy())

        aspirations = exploration_strategy.aspirations
        exploration_strategy.propagate_aspirations( actions,
                                                    tensor(next_observations, device=cfg.device) )

        exploration_strategy.on_done(tensor(dones), timestep=timestep)

        replay_buffer.add( observations      = tensor(observations,        device=cfg.device),
                           actions           = actions,
                           deltas            = tensor(deltas,              device=cfg.device),
                           dones             = tensor(dones | truncations, device=cfg.device),
                           next_observations = tensor(next_observations,   device=cfg.device),
                           aspirations       = aspirations,
                           next_aspirations  = exploration_strategy.aspirations )

        observations = next_observations

        register_criteria_in_stats = cfg.plotted_criteria is not None \
                                        and timestep % cfg.plot_criteria_frequency == 0
        if register_criteria_in_stats:
            stats.register_criteria(target_network, timestep)

        train = timestep >= cfg.training_starts and timestep % cfg.training_frequency == 0
        if train:
            set_learning_rate( optimizer,
                               cfg.learning_rate_scheduler(timestep / cfg.total_timesteps) )

            replay_buffer_sample = replay_buffer.sample(cfg.batch_size).to(cfg.device)

            predicted_criteria = q_network( replay_buffer_sample.observations,
                                            replay_buffer_sample.aspirations,
                                            noisy=False )
            complete_criteria(predicted_criteria)

            td_target = bellman_formula( replay_buffer_sample,
                                         q_network=q_network,
                                         target_network=target_network,
                                         predicted_criteria=predicted_criteria,
                                         cfg=cfg )

            loss = 0
            for criterion, coefficient in cfg.criterion_coefficients_for_loss.items():
                if coefficient == 0:
                    continue

                loss_fn = cfg.criterion_loss_fns[criterion]
                prediction_for_actions = predicted_criteria[criterion].gather(
                    -1,
                    replay_buffer_sample.actions.unsqueeze(-1)
                ).squeeze(-1)
                loss += coefficient * loss_fn(
                    prediction_for_actions,
                    td_target[criterion]
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        update_target_network = timestep >= cfg.training_starts \
                                    and timestep % cfg.target_network_update_frequency == 0
        if update_target_network:
            for target_network_param, q_network_param in zip( target_network.parameters(),
                                                              q_network.parameters() ):
                target_network_param.data.copy_(
                           cfg.soft_target_network_update_coefficient  * target_network_param.data
                    + (1 - cfg.soft_target_network_update_coefficient) * q_network_param.data
                )

    # print(seen_observations)

    if cfg.plotted_criteria is not None:
        stats.plot_criteria(q_network, RestrictToPossibleActionsWrapper(make_env()))

    # if cfg.soft_target_network_update_coefficient != 0 returning the q_network is not the same as
    # returning the target network
    return target_network

def set_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

def compute_total(agent, env, state, state_aspiration, first_action=None):
    if isinstance(state_aspiration, (int, float)):
        state_aspiration = (state_aspiration, state_aspiration)

    total = 0.
    env.reset() # reset just in case
    env.set_state(state)
    observation = state
    done = False
    first_iteration = True
    while not done:
        if first_iteration and first_action is not None:
            first_iteration = False
            action = first_action
            action_aspiration = agent.aspiration4action(state, action, state_aspiration)
        else:
            action, action_aspiration = agent.localPolicy(observation, state_aspiration).sample()[0]
        next_observation, delta, done, truncated, _ = env.step(action)
        done = done or truncated
        total += delta
        state_aspiration = agent.propagateAspiration(observation, action, action_aspiration, Edel=None, nextState=next_observation)
        observation = next_observation
    return total

def smoothen(xs, smoothness):
    return [mean(chunk) for chunk in chunked(xs, smoothness)]

@dataclass
class DQNTrainingStatistics:
    cfg: DQNConfig
    criterion_history: Dict[Tuple["timestep", "state", "state_aspiration", "criterion", "action"], float] = \
        field(default_factory=lambda: dict())

    def register_criteria(self, model, timestep):
        for state in self.cfg.states_for_plotting_criteria:
            for state_aspiration in self.cfg.state_aspirations_for_plotting_criteria:
                state_as_tensor = tensor([state], dtype=torch.float, device=self.cfg.device)
                state_aspiration_low, state_aspiration_high = state_aspiration
                state_aspiration_as_tensor = IntervalTensor(
                    tensor([state_aspiration_low],  dtype=torch.float, device=self.cfg.device), 
                    tensor([state_aspiration_high], dtype=torch.float, device=self.cfg.device)
                )
                criteria = model(state_as_tensor, state_aspiration_as_tensor, noisy=False)
                complete_criteria(criteria)
                for criterion in self.cfg.plotted_criteria:
                    for action in self.cfg.actions_for_plotting_criteria:
                        self.criterion_history[timestep, state, state_aspiration, criterion, action] = \
                            criteria[criterion].squeeze(0)[action].item()

    def ground_truth_criteria(self, model, env) -> Dict[Tuple["state", "state_aspiration", "criterion", "action"], float] | None:
        if self.cfg.planning_agent_for_plotting_ground_truth is None:
            return None

        criteria = dict()

        for state in self.cfg.states_for_plotting_criteria:
            for state_aspiration in self.cfg.state_aspirations_for_plotting_criteria:
                for criterion in self.cfg.plotted_criteria:
                    for action in self.cfg.actions_for_plotting_criteria:
                        criterion_function =\
                            getattr(self.cfg.planning_agent_for_plotting_ground_truth, criterion)
                        
                        if criterion in ["maxAdmissibleQ", "minAdmissibleQ"]:
                            possible_action = action in self.cfg.planning_agent_for_plotting_ground_truth.possible_actions(state)
                            criterion_value = \
                                criterion_function(state, action) if possible_action else None
                        elif criterion in ["Q"]:
                            agent = AgentMDPDQN( self.cfg.satisfia_agent_params,
                                                 model
                                                    if self.cfg.frozen_model_for_exploration is None
                                                        else self.cfg.frozen_model_for_exploration,
                                                 env.action_space.n )
                            criterion_value = mean(Parallel(n_jobs=-1)(
                                delayed(compute_total)( agent,
                                                        env,
                                                        state,
                                                        state_aspiration,
                                                        first_action=action )
                                for _ in tqdm(range(1000)) # TO DO: ERROR BARS!!!
                            ))
                        else:
                            raise ValueError(f"Unknown criterion '{criterion}'.")
                        
                        criteria[state, state_aspiration, criterion, action] = criterion_value

        return criteria

    def plot_criteria(self, model, env):
        ground_truth_criteria = self.ground_truth_criteria(model, env)

        fig = Figure()
        fig.update_layout(title = "Predictde criteria during DQN training.")

        timesteps = sorted(list(set(x[0] for x in self.criterion_history.keys())))
        dropdown_menu_titles = []
        first_iteration = True
        for criterion in self.cfg.plotted_criteria:
            for state in self.cfg.states_for_plotting_criteria:
                for state_aspiration in self.cfg.state_aspirations_for_plotting_criteria:
                    dropdown_menu_titles.append(f"{criterion} in state {state} with state aspiration {state_aspiration}")
                    for action in self.cfg.actions_for_plotting_criteria:
                        y = [ self.criterion_history[timestep, state, state_aspiration, criterion, action]
                              for timestep in timesteps ]
                        fig.add_scatter(
                            x = smoothen(timesteps, self.cfg.plot_criteria_smoothness),
                            y = smoothen(y,         self.cfg.plot_criteria_smoothness),
                            line = dict(color=DEFAULT_PLOTLY_COLORS[action]),
                            name = f"action {action}",
                            visible = first_iteration
                        )
                        if ground_truth_criteria is not None:
                            fig.add_scatter(
                                x = [timesteps[0], timesteps[-1]],
                                y = [ground_truth_criteria[state, state_aspiration, criterion, action]] * 2,
                                line = dict(dash="dot", color=DEFAULT_PLOTLY_COLORS[action]),
                                name = f"action {action} ground truth",
                                visible = first_iteration
                            )
                    first_iteration = False
        
        num_plotted_actions = len(self.cfg.actions_for_plotting_criteria)
        num_scatters_per_dropdown_menu_option = \
            num_plotted_actions if ground_truth_criteria is None else 2 * num_plotted_actions
        fig.update_layout(updatemenus=[dict(
            direction="down",
            showactive=True,
            buttons=[
                dict( label=menu_title,
                      method="update",
                      args=[dict(
                          visible =   [False] * i
                                              * num_scatters_per_dropdown_menu_option
                                    + [True]  * num_scatters_per_dropdown_menu_option
                                    + [False] * (len(dropdown_menu_titles) - i - 1)
                                              * num_scatters_per_dropdown_menu_option
                      )] )
                for i, menu_title in enumerate(dropdown_menu_titles)
            ]
        )])

        fig.show()
