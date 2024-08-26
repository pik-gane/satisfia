from satisfia.agents.learning.dqn.config import DQNConfig
import satisfia.agents.learning.dqn.agent_mdp_dqn as agent_mpd_dqn
from satisfia.agents.learning.dqn.criteria import complete_criteria
from satisfia.util.interval_tensor import IntervalTensor, relative_position, interpolate

from torch import Tensor, empty, ones, full_like, randint, bernoulli, no_grad, allclose
from torch.nn import Module
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import random

class ExplorationStrategy:
    def __init__(self, target_network: Module, cfg: DQNConfig, num_actions: int):
        self.target_network = target_network
        self.cfg = cfg
        self.num_actions = num_actions

        self.aspirations = IntervalTensor( empty(self.cfg.num_envs, device=cfg.device),
                                           empty(self.cfg.num_envs, device=cfg.device) )
        self.on_done(dones=ones(self.cfg.num_envs, dtype=bool, device=cfg.device), timestep=0)

    @no_grad()
    def __call__(self, observations: Tensor, timestep: int):
        actions = self.Boltzmann_periodic_policy_actions(observations, timestep=timestep, period=1).sample()
        
        exploration_rate = self.cfg.exploration_rate_scheduler(timestep / self.cfg.total_timesteps)
        explore = bernoulli(full_like(actions, exploration_rate, dtype=float)).bool()
        actions[explore] = randint( low=0,
                                    high=self.num_actions,
                                    size=(explore.int().sum().item(),),
                                    device=self.cfg.device )

        return actions

    @no_grad()
    def Boltzmann_probabilistic_policy_actions(self, observations, probability):
        criteria = self.target_network(observations)
        complete_criteria(criteria)
        self.criteria = criteria
        max_q_values = self.target_network(observations)['maxAdmissibleQ']
        min_q_values = self.target_network(observations)['minAdmissibleQ']
        temperature = self.cfg.temperature
        if random.random() >probability:
            boltzmann_probabilities = F.softmax(max_q_values / temperature, dim=-1)
        else:
            boltzmann_probabilities = F.softmax(min_q_values / temperature, dim=-1)
        policy_distribution = Categorical(probs=boltzmann_probabilities)
        return policy_distribution
    
    @no_grad()
    def Boltzmann_periodic_policy_actions(self, observations, timestep, period):
        criteria = self.target_network(observations, self.aspirations)
        complete_criteria(criteria)
        self.criteria = criteria
        max_q_values = self.target_network(observations, self.aspirations)['maxAdmissibleQ']
        min_q_values = self.target_network(observations, self.aspirations)['minAdmissibleQ']
        temperature = self.cfg.temperature
        if  timestep%(2*period)<=period:
            boltzmann_probabilities = F.softmax(max_q_values / temperature, dim=-1)
        else:
            boltzmann_probabilities = F.softmax(min_q_values / temperature, dim=-1)
        policy_distribution = Categorical(probs=boltzmann_probabilities)
        return policy_distribution

    @no_grad()
    def satisfia_policy_actions(self, observations: Tensor) -> Categorical:
        criteria = self.target_network(observations, self.aspirations)
        # criteria["maxAdmissibleQ"], criteria["minAdmissibleQ"] = criteria["maxAdmissibleQ"].maximum(criteria["minAdmissibleQ"]), criteria["maxAdmissibleQ"].minimum(criteria["minAdmissibleQ"])
        complete_criteria(criteria)
        self.criteria = criteria
        return agent_mpd_dqn.local_policy( self.cfg.satisfia_agent_params,
                                           criteria,
                                           self.aspirations )

    # TO DO: move all this stuff into agent_mdp_dqn.py
    def propagate_aspirations(self, actions: Tensor, next_observations: Tensor):
        state_aspirations  = agent_mpd_dqn.state_aspirations (self.criteria, self.aspirations)
        action_aspirations = agent_mpd_dqn.action_aspirations(self.criteria, state_aspirations)
        action_aspirations = action_aspirations.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        min_admissible_q = self.criteria["minAdmissibleQ"]
        max_admissible_q = self.criteria["maxAdmissibleQ"]
        min_admissible_q = min_admissible_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        max_admissible_q = max_admissible_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        lambda_low  = relative_position(min_admissible_q, action_aspirations.lower, max_admissible_q)
        lambda_high = relative_position(min_admissible_q, action_aspirations.upper, max_admissible_q)
        
        # this will be recalculated in the next call to satisfia_policy_actions, which is necessary
        # there the aspirations would be different
        next_criteria = self.target_network(next_observations, self.aspirations)
        complete_criteria(next_criteria)

        next_min_admissible_v = next_criteria["minAdmissibleV"]
        next_max_admissible_v = next_criteria["minAdmissibleV"]
        
        return IntervalTensor(
            interpolate(next_min_admissible_v, lambda_low,  next_max_admissible_v),
            interpolate(next_min_admissible_v, lambda_high, next_max_admissible_v)
        )

    @no_grad()
    def on_done(self, dones: Tensor, timestep: int):
        self.new_aspirations(which=dones)

        if self.cfg.noisy_network_exploration:
            self.new_network_noise(timestep=timestep, which_in_batch=dones)

    @no_grad()
    def new_aspirations(self, which: Tensor):
        num_new_aspirations = sum(which.int()).item()
        self.aspirations[which] = \
            self.cfg.aspiration_sampler(num_new_aspirations).to(self.cfg.device)

    @no_grad()
    def new_network_noise(self, timestep: int, which_in_batch: Tensor | None = None):
        std = self.cfg.noisy_network_exploration_rate_scheduler(timestep / self.cfg.total_timesteps)
        self.target_network.new_noise(std, which_in_batch)