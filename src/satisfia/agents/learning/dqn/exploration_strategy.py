from satisfia.agents.learning.dqn.config import DQNConfig
import satisfia.agents.learning.dqn.agent_mdp_dqn as agent_mpd_dqn
from satisfia.agents.learning.dqn.criteria import complete_criteria
from satisfia.util.interval_tensor import IntervalTensor

from torch import Tensor, empty, ones, full_like, randint, bernoulli, no_grad
from torch.nn import Module
from torch.distributions.categorical import Categorical

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
        actions = self.satisfia_policy_actions(observations).sample()

        exploration_rate = self.cfg.exploration_rate_scheduler(timestep / self.cfg.total_timesteps)
        explore = bernoulli(full_like(actions, exploration_rate, dtype=float)).bool()
        actions[explore] = randint( low=0,
                                    high=self.num_actions,
                                    size=(explore.int().sum().item(),),
                                    device=self.cfg.device )

        return actions

    @no_grad()
    def satisfia_policy_actions(self, observations: Tensor) -> Categorical:
        criteria = self.target_network(observations, self.aspirations)
        complete_criteria(criteria)
        return agent_mpd_dqn.local_policy( self.cfg.satisfia_agent_params,
                                           criteria,
                                           self.aspirations )

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
