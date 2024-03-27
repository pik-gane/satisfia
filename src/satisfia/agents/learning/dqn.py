from dataclasses import dataclass, field
import gymnasium as gym
from typing import Optional, List, Dict, Callable, Optional, Tuple
import random
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from functools import partial, lru_cache
from more_itertools import chunked
from statistics import mean
from more_itertools import pairwise
from plotly.graph_objects import Figure, Scatter
from plotly.subplots import make_subplots
from os.path import isfile
import pickle

from ..makeMDPAgentSatisfia import AgentMDPLearning

class PiecewiseLinearScheduler:
    def __init__(self, join_points: List[Tuple[float, float]]):
        self.join_points = join_points
        assert all(x1 < x2 for (x1, y1), (x2, y2) in pairwise(self.join_points))

    def __call__(self, x):
        for (x1, y1), (x2, y2) in pairwise(self.join_points):
            if x1 <= x <= x2:
                return y1 + (x - x1) / (x2 - x1) * (y2 - y1)
        assert False

@dataclass
class DQNConfig:
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    tqdm: bool = True
    """print a progress bar while training"""
    return_plot: bool = True
    """return a plot object with the training statistics"""

    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    double_q_learning: bool = True
    """use double q learning to overcome q value overestimation"""
    lambda_high: float = 1.
    lambda_low: float = 0.
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """exploration rate scheduler"""
    eps_scheduler: Callable[[float], float] = field(default_factory=lambda: PiecewiseLinearScheduler([(0., 1.), (0.5, 0.05), (1., 0.05)]))
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    unbias: bool = True
    """learn the q overestimation bias and subtract it"""
    train_unbias_frequency: int = 10
    """the frequency of training the unbiasing model"""

"""
def make_mdp_agent_dqn( params: Dict,
                        model: nn.Module,
                        env: gym.Env,
                        config: DQNConfig,
                        save_to_file: str = None ) -> Tuple[AgentMDPLearning, Optional[Figure]]:
    
    load_from_file = save_to_file is not None and isfile(save_to_file)

    if load_from_file:
        with open(save_to_file, "rb") as f:
            saved = pickle.load(f)
        maximizer_predictor_model = saved["maximizer_model"]
        minimizer_predictor_model = saved["minimizer_model"]
        fig = None

    else:
        maximizer_control_model = model
        maximizer_predictor_model = deepcopy(model)
        minimizer_control_model = deepcopy(model)
        minimizer_predictor_model = deepcopy(model)
        maximizer_training_statistics = train_dqn(maximizer_control_model, maximizer_predictor_model, env,                    config)
        minimizer_training_statistics = train_dqn(minimizer_control_model, minimizer_predictor_model, FlipRewardWrapper(env), DQNConfig(total_timesteps=20_000)) # temporarily short training

        if save_to_file is not None:
            saved = {"maximizer_model": maximizer_predictor_model, "minimizer_model": minimizer_predictor_model}
            with open(save_to_file, "wb") as f:
                pickle.dump(saved, f)

        if config.return_plot:
            fig = dqn_training_statistics_plot( maximizer_statistics = maximizer_training_statistics,
                                                minimizer_statistics = minimizer_training_statistics )
        else:
            fig = None

    agent = AgentMDPLearning( params           = params,
                              maxAdmissibleQ   = q_network_to_function(maximizer_predictor_model),
                              minAdmissibleQ   = q_network_to_function(minimizer_predictor_model),
                              possible_actions = (lambda state: list(range(env.action_space.n))), )
    
    # temporary
    agent.minimizer_model = minimizer_predictor_model
    agent.maximizer_model = maximizer_predictor_model

    return agent, fig

class FlipRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, terminated, info = self.env.step(action)
        return observation, -reward, done, terminated, info
"""
        
@dataclass
class DQNTrainingStatistics:
    def __init__(self, max_and_min=False):
        if max_and_min:
            self.episodic_returns = {"min": dict(), "max": dict()}
            self.episodic_lengths = {"min": dict(), "max": dict()}
            self.td_losses        = {"min": dict(), "max": dict()}
            self.predictor_losses = {"min": dict(), "max": dict()}
        else:
            self.episodic_returns = dict()
            self.episodic_lengths = dict()
            self.td_losses        = dict()
            self.predictor_losses = dict()

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

"""
def train_dqn_simple(model: nn.Module, env: gym.Env, cfg: DQNConfig):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    env = gym.wrappers.RecordEpisodeStatistics(env)
    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = model.to(device)
    optimizer = optim.AdamW(q_network.parameters(), lr=cfg.learning_rate)
    target_network = deepcopy(model).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(cfg.buffer_size, device)
    
    episodic_return_history = dict()
    episodic_length_history = dict()
    td_loss_history = dict()

    obs, _ = env.reset()
    for global_step in tqdm(range(cfg.total_timesteps)) if cfg.tqdm else range(cfg.total_timesteps):
        epsilon = cfg.eps_scheduler(global_step / cfg.total_timesteps)
        if random.random() < epsilon:
            actions = env.action_space.sample()
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = q_values.argmax().item()

        next_obs, rewards, terminations, truncations, info = env.step(actions)

        rb.add(obs, next_obs, np.array(actions), np.array(rewards), np.array(terminations))

        obs = next_obs

        if terminations or truncations:
            obs, _ = env.reset()

            episodic_return_history[global_step] = info["episode"]["r"].item()
            episodic_length_history[global_step] = info["episode"]["l"].item()

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if global_step % cfg.train_frequency == 0:
                data = rb.sample(cfg.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations.float()).max(dim=1)
                    td_target = data.rewards.flatten() + cfg.gamma * target_max * (1 - data.terminations.flatten().float())
                old_val = q_network(data.observations.float()).gather(1, data.actions.unsqueeze(-1)).squeeze()
                loss = F.mse_loss(td_target, old_val)

                td_loss_history[global_step] = loss.item()

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % cfg.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    fig = Figure()
    fig.add_trace(Scatter( x      = list(mean(slice) for slice in chunked(episodic_length_history.keys(), 100)),
                           y      = list(mean(slice) for slice in chunked(episodic_length_history.values(), 100)),
                           name   = "episodic length",
                           yaxis  = "y2",
                           mode   = "markers",
                           marker = {"opacity": 0.5} ))
    fig.add_trace(Scatter( x      = list(mean(slice) for slice in chunked(td_loss_history.keys(), 100)),
                           y      = list(mean(slice) for slice in chunked(td_loss_history.values(), 100)),
                           name   = "TD loss",
                           yaxis  = "y4",
                           mode   = "markers",
                           marker = {"opacity": 0.5} ))
    fig.add_trace(Scatter( x      = list(mean(slice) for slice in chunked(episodic_return_history.keys(), 100)),
                           y      = list(x for x in (mean(slice) for slice in chunked(episodic_return_history.values(), 100))),
                           name   = "episodic return",
                           mode   = "markers",
                           marker = {"opacity": 0.5}))
    
    fig.update_layout(
        # title=f"{weight_decay=}",
        xaxis_title="Global step",
        yaxis=dict(title="episodic return"),
        yaxis2=dict(title="episodic length", overlaying="y", side="right", type="log"),
        yaxis4=dict(title="TD loss", anchor="free", overlaying="y", autoshift=True),
    )

    fig.show()

    env.close()

    return target_network
"""

class MinMaxLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.max_linear = nn.Linear(in_features, out_features)
        self.min_linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return { "max": self.max_linear(x),
                 "min": self.min_linear(x) }
    
"""
class MinMaxUnbiasLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.max_linear = nn.Linear(in_features, out_features)
        self.min_linear = nn.Linear(in_features, out_features)
        self.max_bias_linear = nn.Linear(in_features, 1)
        self.min_bias_linear = nn.Linear(in_features, 1)

    def forward(self, x):
        max_output = self.max_linear(x)
        min_output = self.min_linear(x)
        max_bias   = self.max_bias_linear(x)
        min_bias   = self.min_bias_linear(x)
        return { "max":          max_output,
                 "min":          min_output,
                 "max_unbiased": max_output - max_bias,
                 "min_unbiased": min_output - min_bias }
"""

class MinMaxLinear(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5):
        super().__init__()
        self.average_linear = nn.Linear(in_features, out_features)
        # self.log_half_length_linear = nn.Linear(in_features, out_features)
        self.half_length = nn.Linear(in_features, out_features)
        self.eps = eps

    def forward(self, x):
        average = self.average_linear(x)
        half_length = self.half_length(x).abs()
        # log_half_length = self.log_half_length_linear(x)
        # half_length = log_half_length.exp()
        return { "max": average + half_length + self.eps,
                 "min": average - half_length - self.eps }
    
class MinMaxUnbiasLinear(nn.Module):
    def __init__(self, in_features, out_fetaures):
        super().__init__()
        self.biased = MinMaxLinear(in_features, out_fetaures)
        self.unbiased = MinMaxLinear(in_features, out_fetaures)

    def forward(self, x):
        biased = self.biased(x)
        unbiased = self.unbiased(x)
        return { "max":          biased["max"],
                 "min":          biased["min"],
                 "max_unbiased": unbiased["max"],
                 "min_unbiased": unbiased["min"] }

def train_dqn(model: nn.Module, env: gym.Env, cfg: DQNConfig) -> DQNTrainingStatistics:
    stats = DQNTrainingStatistics(max_and_min=True)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    env = gym.wrappers.RecordEpisodeStatistics(env)
    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    env = { "max": env,
            "min": deepcopy(env) }

    target_network = model.to(device)
    q_network = deepcopy(model).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate)

    replay_buffer = { "max": ReplayBuffer(cfg.buffer_size, device),
                      "min": ReplayBuffer(cfg.buffer_size, device) }

    observation = {"max": None, "min": None}
    for max_or_min in ["max", "min"]:
        observation[max_or_min], _ = env[max_or_min].reset()

    torch_max_or_min       = {"max": torch.max,    "min": torch.min}
    torch_argmax_or_argmin = {"max": torch.argmax, "min": torch.argmin}

    rewards_this_episode = {"max": [], "min": []}
    for global_step in tqdm(range(cfg.total_timesteps)) if cfg.tqdm else range(cfg.total_timesteps):
        possible_actions = env[max_or_min].env.env.possible_actions(tuple(observation[max_or_min]))
        # assert possible_actions != []
        STAY_IN_PLACE_ACTION = 4
        if STAY_IN_PLACE_ACTION not in possible_actions:
            possible_actions.append(STAY_IN_PLACE_ACTION)

        epsilon = cfg.eps_scheduler(global_step / cfg.total_timesteps)
        for max_or_min in ["max", "min"]:
            if random.random() < epsilon:
                # action = env[max_or_min].action_space.sample()
                action = random.choice(possible_actions)
            else:
                q_values = q_network(torch.tensor(observation[max_or_min]).to(device))[max_or_min]
                mask = torch.full_like(q_values, float("-inf"))
                mask[possible_actions] = 0
                action = torch_argmax_or_argmin[max_or_min](q_values + mask).item()

            next_observation, reward, termination, truncation, info = env[max_or_min].step(action)
            rewards_this_episode[max_or_min].append(reward)

            replay_buffer[max_or_min].add(observation[max_or_min], next_observation, np.array(action), np.array(reward), np.array(termination))

            observation[max_or_min] = next_observation

            if termination or truncation:
                observation[max_or_min], _ = env[max_or_min].reset()

                episodic_return = info["episode"]["r"].item()
                episodic_length = info["episode"]["l"].item()

                replay_buffer[max_or_min].end_episode(rewards_this_episode[max_or_min], discount=cfg.gamma)
                rewards_this_episode[max_or_min] = []

                stats.episodic_returns[max_or_min][global_step] = episodic_return
                stats.episodic_lengths[max_or_min][global_step] = episodic_length

        train = global_step > cfg.learning_starts and global_step % cfg.train_frequency == 0
        if train:
            loss = {"max": None, "min": None}
            for max_or_min in ["max", "min"]:
                data = replay_buffer[max_or_min].sample(cfg.batch_size)
                with torch.no_grad():
                    if cfg.double_q_learning:
                        target = target_network(data.next_observations)[max_or_min]
                        q = q_network(data.next_observations)[max_or_min]
                        target_min = target.gather(-1, torch.argmin(q, -1).unsqueeze(-1)).squeeze(-1)
                        target_max = target.gather(-1, torch.argmax(q, -1).unsqueeze(-1)).squeeze(-1)
                    else:
                        target_min, _ = torch.min(target_network(data.next_observations)[max_or_min], dim=1)
                        target_max, _ = torch.max(target_network(data.next_observations)[max_or_min], dim=1)
                    lam = {"max": cfg.lambda_high, "min": cfg.lambda_low}[max_or_min]
                    target_max_min_combination = lam * target_max + (1 - lam) * target_min
                    td_target = data.rewards.flatten() \
                              + cfg.gamma * target_max_min_combination * (1 - data.terminations.float()).flatten()
                old_val = q_network(data.observations)[max_or_min].gather(1, data.actions.unsqueeze(-1)).squeeze(-1)
                loss[max_or_min] = F.mse_loss(td_target, old_val)

                stats.td_losses[max_or_min][global_step] = loss[max_or_min].item()

            optimizer.zero_grad()
            (loss["max"] + loss["min"]).backward()
            optimizer.step()

        update_target_network = global_step >= cfg.learning_starts and global_step % cfg.target_network_frequency == 0
        if update_target_network:
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(
                    cfg.tau * q_network_param.data + (1.0 - cfg.tau) * target_network_param.data
                )

        train_predictor = cfg.unbias and global_step >= cfg.learning_starts and global_step % cfg.train_unbias_frequency == 0
        if train_predictor:
            loss = {"max": None, "min": None}
            for max_or_min in ["max", "min"]:
                data = replay_buffer[max_or_min].sample_totals(cfg.batch_size)
                q = model(data.observations)[f"{max_or_min}_unbiased"]
                pred = q.gather(1, data.actions.unsqueeze(-1)).squeeze()
                loss[max_or_min] = F.mse_loss(pred, data.totals)
                stats.predictor_losses[max_or_min][global_step] = loss[max_or_min].item()

            optimizer.zero_grad()
            (loss["max"] + loss["min"]).backward()
            optimizer.step()

    return stats

def q_network_to_function(model: nn.Module) -> Callable[["state", "action"], "q_value"]:
    # @lru_cache
    def all_q_values(state: torch.Tensor) -> torch.Tensor:
        return model(torch.tensor(state).float())
    
    def q_values(state: torch.Tensor, action: int) -> float:
        return all_q_values(state)[action].item()
    
    return q_values

@dataclass
class ReplayBufferSample:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor

@dataclass
class ReplayBufferTotalsSample:
    observations: torch.Tensor
    actions: torch.Tensor
    totals: torch.Tensor

class ReplayBuffer:
    def __init__(self, size, device):
        self.size = size
        self.device = device

        self.observations = None
        self.next_observations = None
        self.actions = None
        self.rewards = None
        self.terminations = None

        self.totals = np.zeros(self.size, dtype=np.float32)

        self.end_last_episode = 0
        self.num_written = 0

    def add(self, observation, next_observation, action, reward, termination):
        if self.observations is None:
            self.observations      = np.empty((self.size, *observation.shape), dtype=observation.dtype)
            self.next_observations = np.empty((self.size, *next_observation.shape), dtype=next_observation.dtype)
            self.actions           = np.empty(self.size, dtype=int)
            self.rewards           = np.empty(self.size, dtype=np.float32)
            self.terminations      = np.empty(self.size, dtype=bool)

        self.observations     [self.num_written % self.size, ...] = observation
        self.next_observations[self.num_written % self.size, ...] = next_observation
        self.actions          [self.num_written % self.size, ...] = action
        self.rewards          [self.num_written % self.size, ...] = reward
        self.terminations     [self.num_written % self.size, ...] = termination

        self.num_written += 1

    def end_episode(self, rewards, discount):
        i = (self.num_written - 1) % self.size
        total = 0.
        for reward in rewards[::-1]:
            total = reward + discount * total
            self.totals[i] = total
            i = (i - 1) % self.size
        assert i == (self.end_last_episode - 1) % self.size
        self.end_last_episode = self.num_written

    def sample(self, sample_size):
        i = np.random.choice(min(self.size, self.num_written), sample_size, replace=False)
        return ReplayBufferSample( observations      = torch.tensor(self.observations     [i, ...]),
                                   next_observations = torch.tensor(self.next_observations[i, ...]),
                                   actions           = torch.tensor(self.actions          [i, ...]),
                                   rewards           = torch.tensor(self.rewards          [i, ...]),
                                   terminations      = torch.tensor(self.terminations     [i, ...]) )

    def sample_totals(self, sample_size):
        num_without_total = self.num_written - self.end_last_episode
        num_with_total = min(self.size, self.num_written) - num_without_total
        i = np.random.choice(num_with_total, sample_size, replace=False)
        i = self.end_last_episode - 1 - i
        i %= self.size
        return ReplayBufferTotalsSample( observations = torch.tensor(self.observations[i, ...]),
                                         actions      = torch.tensor(self.actions     [i, ...]),
                                         totals       = torch.tensor(self.totals      [i, ...]) )

def dqn_training_statistics_plot(maximizer_statistics: DQNTrainingStatistics, minimizer_statistics: DQNTrainingStatistics) -> Figure:
    minimizer_statistics.episodic_returns = {timestep: -episodic_return for timestep, episodic_return in minimizer_statistics.episodic_returns.items()}

    statistics = {"max": maximizer_statistics, "min": minimizer_statistics}

    fig = make_subplots( rows           = 2,
                         cols           = 1,
                         subplot_titles = ("Maximizer training statistics", "Minimizer training_statistics") )
    
    for i_row, max_or_min in [(1, "max"), (2, "min")]:
        for i_statistic, (name, statistic, mode) in enumerate([ 
            ("episodic return", statistics[max_or_min].episodic_returns, "markers"),
            ("episodic length", statistics[max_or_min].episodic_lengths, "markers"),
            ("td loss",         statistics[max_or_min].td_losses,        "markers"),
            ("predictor loss",  statistics[max_or_min].predictor_losses, "markers")
        ]):

            fig.add_trace( Scatter( x     = list(statistic.keys()),
                                    y     = list(statistic.values()),
                                    name  = name,
                                    yaxis = f"y{i_statistic+1 if i_statistic > 0 else ''}",
                                    mode  = mode ),
                           row=i_row,
                           col=1 )
            
        if i_statistic == 0:
            fig.update_layout(yaxis=dict(title=name))
        else:
            fig.update_layout(
                **{f"yaxis{i_statistic+1}": dict(title=name, anchor="free", overlaying="y", autoshift=True)}
            )

    return fig
