from satisfia.agents.makeMDPAgentSatisfia import AspirationAgent
import gymnasium as gym
import torch
from torch import tensor, Tensor, empty, cat, randn_like, randperm, where, no_grad
from satisfia.agents.makeMDPAgentSatisfia import AgentMDPPlanning
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.optim import AdamW
import numpy as np
import random
from functools import cache
from dataclasses import dataclass, field
from tqdm import tqdm
from more_itertools import pairwise, chunked
from statistics import mean
from typing import List, Callable, Iterable, Dict, Tuple, Any
from plotly.graph_objects import Figure, Layout
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS

class PiecewiseLinearScheduler:
    def __init__(self, x: List[float], y: List[float]):
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)
        assert len(self.x) > 0
        assert all(x1 < x2 for x1, x2 in pairwise(self.x))

    def __call__(self, x: float):
        if x <= self.x[0]:
            return self.y[0]
        if x >= self.x[-1]:
            return self.y[-1]
        for (x1, x2), (y1, y2) in zip(pairwise(self.x), pairwise(self.y)):
            if x1 <= x <= x2:
                return y1 + (x - x1) / (x2 - x1) * (y2 - y1)
        assert False, "unreachable"

@dataclass
class DQNConfig:
    train_minimizer: bool = True
    train_maximizer: bool = True
    total_timesteps: int = 500_000
    lambdas: Dict[str, float] = field(default_factory=lambda: {"max": 1., "min": 0.})
    lambda_sampler: Callable[[], Callable[[], float]] | None = None
    aspiration_sampler: Callable[[], Tuple[float, float]] = None
    discount: float = 0.99
    exploration_rate_scheduler: Callable[[float], float] = \
        PiecewiseLinearScheduler([0., 0.5, 1.], [1., 0.05, 0.05])
    parameter_noise_exploration: bool = False
    mix_maximizer_and_minimizer_strategies: bool = False
    learning_rate: float = 2.5e-4
    batch_size: int = 128
    train_corresponding_network: bool = False
    double_q_learning: bool = False
    fraction_samples_from_opposite_replay_buffer: float = 0
    true_double_q_learning: bool = False
    replay_buffer_size: int = 10_000
    learning_starts: int = 10_000
    target_network_update_frequency: int = 500
    train_frequency: int = 10
    plot: bool = True
    plot_smoothness: int = 1_000
    plot_title: str = "Training DQN."
    plot_q_values: bool = False
    observations_for_plotting_q_values: List[Any] | None = None
    actions_for_plotting_q_values: List[int] | None = None
    lambdas_for_plotting_q_values: List[float] | None = None
    plot_q_values_frequency: int = 100

    def __post_init__(self):
        assert self.train_maximizer or self.train_minimizer

class AgentMDPTabularLearning(AspirationAgent):
    def __init__(self, params: dict, planning_agent: AgentMDPPlanning, maximizer_q_table: Dict[Tuple, List[float]], minimizer_q_table: Dict[Tuple, List[float]]):
        super().__init__(params | {"defaultPolicy": planning_agent.world.default_policy})
        self.planning_agent = planning_agent
        self.maximizer_q_table = maximizer_q_table
        self.minimizer_q_table = minimizer_q_table

    def maxAdmissibleQ(self, state, action):
        if state not in self.maximizer_q_table:
            self.maximizer_q_table[state] = [np.random.normal() for _ in range(self.planning_agent.world.action_space.n)]
        return self.maximizer_q_table[state][action]

    def minAdmissibleQ(self, state, action):
        if state not in self.minimizer_q_table:
            self.minimizer_q_table[state] = [np.random.normal() for _ in range(self.planning_agent.world.action_space.n)]
        return self.minimizer_q_table[state][action]

    def possible_actions(self, state):
        return self.planning_agent.possible_actions(state)

def train_tabular_q_network_on_satisfia_policy(make_env: gym.Env, cfg: DQNConfig):
    env = make_env()

    maximizer_q_table = dict()
    minimizer_q_table = dict()
    stats = TabularQLeraningStatistics(cfg)

    agent_params = {}

    planning_agent = AgentMDPPlanning(agent_params, make_env())
    tabular_learning_agent = AgentMDPTabularLearning(agent_params, planning_agent, maximizer_q_table=maximizer_q_table, minimizer_q_table=minimizer_q_table)
    aspiration = cfg.aspiration_sampler()
    observation, _ = env.reset()
    for timestep in tqdm(range(cfg.total_timesteps), desc="tabular q learning"):
        exploration_rate = cfg.exploration_rate_scheduler(timestep / cfg.total_timesteps)
        explore = random.random() <= exploration_rate
        if explore:
            action = random.choice(tabular_learning_agent.possible_actions(observation))
            action_aspiration = tabular_learning_agent.aspiration4action(observation, action, aspiration)
        else:
            action, action_aspiration = tabular_learning_agent.localPolicy(observation, aspiration).sample()[0]

        next_observation, delta, done, terminated, _ = env.step(action)
        done = done or terminated

        if observation not in maximizer_q_table:
            maximizer_q_table[observation] = [np.random.normal() for _ in range(env.action_space.n)]
        if observation not in minimizer_q_table:
            minimizer_q_table[observation] = [np.random.normal() for _ in range(env.action_space.n)]
        if next_observation not in maximizer_q_table:
            maximizer_q_table[next_observation] = [np.random.normal() for _ in range(env.action_space.n)]
        if next_observation not in minimizer_q_table:
            minimizer_q_table[next_observation] = [np.random.normal() for _ in range(env.action_space.n)]

        possible_next_actions = planning_agent.possible_actions(next_observation)
        # if possible_next_actions != []:
        #     assert 4 in possible_next_actions
        # if 4 not in possible_next_actions:
        #     possible_next_actions.append(4)
        next_maximizer_q = [q for action, q in enumerate(maximizer_q_table[next_observation]) if action in possible_next_actions]
        maximizer_q_table[observation][action] = (1 - cfg.learning_rate) * maximizer_q_table[observation][action] \
            + cfg.learning_rate * (delta + (cfg.discount * max(next_maximizer_q) if not done else 0.))
        next_minimizer_q = [q for action, q in enumerate(minimizer_q_table[next_observation]) if action in possible_next_actions]
        minimizer_q_table[observation][action] = (1 - cfg.learning_rate) * minimizer_q_table[observation][action] \
            + cfg.learning_rate * (delta + (cfg.discount * min(next_minimizer_q) if not done else 0.))

        if timestep % cfg.plot_q_values_frequency == 0:
            stats.register_q_values("max", tuple(observation.int().tolist()), timestep, tuple(maximizer_q_table[observation].tolist()))
            stats.register_q_values("min", tuple(observation.int().tolist()), timestep, tuple(minimizer_q_table[observation].tolist()))

        aspiration = tabular_learning_agent.propagateAspiration(observation, action, action_aspiration, Edel=None, nextState=next_observation)
        observation = next_observation

        if done:
            planning_agent = AgentMDPPlanning(agent_params, make_env())
            tabular_learning_agent = AgentMDPTabularLearning(agent_params, planning_agent, maximizer_q_table=maximizer_q_table, minimizer_q_table=minimizer_q_table)
            aspiration = cfg.aspiration_sampler()
            observation, _ = env.reset()

    if cfg.plot:
        stats.plot(make_env)

    def q_table_to_function(q):
        def f(observation):
            observation = tuple(observation.flatten().tolist())
            if observation not in q:
                q[observation] = {action: np.random.normal() for action in range(env.action_space.n)}
            return tensor([q[observation][action] for action in range(env.action_space.n)])
        return f

    return { "maximizer": q_table_to_function(maximizer_q_table),
             "minimizer": q_table_to_function(minimizer_q_table) }


def train_tabular_q_network(make_env: gym.Env, cfg: DQNConfig):
    stats = DQNTrainingStatistics(cfg, keys=[(0, "max"), (0, "min")])

    q_table = {"max": dict(), "min": dict()}

    envs = {"max": make_env(), "min": make_env()}

    observations = dict()
    for key in ["max", "min"]:
        observations[key], _ = envs[key].reset()
    for timestep in tqdm(range(cfg.total_timesteps), desc="tabular q learning"):
        exploration_rate = cfg.exploration_rate_scheduler(timestep / cfg.total_timesteps)
        for key in ["max", "min"]:
            max_or_min = {"max": max, "min": min}[key]

            if hasattr(envs[key], "possible_actions"):
                possible_actions = envs[key].possible_actions()
            else:
                possible_actions = list(range(envs[key].action_space.n))


            if observations[key] not in q_table[key]:
                q_table[key][observations[key]] = {action: np.random.normal() for action in range(envs[key].action_space.n)}

            explore = random.random() < exploration_rate
            if explore:
                action = random.choice(possible_actions)
            else:
                possible_actions_with_qs = [action for action in possible_actions if action in q_table[key][observations[key]]]
                possible_qs = [q_table[key][observations[key]][action] for action in possible_actions_with_qs]
                action = possible_actions_with_qs[possible_qs.index(max_or_min(possible_qs))]
                # print(f"{possible_actions_with_qs=} {possible_qs=} {action=} {possible_qs.index(max_or_min(possible_qs))=}")

            # print(key, observations[key], action, explore)

            next_observation, reward, done, truncated, _ = envs[key].step(action)
            done = done or truncated

            train_key = key if cfg.fraction_samples_from_opposite_replay_buffer(timestep / cfg.total_timesteps) <= random.random() else {"max": "min", "min": "max"}[key]

            if observations[key] not in q_table[train_key]:
                q_table[train_key][observations[key]] = {action: np.random.normal() for action in range(envs[key].action_space.n)}

            if next_observation not in q_table[train_key]:
                q_table[train_key][next_observation] = {action: np.random.normal() for action in range(envs[key].action_space.n)}

            # print(key, observations[key], action, reward + cfg.discount * max_or_min(q_table[key][next_observation].values()) * float(not done))
            q_table[train_key][observations[key]][action] = (1 - cfg.learning_rate) * q_table[train_key][observations[key]][action] \
                + cfg.learning_rate * (reward + cfg.discount * {"max": max, "min": min}[train_key](q_table[train_key][next_observation].values()) * float(not done))

            if timestep % cfg.plot_q_values_frequency == 0:
                stats.q_values[0, key][timestep] = {
                    observation_for_q_value: [q_table[key][observation_for_q_value].get(action) for action in range(envs[key].action_space.n)]
                                                if observation_for_q_value in q_table[key] else [None] * envs[key].action_space.n
                    for observation_for_q_value in cfg.observations_for_plotting_q_values
                }

            observations[key] = next_observation

            if done:
                observations[key], _ = envs[key].reset()
            
    if cfg.plot:
        stats.plot(make_env=make_env)

    def q_table_to_function(q):
        def f(observation):
            observation = tuple(observation.flatten().tolist())
            if observation not in q:
                q[observation] = {action: np.random.normal() for action in range(envs["max"].action_space.n)}
            return tensor([q[observation][action] for action in range(envs["max"].action_space.n)])
        return f

    return { "maximizer": q_table_to_function(q_table["max"]),
             "minimizer": q_table_to_function(q_table["min"]) }

class AgentMDPLearning(AspirationAgent):
    # the planning agent is only used for possible_actions and default_policy
    def __init__( self,
                  params: dict,
                  maximizer_model: Callable[[Tensor], Tensor],
                  minimizer_model: Callable[[Tensor], Tensor],
                  planning_agent: AgentMDPPlanning,
                  observation_action_pair_distribution: Dict[str, Dict[Tuple["observation", "action"], float]] | None = None,
                  min_possible_action_frequency: float | None = None):
        
        super().__init__(params | {"defaultPolicy": planning_agent.world.default_policy})
        self.planning_agent = planning_agent
        self.maximizer_model = maximizer_model
        self.minimizer_model = minimizer_model
        self.observation_action_pair_distribution = observation_action_pair_distribution
        self.min_possible_action_frequency = min_possible_action_frequency

    @cache
    def maxAdmissibleQTable(self, state):
        state = tensor([*state], dtype=torch.float)
        return tuple(x.item() for x in self.maximizer_model(state))
    
    @cache
    def minAdmissibleQTable(self, state):
        state = tensor([*state], dtype=torch.float)
        return tuple(x.item() for x in self.minimizer_model(state))

    def maxAdmissibleQ(self, state, action):
        return self.maxAdmissibleQTable(state)[action]

    def minAdmissibleQ(self, state, action):
        return self.minAdmissibleQTable(state)[action]

    def possible_actions(self, state):
        return self.planning_agent.possible_actions(state)

def train_dqn_on_satisfia_policy(make_model: Callable[[], Module], make_env: Callable[[], gym.Env], cfg: DQNConfig):
    stats = DQNTrainingStatistics(cfg, keys=[(0, "max"), (0, "min")])
    observations_for_plotting_q_values = set(cfg.observations_for_plotting_q_values) if cfg.observations_for_plotting_q_values else set()

    env = ToTensorWrapper(make_env())
    q_networks = {"max": make_model(), "min": make_model()}
    target_networks = {"max": make_model(), "min": make_model()}
    for max_or_min in ["max", "min"]:
        target_networks[max_or_min].load_state_dict(q_networks[max_or_min].state_dict())
    optimizers = {max_or_min: AdamW(n.parameters(), lr=cfg.learning_rate) for max_or_min, n in q_networks.items()}

    replay_buffer = ReplayBuffer(cfg.replay_buffer_size, num_actions=env.action_space.n)

    agent_params = {}

    planning_agent = AgentMDPPlanning(agent_params, make_env())
    learning_agent = AgentMDPLearning(agent_params, maximizer_model=target_networks["max"], minimizer_model=target_networks["min"], planning_agent=planning_agent)
    aspiration = cfg.aspiration_sampler()

    observation, _ = env.reset()
    for timestep in tqdm(range(cfg.total_timesteps), desc="training dqn"):
        possible_actions = learning_agent.possible_actions(tuple(observation.int().tolist()))

        exploration_rate = cfg.exploration_rate_scheduler(timestep / cfg.total_timesteps)
        explore = random.random() <= exploration_rate
        if explore:
            action = random.choice(possible_actions)
            action_aspiration = learning_agent.aspiration4action(observation, action, aspiration)
        else:
            action, action_aspiration = learning_agent.localPolicy(tuple(observation.int().tolist()), aspiration).sample()[0]

        next_observation, delta, done, truncated, _ = env.step(action)
        done = done or truncated

        replay_buffer.add( observation = observation,
                           action = action,
                           next_observation = next_observation,
                           delta = delta,
                           possible_actions = possible_actions,
                           done = done )

        if not done and cfg.observations_for_plotting_q_values is None:
            observations_for_plotting_q_values.add(tuple(observation.int().tolist()))

        record_q_values = cfg.plot_q_values and timestep % cfg.plot_q_values_frequency == 0
        if record_q_values:
            for max_or_min in ["max", "min"]:
                # print("q", max_or_min, q_networks[max_or_min](observation)[4])
                # print(q_networks["max"](tensor([9., 2, 2]))[4])
                stats.q_values[(0, max_or_min)][timestep] = {
                    observation_for_q_value: q_networks[max_or_min](tensor(list(observation_for_q_value), dtype=torch.float32)).tolist()
                    for observation_for_q_value in observations_for_plotting_q_values
                }
                # if max_or_min == "max":
                #     timesteps = list(stats.q_values[(0, "max")].keys())
                #     print(timesteps)
                #     print([stats.q_values[(0, "max")][timestep][tuple(observation.int().tolist())][action] for timestep in timesteps])

        aspiration = learning_agent.propagateAspiration( tuple(observation.int().tolist()),
                                                         action,
                                                         action_aspiration,
                                                         Edel=None,
                                                         nextState=tuple(next_observation.int().tolist()) )
        observation = next_observation
        if done or truncated:
            observation, _ = env.reset()
            aspiration = cfg.aspiration_sampler()

            # needed to reset cache
            learning_agent = AgentMDPLearning(agent_params, maximizer_model=target_networks["max"], minimizer_model=target_networks["min"], planning_agent=planning_agent)

        train = timestep >= cfg.learning_starts and timestep % cfg.train_frequency == 0
        if train:
            data = replay_buffer.sample(cfg.batch_size)
            for max_or_min in ["max", "min"]:
                with no_grad():
                    target = target_networks[max_or_min](data.next_observations)
                    target_argmax_network = q_networks[max_or_min] if cfg.double_q_learning else target_networks[max_or_min]
                    target_for_argmax_or_argmin = where( data.possible_actions,
                                                        target_argmax_network(data.next_observations),
                                                        {"max": float("-inf"), "min": float("inf")}[max_or_min] )
                    target_argmax_or_argmin = {"max": torch.argmax, "min": torch.argmin}[max_or_min](target_for_argmax_or_argmin, dim=-1)
                    target_max_or_min = target.gather(-1, target_argmax_or_argmin.unsqueeze(-1)).squeeze(-1)
                    td_target = data.deltas + cfg.discount * target_max_or_min * data.dones.logical_not().float()
                    # if max_or_min == "max":
                    #     print(td_target[(data.observations == tensor([9, 2, 2])).all(-1).logical_and(data.actions == 4)].numel())
                    # print("target", td_target[(data.observations == tensor([9, 2, 2])).all(-1).logical_and(data.actions == 4)])
                q = q_networks[max_or_min](data.observations)
                # print("q", q[(data.observations == tensor([9, 2, 2])).all(-1), 4])
                # if max_or_min == "max":
                #     print(q_networks[max_or_min](tensor([9., 2, 2]))[4])
                q = q.gather(-1, data.actions.unsqueeze(-1)).squeeze(-1)
                td_loss = mse_loss(q, td_target)
                optimizers[max_or_min].zero_grad()
                td_loss.backward()
                optimizers[max_or_min].step()

        update_target_network = timestep >= cfg.learning_starts and timestep % cfg.target_network_update_frequency == 0
        if update_target_network:
            for max_or_min in ["max", "min"]:
                target_networks[max_or_min].load_state_dict(q_networks[max_or_min].state_dict())

            # needed to reset cache
            learning_agent = AgentMDPLearning(agent_params, maximizer_model=target_networks["max"], minimizer_model=target_networks["min"], planning_agent=planning_agent)

    if cfg.plot:
        stats.plot(make_env=make_env, observations_for_plotting_q_values=sorted(list(observations_for_plotting_q_values)))

    return {"maximizer": target_networks["max"], "minimizer": target_networks["min"]}

def train_dqn_random_lambda(make_model: Callable[[], Module], make_env: Callable[[], gym.Env], cfg: DQNConfig):
    stats = DQNTrainingStatistics(cfg, keys=[()])

    env = ToTensorWrapper(make_env())
    q_network = make_model()
    target_network = make_model()
    target_network.load_state_dict(q_network.state_dict())
    optimizer = AdamW(q_network.parameters(), cfg.learning_rate)

    replay_buffer = ReplayBuffer(cfg.replay_buffer_size, num_actions=env.action_space.n)

    observation, _ = env.reset()
    episode_deltas = []
    episode_lambda_sampler = cfg.lambda_sampler()
    for timestep in tqdm(range(cfg.total_timesteps), desc="training dqn"):
        lambda_ = episode_lambda_sampler()

        if hasattr(env, "possible_actions"):
            possible_actions = env.env.possible_actions()
        else:
            possible_actions = list(range(env.action_space.n))

        exploration_rate = cfg.exploration_rate_scheduler(timestep / cfg.total_timesteps)
        explore = random.random() <= exploration_rate
        if explore:
            action = random.choice(possible_actions)
        else:
            q = q_network(tensor([*observation] + [lambda_], dtype=torch.float))
            action = possible_actions[q[possible_actions].argmax().item()]

        next_observation, delta, done, truncated, _ = env.step(action)
        done = done or truncated
        episode_deltas.append(delta)

        replay_buffer.add( observation = observation,
                           action = action,
                           next_observation = next_observation,
                           delta = delta,
                           possible_actions = possible_actions,
                           done = done,
                           lambda_ = lambda_ )
        
        observation = next_observation

        if done or truncated:
            observation, _ = env.reset()
            episode_lambda_sampler = cfg.lambda_sampler()
            stats.episode_lengths[()][timestep] = len(episode_deltas)
            stats.totals[()][timestep] = mean(episode_deltas)
            episode_deltas = []

        record_q_values = cfg.plot_q_values and timestep % cfg.plot_q_values_frequency == 0
        if record_q_values:
            for lambda_for_q_value in cfg.lambdas_for_plotting_q_values:
                stats.q_values[lambda_for_q_value][timestep] = {
                    observation_for_q_value: q_network(tensor([*observation_for_q_value] + [lambda_for_q_value], dtype=torch.float)).tolist()
                    for observation_for_q_value in cfg.observations_for_plotting_q_values
                }

        train = timestep >= cfg.learning_starts and timestep % cfg.train_frequency == 0
        if train:
            with no_grad():
                target_argmax_network = q_network if cfg.double_q_learning else target_network
                data = replay_buffer.sample(cfg.batch_size)
                target = target_network(cat((data.observations, data.lambdas.unsqueeze(-1)), -1))
                target_for_argmax_and_argmin = target_argmax_network(cat((data.next_observations, data.lambdas.unsqueeze(-1)), -1))
                target_for_argmax = where(data.possible_actions, target_for_argmax_and_argmin, float("-inf"))
                target_for_argmin = where(data.possible_actions, target_for_argmax_and_argmin, float("inf"))
                target_argmax = target_for_argmax.argmax(-1)
                target_argmin = target_for_argmin.argmin(-1)
                target_max = target.gather(-1, target_argmax.unsqueeze(-1)).squeeze(-1)
                target_min = target.gather(-1, target_argmin.unsqueeze(-1)).squeeze(-1)
                target_max_min_mix = (1 - data.lambdas) * target_min + data.lambdas * target_max
                td_target = data.deltas + cfg.discount * target_max_min_mix *   data.dones.logical_not().float()
            q = q_network(cat((data.observations, data.lambdas.unsqueeze(-1)), -1))
            q = q.gather(-1, data.actions.unsqueeze(-1)).squeeze(-1)
            td_loss = mse_loss(q, td_target)
            stats.td_losses[()][timestep] = td_loss.item()
            optimizer.zero_grad()
            td_loss.backward()
            optimizer.step()

        update_target_network = timestep >= cfg.learning_starts and timestep % cfg.target_network_update_frequency == 0
        if update_target_network:
            target_network.load_state_dict(q_network.state_dict())

    if cfg.plot:
        stats.plot(make_env=make_env)

    return { "maximizer": lambda observation: q_network(tensor([*observation, 1.])),
             "minimizer": lambda observation: q_network(tensor([*observation, 0.])) }

def train_dqn(make_model: Callable[[], Module], make_env: Callable[[], gym.Env], cfg: DQNConfig):
    max_or_min_keys = []
    if cfg.train_maximizer:
        max_or_min_keys.append("max")
    if cfg.train_minimizer:
        max_or_min_keys.append("min")
    network_keys = [0, 1] if cfg.true_double_q_learning else [0]
    keys = [ (network_key, max_or_min_key)
             for network_key in network_keys
             for max_or_min_key in max_or_min_keys ]

    stats = DQNTrainingStatistics(cfg, keys)

    envs            = {key: ToTensorWrapper(make_env())          for key in keys}
    q_networks      = {key: make_model()                         for key in keys}
    target_networks = {key: make_model()                         for key in keys}
    buffer_models   = {key: make_model()                         for key in keys}
    replay_buffers  = { key: ReplayBuffer(cfg.replay_buffer_size, num_actions=envs[key].action_space.n)
                        for key in keys}
    optimizers      = { key: AdamW(q_network.parameters(), lr=cfg.learning_rate)
                        for key, q_network in q_networks.items() }
    
    def update_buffer_model(key, exploration_rate):
        buffer_models[key].load_state_dict(target_networks[key].state_dict())
        with no_grad():
            for param in buffer_models[key].parameters():
                    param = param + exploration_rate * randn_like(param)
    
    for key in keys:
        update_buffer_model(key, exploration_rate=cfg.exploration_rate_scheduler(0))

    observations   = {key: env.reset()[0] for key, env in envs.items()}
    episode_deltas = {key: []             for key in keys}
    for timestep in tqdm(range(cfg.total_timesteps), desc="training dqn"):
        exploration_rate = cfg.exploration_rate_scheduler(timestep / cfg.total_timesteps)
        for key in keys:
            if hasattr(envs[key], "possible_actions"):
                possible_actions = envs[key].env.possible_actions()
            else:
                possible_actions = list(range(envs[key].action_space.n))

            if cfg.mix_maximizer_and_minimizer_strategies:
                minimizer_action = random.random() <= cfg.lambdas[key[1]]
            else:
                minimizer_action = {"max": False, "min": True}[key[1]]

            if cfg.parameter_noise_exploration:
                q = buffer_models[key](observations[key])
                argmax_or_argmin = torch.argmin if minimizer_action else torch.argmax
                action = possible_actions[argmax_or_argmin(q[possible_actions]).item()]
            else:
                explore = random.random() <= exploration_rate
                if explore:
                    action = random.choice(possible_actions)
                else:
                    q = target_networks[key](observations[key])
                    argmax_or_argmin = torch.argmin if minimizer_action else torch.argmax
                    # argmax_or_argmin = {"max": torch.argmax, "min": torch.argmin}[key[1]]
                    action = possible_actions[argmax_or_argmin(q[possible_actions]).item()]

            next_observation, delta, done, truncated, _ = envs[key].step(action)
            done = done or truncated
            episode_deltas[key].append(delta)

            replay_buffers[key].add( observation      = observations[key],
                                     action           = action,
                                     next_observation = next_observation,
                                     delta            = delta,
                                     possible_actions = possible_actions,
                                     done             = done,
                                     minimizer_action = minimizer_action )

            observations[key] = next_observation

            if done or truncated:
                observations[key], _ = envs[key].reset()

                stats.totals[key][timestep] = float(sum(episode_deltas[key]))
                stats.episode_lengths[key][timestep] = len(episode_deltas[key])
                episode_deltas[key] = []

                if cfg.parameter_noise_exploration:
                    update_buffer_model(key, exploration_rate=cfg.exploration_rate_scheduler(timestep / cfg.total_timesteps))

            record_q_values = cfg.plot_q_values and timestep % cfg.plot_q_values_frequency == 0
            if record_q_values:
                stats.q_values[key][timestep] = {
                    observation_for_q_value: q_networks[key](tensor([*observation_for_q_value], dtype=torch.float)).tolist()
                    for observation_for_q_value in cfg.observations_for_plotting_q_values
                }


            train = timestep >= cfg.learning_starts and timestep % cfg.train_frequency == 0
            if train:
                network_key, max_or_min_key = key
                with no_grad():
                    target_argmax_networks = q_networks if cfg.double_q_learning else target_networks
                    if cfg.true_double_q_learning:
                        target_argmax_network = target_argmax_networks[{1: 0, 0: 1}[network_key], max_or_min_key]
                    else:
                        target_argmax_network = target_argmax_networks[key]

                    is_scheduled = isinstance(cfg.fraction_samples_from_opposite_replay_buffer, Callable)
                    if is_scheduled:
                        fraction_samples_from_opposite_replay_buffer = cfg.fraction_samples_from_opposite_replay_buffer(timestep / cfg.total_timesteps)
                    else:
                        fraction_samples_from_opposite_replay_buffer = cfg.fraction_samples_from_opposite_replay_buffer

                    if fraction_samples_from_opposite_replay_buffer == 0:
                        data = replay_buffers[key].sample(cfg.batch_size)
                    else:
                        network_key, max_or_min_key = key
                        num_samples_from_opposite = int(fraction_samples_from_opposite_replay_buffer * cfg.batch_size)
                        num_samples_from_corresponding = cfg.batch_size - num_samples_from_opposite
                        data = replay_buffers[key].sample(num_samples_from_corresponding)
                        data_from_opposite = replay_buffers[network_key, {"max": "min", "min": "max"}[max_or_min_key]].sample(num_samples_from_opposite)
                        data = data.concatenate(data_from_opposite)
                        data.shuffle()

                    target = target_networks[key](data.next_observations)
                    target_for_argmax_and_argmin = target_argmax_network(data.next_observations)
                    target_for_argmax = where(data.possible_actions, target_for_argmax_and_argmin, float("-inf"))
                    target_for_argmin = where(data.possible_actions, target_for_argmax_and_argmin, float("inf"))
                    target_argmax = target_for_argmax.argmax(-1)
                    target_argmin = target_for_argmin.argmin(-1)
                    target_max = target.gather(-1, target_argmax.unsqueeze(-1)).squeeze(-1)
                    target_min = target.gather(-1, target_argmin.unsqueeze(-1)).squeeze(-1)

                    lambda_ = cfg.lambdas[max_or_min_key]
                    target_max_min_mix = (1 - lambda_) * target_min + lambda_ * target_max

                    td_target = data.deltas + cfg.discount * target_max_min_mix * data.dones.logical_not().float()

                if cfg.train_corresponding_network:
                    # this does the training twice (once for the maximizer and once for the minimizer) instead of just once
                    # fix this
                    for max_or_min, corresponding_actions in [ ("max", data.minimizer_actions.logical_not()),
                                                               ("min", data.minimizer_actions) ]:
                        q = q_networks[network_key, max_or_min](data.observations[corresponding_actions, ...])
                        q = q.gather(-1, data.actions[corresponding_actions, ...].unsqueeze(-1)).squeeze(-1)
                        td_loss = mse_loss(q, td_target[corresponding_actions])
                        stats.td_losses[network_key, max_or_min][timestep] = td_loss.item()
                        optimizers[network_key, max_or_min].zero_grad()
                        td_loss.backward()
                        optimizers[network_key, max_or_min].step()
                else:
                    q = q_networks[key](data.observations)
                    q = q.gather(-1, data.actions.unsqueeze(-1)).squeeze(-1)
                    td_loss = mse_loss(q, td_target)
                    stats.td_losses[key][timestep] = td_loss.item()
                    optimizers[key].zero_grad()
                    td_loss.backward()
                    optimizers[key].step()

            update_target_network = timestep >= cfg.learning_starts \
                                        and timestep % cfg.target_network_update_frequency == 0
            if update_target_network:
                target_networks[key].load_state_dict(q_networks[key].state_dict())

    if cfg.plot:
        stats.plot(make_env=make_env)

    return {"maximizer": target_networks[0, "max"], "minimizer": target_networks[0, "min"]}

@dataclass
class ReplayBufferSample:
    observations: Tensor
    actions: Tensor
    next_observations: Tensor
    deltas: Tensor
    dones: Tensor
    possible_actions: Tensor
    minimizer_actions: Tensor | None
    lambdas: Tensor | None

    def __post_init__(self):
        assert self.observations.size(0) == self.actions.size(0) == self.next_observations.size(0) \
                 == self.deltas.size(0) == self.dones.size(0)
        if self.minimizer_actions is not None:
            assert self.minimizer_actions.size(0) == self.observations.size(0)
        if self.lambdas is not None:
            assert self.lambdas.size(0) == self.observations.size(0)

    def size(self):
        return self.observations.size(0)

    def concatenate(self, other: "ReplayBufferSample"):
        assert (self.minimizer_actions is None) == (other.minimizer_actions is None)
        assert (self.lambdas is None) == (other.lambdas is None)

        return ReplayBufferSample( observations      = cat((self.observations,      other.observations)),
                                   actions           = cat((self.actions,           other.actions)),
                                   next_observations = cat((self.next_observations, other.next_observations)),
                                   deltas            = cat((self.deltas,            other.deltas)),
                                   dones             = cat((self.dones,             other.dones)),
                                   minimizer_actions = cat((self.minimizer_actions, other.minimizer_actions))
                                                            if self.minimizer_actions is not None else None,
                                   lambdas           = cat((self.lambdas,           other.lambdas))
                                                            if self.lambdas           is not None else None )
    
    def shuffle(self):
        shuffled_indices = randperm(self.size())
        self.observations          = self.observations     [shuffled_indices, ...]
        self.actions               = self.actions          [shuffled_indices, ...]
        self.next_observations     = self.next_observations[shuffled_indices, ...]
        self.deltas                = self.deltas           [shuffled_indices, ...]
        self.dones                 = self.dones            [shuffled_indices, ...]
        if self.minimizer_actions is not None:
            self.minimizer_actions = self.minimizer_actions[shuffled_indices, ...]
        if self.lambdas is not None:
            self.lambdas           = self.lambdas          [shuffled_indices, ...]

class ReplayBuffer:
    def __init__(self, size, num_actions=None):
        self.size = size
        self.initialized = False
        self.num_actions = num_actions

    def add( self,
             observation: Tensor,
             action: int,
             next_observation: Tensor,
             delta: float,
             done: bool,
             possible_actions: Tensor | Iterable[int],
             minimizer_action: bool | None = None,
             lambda_: float | None = None ):
        
        if not isinstance(possible_actions, Tensor):
            assert all(isinstance(action, int) for action in possible_actions)
            assert all(action in range(self.num_actions) for action in possible_actions)
            possible_actions = tensor([action in possible_actions for action in range(self.num_actions)])

        if self.num_actions is None:
            self.num_actions = possible_actions.numel()

        if not self.initialized:
            device = observation.device
            self.observations      = empty(self.size, *observation.shape,      dtype=torch.float, device=device)
            self.actions           = empty(self.size,                          dtype=int, device=device)
            self.next_observations = empty(self.size, *next_observation.shape, dtype=torch.float, device=device)
            self.deltas            = empty(self.size,                          dtype=torch.float, device=device)
            self.dones             = empty(self.size,                          dtype=bool, device=device)
            self.possible_actions  = empty(self.size, self.num_actions,        dtype=bool, device=device)
            self.minimizer_actions = empty(self.size,                          dtype=bool, device=device) \
                                        if minimizer_action is not None else None
            self.lambdas           = empty(self.size,                          dtype=torch.float, device=device) \
                                        if lambda_ is not None else None
    
            self.num_written = 0

            self.initialized = True

        self.observations         [self.num_written % self.size, ...] = observation
        self.actions              [self.num_written % self.size]      = action
        self.next_observations    [self.num_written % self.size, ...] = next_observation
        self.deltas               [self.num_written % self.size]      = delta
        self.dones                [self.num_written % self.size]      = done
        self.possible_actions     [self.num_written % self.size, ...] = possible_actions
        if self.minimizer_actions is not None:
            self.minimizer_actions[self.num_written % self.size]      = minimizer_action
        if self.lambdas is not None:
            self.lambdas          [self.num_written % self.size]      = lambda_

        self.num_written += 1

    def sample(self, sample_size: int) -> ReplayBufferSample:
        if sample_size == 0:
            return ReplayBufferSample( observations      = tensor([]),
                                       actions           = tensor([], dtype=torch.int),
                                       next_observations = tensor([]),
                                       deltas            = tensor([]),
                                       dones             = tensor([], dtype=torch.bool),
                                       possible_actions  = tensor([], dtype=torch.bool),
                                       minimizer_actions = tensor([], dtype=torch.bool) if self.minimizer_actions is not None else None,
                                       lambdas           = tensor([]) if self.lambdas is not None else None )
        
        assert sample_size <= self.num_written
        all_indices = range(min(self.num_written, self.size))
        sample_indices = tensor(random.sample(all_indices, sample_size))
        return ReplayBufferSample( observations      = self.observations     [sample_indices, ...],
                                   actions           = self.actions          [sample_indices],
                                   next_observations = self.next_observations[sample_indices, ...],
                                   deltas            = self.deltas           [sample_indices],
                                   dones             = self.dones            [sample_indices],
                                   possible_actions  = self.possible_actions [sample_indices, ...],
                                   minimizer_actions = self.minimizer_actions[sample_indices]
                                                            if self.minimizer_actions is not None else None,
                                   lambdas           = self.lambdas          [sample_indices]
                                                            if self.lambdas is not None else None )

def smoothen(xs: Iterable[float] | Dict[int, float], smoothness: int):
    if isinstance(xs, Dict):
        return dict(zip( smoothen(xs.keys(),   smoothness),
                         smoothen(xs.values(), smoothness) ))
    
    return [mean(chunk) for chunk in chunked(xs, smoothness)]

@dataclass
class TabularQLeraningStatistics:
    cfg: DQNConfig
    q_values: Dict[str, Dict["observation", Dict[int, List[float]]]] = field(default_factory=lambda: {"max": dict(), "min": dict()})

    def register_q_values(self, max_or_min, observation, timestep, q_values):
        if observation not in self.q_values[max_or_min]:
            self.q_values[max_or_min][observation] = dict()
        self.q_values[max_or_min][observation][timestep] = list(q_values) # we need the call to list because it needs to be cloned!

    def plot(self, make_env):
        actions = list(range(len(next(iter(next(iter(self.q_values["max"].values())).values())))))
        fig = Figure()

        planning_agent = AgentMDPPlanning(params={}, world=make_env())
        plot_titles = []
        first_iteration = True
        for max_or_min in ["max", "min"]:
            for observation in tqdm(self.q_values[max_or_min], desc=f"calculating {max_or_min}imizer true q values"):
                plot_titles.append(f"{max_or_min}imizer {observation=}")
                for action in actions:
                    fig.add_scatter( x = list(self.q_values[max_or_min][observation].keys()),
                                     y = [q[action] for q in self.q_values[max_or_min][observation].values()],
                                     name = f"action {action}",
                                     line = dict(color=DEFAULT_PLOTLY_COLORS[action]),
                                     visible = first_iteration )
                    if action in planning_agent.possible_actions(observation):
                        true_value = { "max": planning_agent.maxAdmissibleQ(observation, action),
                                       "min": planning_agent.minAdmissibleQ(observation, action) }[max_or_min]
                    else:
                        true_value = None
                    fig.add_scatter( x = [0, self.cfg.total_timesteps - 1],
                                     y = [true_value] * 2,
                                     name = f"action {action} true q value",
                                     line = dict(dash="dash", color=DEFAULT_PLOTLY_COLORS[action]),
                                     visible=first_iteration )
                first_iteration = False
                    
        fig.update_layout(updatemenus=[dict( type="dropdown",
                                             direction="down",
                                             buttons=[ dict( label=title,
                                                             method="update",
                                                             args=[dict(visible =   [False] * 2 * len(actions) * i
                                                                                  + [True]  * 2 * len(actions)
                                                                                  + [False] * 2 * len(actions) * (len(plot_titles) - i - 1) )] )
                                                       for i, title in enumerate(plot_titles) ] )])
        
        fig.show()

class DQNTrainingStatistics:
    episode_lengths: Dict[Tuple[int, str], Dict[int, float]]
    totals:          Dict[Tuple[int, str], Dict[int, float]]
    td_losses:       Dict[Tuple[int, str], Dict[int, float]]
    q_values:        Dict[Tuple[int, str] | float, Dict[int, Dict["observation", List[float]]]]

    def __init__(self, cfg: DQNConfig, keys: List[Tuple[int, str]]):
        self.keys = keys
        self.cfg = cfg
        self.episode_lengths = {key: dict() for key in self.keys}
        self.totals          = {key: dict() for key in self.keys}
        self.td_losses       = {key: dict() for key in self.keys}
        if self.cfg.lambdas_for_plotting_q_values is not None:
            self.q_values   = {lambda_: dict() for lambda_ in self.cfg.lambdas_for_plotting_q_values}
        else:
            self.q_values   = {key: dict() for key in self.keys}

    def plot(self, make_env=None, observations_for_plotting_q_values=None):
        fig = make_subplots(
            rows = int(self.cfg.train_maximizer) + self.cfg.train_minimizer,
            cols = 2 if self.cfg.true_double_q_learning else 1,
            subplot_titles = [ f"{'second ' if network_key > 0 else ''}{max_or_min_key}imizer"
                               for network_key, max_or_min_key in self.keys ]
                                    if self.keys != [()] else ""
        )
        fig.update_layout(title=self.cfg.plot_title)
        fig.update_layout(xaxis=dict(title="timestep"))
        fig.update_layout(yaxis=dict(type="log"))

        for statistic_name, statistic in \
                [ ("episode length", self.episode_lengths),
                  ("total",          self.totals),
                  ("td loss",        self.td_losses) ]:
            
            for key in self.keys:
                network_key, max_or_min_key = key if len(key) == 2 else [0, None]
                smooth_statistic = smoothen(statistic[key], self.cfg.plot_smoothness)
                fig.add_scatter(
                    x     = list(smooth_statistic.keys()),
                    y     = list(smooth_statistic.values()),
                    name  = statistic_name,
                    mode  = "markers",
                    row   = 2 if self.cfg.train_maximizer and max_or_min_key == "max" else 1,
                    col   = network_key + 1
                )

        fig.show()

        if self.cfg.plot_q_values:
            if observations_for_plotting_q_values is None:
                observations_for_plotting_q_values = self.cfg.observations_for_plotting_q_values

            fig = Figure()
            fig.update_layout( title = self.cfg.plot_title + " q values",
                               xaxis_title = "timestep",
                               yaxis_title = "q value" )

            n_actions = make_env().action_space.n

            plot_titles = []
            visibilities = []
            first_iteration = True
            for key in self.q_values.keys():
                if make_env is not None and isinstance(key, float):
                    planning_agent = AgentMDPPlanning(params={"maxLambda": key}, world=make_env())
                    minimizer = False
                elif make_env is not None and len(key) == 2:
                    planning_agent = AgentMDPPlanning(params={"maxLambda": self.cfg.lambdas[key[1]]}, world=make_env())
                    minimizer = key[1] == "min"
                else:
                    planning_agent = None

                for observation in observations_for_plotting_q_values:
                    plot_titles.append(f"{key} {observation}")
                    visibilities.append([] if visibilities == [] else [False] * len(visibilities[-1]))
                    for action in list(range(n_actions)):
                        if planning_agent is not None and action not in planning_agent.possible_actions(observation):
                            continue

                        timesteps = list(self.q_values[key].keys())
                        fig.add_scatter( x = timesteps,
                                         y = [self.q_values[key][timestep].get(observation, [None] * n_actions)[action] for timestep in timesteps],
                                         line = dict(color=DEFAULT_PLOTLY_COLORS[action]),
                                         name = f"action {action}",
                                         visible = first_iteration )
                        visibilities[-1].append(True)
                        
                        if planning_agent is not None:
                            fig.add_scatter( x = [0, self.cfg.total_timesteps - 1],
                                             y = [planning_agent.minAdmissibleQ(observation, action) if minimizer else planning_agent.maxAdmissibleQ(observation, action)] * 2,
                                             line_color = DEFAULT_PLOTLY_COLORS[action],
                                             line_dash = "dash",
                                             name = f"action {action} true",
                                             visible = first_iteration )
                            visibilities[-1].append(True)

                        first_iteration = False

            visibilities = [v + [False] * (len(visibilities[-1]) - len(v)) for v in visibilities]

            fig.update_layout(updatemenus=[dict( type="dropdown",
                                             direction="down",
                                             buttons=[ dict( label=title,
                                                             method="update",
                                                             args=[dict(visible=visibilities[i])] )
                                                       for i, title in enumerate(plot_titles) ] )])

            fig.show()

class ToTensorWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _to_tensor(self, x):
        if isinstance(x, tuple):
            x = [*x]
        return tensor(x, dtype=torch.float)
    
    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        return self._to_tensor(observation), info
    
    def step(self, *args, **kwargs):
        observation, reward, done, truncated, info = self.env.step(*args, **kwargs)
        return self._to_tensor(observation), reward, done, truncated, info
