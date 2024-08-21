from __future__ import annotations

from satisfia.agents.makeMDPAgentSatisfia import AgentMDPPlanning
from satisfia.util.interval_tensor import IntervalTensor

from torch import Tensor, rand
from torch.nn import Module, MSELoss
from dataclasses import dataclass, field
from more_itertools import pairwise
from typing import Any, Tuple, List, Dict, Callable

class ConstantScheduler:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, _) -> float:
        return self.value

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

def uniform(size, min, max):
    return min + (max - min) * rand(size)

@dataclass
class UniformAspirationSampler:
    min_aspiration: float
    max_aspiration: float

    def __call__(self, size: int) -> IntervalTensor:
        a = uniform(size, self.min_aspiration, self.max_aspiration)
        b = uniform(size, self.min_aspiration, self.max_aspiration)
        return IntervalTensor(a.minimum(b), a.maximum(b))
    
@dataclass
class UniformPointwiseAspirationSampler:
    min_aspiration: float
    max_aspiration: float

    def __call__(self, size: int) -> IntervalTensor:
        a = uniform(size, self.min_aspiration, self.max_aspiration)
        return IntervalTensor(a, a)

@dataclass
class DQNConfig:
    total_timesteps: int = 500_000
    env_type: str = "mujoco"
    num_envs: int = 1
    async_envs: bool = True
    buffer_size: int = 10_000
    learning_rate_scheduler: Callable[[float], float] = \
        ConstantScheduler(1e-3)
    batch_size: int = 128
    training_starts: int = 10_000
    training_frequency: int = 10
    target_network_update_frequency: int = 500
    soft_target_network_update_coefficient: float = 0.
    discount: float = 0.99
    criterion_coefficients_for_loss: Dict[str, float] = \
        field(default_factory=lambda: dict(maxAdmissibleQ=1., minAdmissibleQ=1.))
    criterion_loss_fns: Dict[str, Callable[[Tensor, Tensor], Tensor]] = \
        field(default_factory=lambda: dict( maxAdmissibleQ = MSELoss(),
                                            minAdmissibleQ = MSELoss(),
                                            Q              = MSELoss() ))
    double_q_learning: bool = True
    exploration_rate_scheduler: Callable[[float], float] = \
        PiecewiseLinearScheduler([0., 0.5, 1.], [1., 0.05, 0.05])
    noisy_network_exploration: bool = True
    noisy_network_exploration_rate_scheduler: Callable[[float], float] = \
        PiecewiseLinearScheduler([0., 0.5, 1.], [1., 0.05, 0.05])
    frozen_model_for_exploration: Module | None = None
    satisfia_policy: bool = True
    satisfia_agent_params: Dict[str, Any] = \
        field(default_factory=lambda: dict(defaultPolicy=None))
    aspiration_sampler: Callable[[int], IntervalTensor] = None
    device: str = "cpu"
    plotted_criteria: List[str] | None = None
    plot_criteria_smoothness: int = 1
    plot_criteria_frequency: int | None = None
    states_for_plotting_criteria: List | None = None
    state_aspirations_for_plotting_criteria: List | None = None
    actions_for_plotting_criteria: List | None = None
    planning_agent_for_plotting_ground_truth: AgentMDPPlanning | None = None