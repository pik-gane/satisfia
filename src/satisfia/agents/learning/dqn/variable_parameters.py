from dataclasses import dataclass
from typing import List, Dict, Callable


class ConstantScheduler:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, _) -> float:
        return self.value


@dataclass
class variable_parameters:
    temperature: float = 0.5
    period: int = 10
    batch_size: int = 128
    total_timesteps: int = 500_000
    learning_rate_scheduler: Callable[[float], float] = \
        ConstantScheduler(1e-3)

variable_params = variable_parameters()