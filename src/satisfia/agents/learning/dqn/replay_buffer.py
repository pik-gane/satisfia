from satisfia.util.interval_tensor import IntervalTensor

import torch
from torch import Tensor, empty, arange
import random
from dataclasses import dataclass

@dataclass
class ReplayBufferSample:
    observations: Tensor
    actions: Tensor
    deltas: Tensor
    dones: Tensor
    next_observations: Tensor
    aspirations: IntervalTensor

    def to(self, device: str) -> "ReplayBufferSample":
        return ReplayBufferSample( observations      = self.observations     .to(device),
                                   actions           = self.actions          .to(device),
                                   deltas            = self.deltas           .to(device),
                                   dones             = self.dones            .to(device),
                                   next_observations = self.next_observations.to(device),
                                   aspirations       = self.aspirations      .to(device) )

class ReplayBuffer:
    def __init__(self, size, device="cpu"):
        self.size = size
        self.device = device
        self.num_written = 0
        self.initialized = False

    def add( self,
             observations: Tensor,
             actions: Tensor,
             deltas: Tensor,
             dones: Tensor,
             next_observations: Tensor,
             aspirations: IntervalTensor ):
        
        if not self.initialized:
            self.observations      = empty(self.size, *observations.shape[1:],      device=self.device)
            self.actions           = empty(self.size, dtype=torch.long,             device=self.device)
            self.deltas            = empty(self.size,                               device=self.device)
            self.dones             = empty(self.size, dtype=torch.bool,             device=self.device)
            self.next_observations = empty(self.size, *next_observations.shape[1:], device=self.device)
            self.aspirations       = IntervalTensor( empty(self.size, device=self.device),
                                                     empty(self.size, device=self.device) )

            self.initialized = True

        num_newly_written = observations.size(0)

        i_write = arange(self.num_written, self.num_written + num_newly_written) % self.size
        self.observations     [i_write, ...] = observations.float()
        self.actions          [i_write]      = actions
        self.deltas           [i_write]      = deltas.float()
        self.dones            [i_write]      = dones
        self.next_observations[i_write, ...] = next_observations.float()
        self.aspirations      [i_write]      = aspirations

        self.num_written += num_newly_written

    def sample(self, how_many: int):
        i = random.sample(range(min(self.num_written, self.size)), how_many)
        return ReplayBufferSample( observations      = self.observations[i, ...],
                                   actions           = self.actions[i],
                                   deltas            = self.deltas[i],
                                   dones             = self.dones[i],
                                   next_observations = self.next_observations[i, ...],
                                   aspirations       = self.aspirations[i] )