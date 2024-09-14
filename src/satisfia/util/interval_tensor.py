from torch import Tensor, full_like, stack
from dataclasses import dataclass
from numbers import Number
from typing import Union

@dataclass
class IntervalTensor:
    lower: Tensor
    upper: Tensor

    def clip_to(self, bounds: "IntervalTensor"):
        return IntervalTensor( self.lower.maximum(bounds.lower).minimum(bounds.upper),
                               self.upper.maximum(bounds.lower).minimum(bounds.upper) )
    
    def midpoint(self) -> Tensor:
        return (self.lower + self.upper) / 2

    def squeeze(self, dim):
        return IntervalTensor( self.lower.squeeze(dim),
                               self.upper.squeeze(dim) )

    def unsqueeze(self, dim):
        return IntervalTensor( self.lower.unsqueeze(dim),
                               self.upper.unsqueeze(dim) )
    
    def where(self, condition: Tensor, other: "IntervalTensor") -> "IntervalTensor":
        return IntervalTensor( self.lower.where(condition, other.lower),
                               self.upper.where(condition, other.upper) )
    
    def to(self, device):
        return IntervalTensor(self.lower.to(device), self.upper.to(device))
    
    def gather(self, dim: int, other: Tensor) -> "IntervalTensor":
        return IntervalTensor( self.lower.gather(dim, other),
                               self.upper.gather(dim, other) )

    def __add__(self, other: Union["IntervalTensor", Tensor, Number]) -> "IntervalTensor":
        if isinstance(other, IntervalTensor):
            return IntervalTensor( self.lower + other.lower,
                                   self.upper + other.upper )
        
        if isinstance(other, (Tensor, Number)):
            return IntervalTensor( self.lower + other,
                                   self.upper + other )

        raise TypeError(f"Cannot add IntervalTensor and object of type '{type(other)}'.")
    
    def __sub__(self, other: Union["IntervalTensor", Tensor, Number]) -> "IntervalTensor":
        if isinstance(other, IntervalTensor):
            return IntervalTensor( self.lower + other.upper,
                                   self.upper + other.lower )
        
        if isinstance(other, (Tensor, Number)):
            return IntervalTensor( self.lower + other,
                                   self.upper + other )

        raise TypeError(f"Cannot add IntervalTensor and object of type '{type(other)}'.")
    
    def __getitem__(self, i) -> "IntervalTensor":
        return IntervalTensor(self.lower[i], self.upper[i])
    
    def __iter__(self):
        return iter(zip(self.lower, self.upper))
    
    def __len__(self):
        return len(self.lower)


    def __setitem__(self, i, x: "IntervalTensor"):
        self.lower[i] = x.lower
        self.upper[i] = x.upper

def interpolate(x, lambda_, y):
    return x + (y - x) * lambda_

def relative_position(x, z, y, eps=1e-5):
    diff = y - x
    return ((z - x) / diff).where(diff.abs() >= eps, full_like(diff, 0.5))