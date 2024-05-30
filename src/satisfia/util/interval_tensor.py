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

    def unsqueeze(self, dim):
        return IntervalTensor( self.lower.unsqueeze(dim),
                               self.upper.unsqueeze(dim) )
    
    def where(self, condition: Tensor, other: "IntervalTensor") -> "IntervalTensor":
        return IntervalTensor( self.lower.where(condition, other.lower),
                               self.upper.where(condition, other.upper) )
    
    def to(self, device):
        return IntervalTensor(self.lower.to(device), self.upper.to(device))
    

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
    
    def __setitem__(self, i, x: "IntervalTensor"):
        self.lower[i] = x.lower
        self.upper[i] = x.upper

def relative_position(x, z, y, eps=1e-5):
    diff = y - x
    return ((z - x) / diff).where(diff.abs() >= eps, full_like(diff, 0.5))