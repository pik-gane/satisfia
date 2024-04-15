from enum import Enum
from functools import cached_property
from typing import NamedTuple

class Location(NamedTuple):
    x: int
    y: int

    def __sub__(self, other: "Location") -> "Location":
        return Location(self.x - other.x, self.y - other.y)

    def __add__(self, other: "Location") -> "Location":  # type: ignore[override]
        return Location(self.x + other.x, self.y + other.y)

class Direction(Enum):
    UP = Location(0, -1)
    RIGHT = Location(1, 0)
    DOWN = Location(0, 1)
    LEFT = Location(-1, 0)
    STAY = Location(0, 0)

    @cached_property
    def opposite(self) -> "Direction":
        return type(self)(Location(-self.value.x, -self.value.y))


