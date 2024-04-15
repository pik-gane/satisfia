from enum import IntEnum
from typing import (
    ClassVar,
    Generic,
    Self,
    TypeVar,
)

from world_model.types import Direction, Location

ObjState = TypeVar("ObjState", bound=IntEnum)


class ObjectType(Generic[ObjState]):
    symbol: ClassVar[str]
    immobile: ClassVar[bool]

    by_symbol: ClassVar[dict[str, type["ObjectType"]]] = {}
    by_id: ClassVar[dict[int, "ObjectType"]] = {}

    def __init_subclass__(cls, symbol: str) -> None:
        cls.symbol = symbol
        cls.by_symbol[symbol] = cls

    @classmethod
    def next_id(cls):
        return len(cls.by_id) + 1

    id: int
    state: ObjState | None
    """ None is interpreted as "initial state" """
    location: Location

    def __init__(self, location: Location, *, copying: Self | None = None) -> None:
        if copying is not None:
            self.id = copying.id
            self.state = copying.state
            self.location = location
        else:
            self.id = self.next_id()
            self.by_id[self.id] = self
            self.state = None
            self.location = location

    def __str__(self) -> str:
        return f"<{self.symbol}-{self.id}>"

    def __hash__(self) -> int:
        return hash((self.id, self.state, self.location))

    def flat_state(self) -> tuple:
        return (*self.location, self.state.value if self.state else 0)

    def moved(self: Self, direction: Direction) -> "Self":
        return type(self)(self.location + direction.value, copying=self)


class Wall(ObjectType, symbol="#"):
    pass


class EmptySpace(ObjectType, symbol=" "):
    def __init__(self, location: Location) -> None:
        self.id = 0
        self.state = None
        self.location = location

    def __str__(self) -> str:
        return "<   >"


class UnevenGround(ObjectType, symbol="~"):
    # Agents/boxes might fall off to any side except to where agent came from,
    # with equal probability
    pass


class Pinnacle(ObjectType, symbol="^"):
    # Pinnacle (Climbing on it will result in falling off to any side except
    # to where agent came from, with equal probability)
    pass


class Box(ObjectType, symbol="X"):
    # can be pushed around but not pulled, can slide and fall off.
    # Heavy, so agent can only push one at a time
    pass


class Agent(ObjectType, symbol="A"):
    collected_delta: float

    def __init__(self, location: Location, delta: float = 0.0, *, copying=None) -> None:
        super().__init__(location, copying=copying)
        self.collected_delta = delta

    def collect(self: Self, additional_delta: float) -> Self:
        return type(self)(
            self.location, self.collected_delta + additional_delta, copying=self
        )

    def moved(self: Self, direction: Direction) -> "Self":
        res = super().moved(direction)
        res.collected_delta = self.collected_delta
        return res

    def flat_state(self) -> tuple:
        return (*super().flat_state(), self.collected_delta)


class SlipperyGround(ObjectType, symbol="-"):
    # Slippery ground (Agents and boxes might slide along in a straight line;
    # after sliding by one tile,
    # a coin is tossed to decide whether we slide another tile, and this is repeated
    # until the coin shows heads or we hit an obstacle.
    # All this motion takes place within a single time step.)
    pass


class Goal(ObjectType, symbol="G"):
    # Goal or exit door (acting while on it ends the episode)
    pass


class EmptyToWallState(IntEnum):
    WALL = 1


class EmptyToWall(ObjectType[EmptyToWallState], symbol=","):
    # Empty tile that turns into a wall after leaving it (so that one cannot go back)
    def to_wall(self) -> Self:
        n = type(self)(self.location, copying=self)
        n.state = EmptyToWallState.WALL
        return n


class DeltaState(IntEnum):
    COLLECTED = 1


class Delta(ObjectType[DeltaState], symbol="Δ"):
    # positive or negative, can be collected once, does not end the episode
    delta: float

    def __init__(self, location: Location, delta: float, *, copying=None) -> None:
        super().__init__(location, copying=copying)
        self.delta = delta

    def __str__(self) -> str:
        return f"<Δ {int(self.delta)}>"

    def collect(self: Self) -> Self:
        res = type(self)(self.location, self.delta, copying=self)
        res.state = DeltaState.COLLECTED
        return res

    def flat_state(self) -> tuple:
        return (*super().flat_state(), self.delta)


class GlassState(IntEnum):
    BROKEN = 1


class Glass(ObjectType[GlassState], symbol="|"):
    # A pane of glass, will break if anything moves into it from left or right,
    # and can be pushed up or down
    def break_glass(self):
        res = self.moved(Direction.STAY)
        res.state = GlassState.BROKEN
        return res


class FragileState(IntEnum):
    DESTROYED = 1


class Fragile(ObjectType[FragileState], symbol="F"):
    # A fragile object or organism (might move around on its own,
    # is destroyed when stepped upon by the agent)
    def destroy(self):
        res = self.moved(Direction.STAY)
        res.state = FragileState.DESTROYED
        return res
