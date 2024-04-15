from itertools import chain
from enum import IntEnum
from typing import (
    Generic,
    Iterable,
    Never,
    NamedTuple,
    Self,
    Type,
    TypeAlias,
    TypeVar,
    overload,
)
from logging import getLogger

import numpy as np
# import pygame
# from gymnasium import spaces

from satisfia.util.distribution import categorical
from world_model.mdp_world_model import MDPWorldModel

from .types import *
from .objects import *


log = getLogger(__name__)

T = TypeVar("T")


def tup_replace_at(tup: tuple[T, ...], ix: int, val: T) -> tuple[T, ...]:
    return tup[:ix] + (val,) + tup[ix + 1 :]


class Cell(tuple[ObjectType, ...]):
    def get(self, ot: Type[T] | tuple[Type[T], ...]) -> T | None:
        for t in self:
            if isinstance(t, ot):
                return t
        return None

    def __str__(self) -> str:
        n = len(self)
        if n == 0:
            return "<   >"
        elif n == 1:
            return str(self[0])
        return f"<{''.join(str(o.id) for o in self)}>"

    def __sub__(self, obj: ObjectType) -> "Cell":
        return Cell(tuple(o for o in self if o is not obj))

    def __add__(self, obj: ObjectType) -> "Cell":  # type: ignore[override]
        return Cell(tuple((*self, obj)))


ObsType = TypeVar("ObsType")
Exact = bool
Probability = float
Action = Direction
StateChanges = dict[ObjectType, ObjectType]
Grid = tuple[tuple[Cell, ...], ...]


def _print_changes(changes: StateChanges):
    for bef, aft in changes.items():
        log.debug(f"{bef} ({bef.location}) -> {aft} ({aft.location})")


# based in large part on https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/


class SimpleGridworld(Generic[ObsType], MDPWorldModel[ObsType, Action, None]):
    size: Location
    grid: Grid
    agents: list[Agent]
    t: int
    max_episode_length: int
    uneven_ground_prob: float
    move_probability_F: float

    TransitionDistribution: TypeAlias = dict[Self, tuple[Probability, Exact]]

    @overload
    def __init__(
        self,
        grid: list[list[str]],
        delta_grid: list[list[str]] | None = None,
        *,
        cell_code2delta: dict[str, float] | None = None,
        max_episode_length=1e10,
        uneven_ground_prob: Probability = 0.25,
        move_probability_F: Probability = 0,
        # render_mode=None,
        # time_deltas=[0],
        # timeout_delta=0,
        # fps=4,
        copying: None = None,
        changes: None = None,
    ):
        ...

    @overload
    def __init__(
        self,
        grid: None,
        delta_grid: None = None,
        *,
        cell_code2delta: None = None,
        max_episode_length=1e10,
        uneven_ground_prob: Probability = 0.25,
        move_probability_F: Probability = 0,
        # render_mode=None,
        # time_deltas=[0],
        # timeout_delta=0,
        # fps=4,
        copying: Self,
        changes: dict[ObjectType, ObjectType] | None = None,
    ):
        ...

    def __init__(
        self,
        grid: list[list[str]] | None,
        delta_grid: list[list[str]] | None | Never = None,
        *,
        cell_code2delta: dict[str, float] | None = None,
        max_episode_length=1e10,
        uneven_ground_prob: Probability = 0.25,
        move_probability_F: Probability = 0,
        # render_mode=None,
        # time_deltas=[0],
        # timeout_delta=0,
        # fps=4,
        copying: Self | None = None,
        changes: dict[ObjectType, ObjectType] | None = None,
    ):
        if copying is not None:
            assert grid is None and changes is not None
            self.size = copying.size
            self.agents = [changes.get(a, a) for a in copying.agents]
            self.t = copying.t
            self.max_episode_length = copying.max_episode_length
            self.uneven_ground_prob = copying.uneven_ground_prob
            self.move_probability_F = copying.move_probability_F
            self.grid = copying.grid
            # _print_changes(changes)
            for old, new in changes.items():
                self._set_cell(new.location, self.at(new.location) + new)
                self._set_cell(old.location, self.at(old.location) - old)
        else:
            assert grid is not None
            assert copying is None
            self.size = Location(len(grid[0]), len(grid))
            self.agents = []
            self.t = 0
            self.max_episode_length = int(max_episode_length)
            self.uneven_ground_prob = uneven_ground_prob
            self.move_probability_F = move_probability_F

            _grid: list[tuple[Cell, ...]] = []
            for x in range(self.size.x):
                inner: list[Cell] = []
                for y in range(self.size.y):
                    cls = ObjectType.by_symbol[grid[y][x]]
                    loc = Location(x, y)
                    i: ObjectType
                    if cls is Delta:
                        assert delta_grid
                        assert cell_code2delta
                        i = Delta(loc, cell_code2delta[delta_grid[y][x]])
                    elif cls is EmptySpace:
                        inner.append(Cell(()))
                        continue
                    else:
                        i = cls(loc)
                    inner.append(Cell((i,)))
                    if isinstance(i, Agent):
                        self.agents.append(i)
                _grid.append(tuple(inner))
            self.grid = tuple(_grid)

    def __hash__(self) -> int:
        return hash(self.grid)

    def flat_state(self) -> tuple[float | int]:
        return (
            self.t,
            *chain.from_iterable(o.flat_state() for o in ObjectType.by_id.values()),
        )

    def state_embedding(self):
        # Discart time
        return np.array(self.flat_state()[1:], dtype=np.float32)

    def __getitem__(self, x: int) -> tuple[Cell, ...]:
        return self.grid[x]

    def at(self, loc: tuple[int, int]) -> Cell:
        return self.grid[loc[0]][loc[1]]

    def _set_cell(self, loc: Location, cell: Cell) -> None:
        self.grid = tup_replace_at(
            self.grid,
            loc.x,
            tup_replace_at(self.grid[loc.x], loc.y, cell),
        )

    def __str__(self) -> str:
        res = ""
        for y in range(self.size.y):
            for x in range(self.size.x):
                res += str(self.grid[x][y])
            res += "\n"
        return res

    def _can_move(
        self,
        direction: Direction,
        who: ObjectType | None = None,
    ):
        """
        Return True if the agent or other object (designated by the who parameter)
        can move from the given location to the given target_location.
        """
        if who is None:
            assert len(self.agents) == 1
            who = self.agents[0]

        to_loc = who.location + direction.value
        if not (0 <= to_loc.x < self.size.x and 0 <= to_loc.y < self.size.y):
            return False

        def non_obstat(o: ObjectType):
            match o:
                case Wall():
                    return False
                case EmptyToWall():
                    return o.state != EmptyToWallState.WALL
                case Agent() as a:
                    return a == who
                case Box() as b:
                    return isinstance(who, Agent) and self._can_move(direction, b)
                case Glass() as g:
                    if direction in (Direction.RIGHT, Direction.LEFT):
                        return True
                    return isinstance(who, (Agent, Fragile)) and self._can_move(
                        direction, g
                    )
                case UnevenGround(), Pinnacle(), SlipperyGround() if isinstance(
                    o, (Box, Glass)
                ):
                    raise NotImplementedError(
                        "boxes cannot slide/fall yet"
                    )  # TODO: let boxes slide/fall like agents!
            return True

        target = self.at(to_loc)
        return all(non_obstat(o) for o in target)

    def possible_actions(self, a: ObjectType, *, include_stay) -> Iterable[Action]:
        """Return a list of possible actions from the given state."""
        actions = [
            action
            for action in Direction
            if self._can_move(action, a)
            if include_stay or Direction.STAY is not action
        ]
        if not actions:
            raise ValueError(f"No possible actions from state {self.flat_state()}")
        return actions

    def default_policy(self) -> categorical:
        """Return a default action, if any"""
        return categorical([Direction.STAY], [1.0])

    def is_terminal(self):
        """Return True if the given state is a terminal state."""
        self.flat_state
        is_at_goal = any(
            any(isinstance(o, Goal) for o in self.at(agent.location))
            for agent in self.agents
        )
        return is_at_goal or (self.t == self.max_episode_length)

    def tick(
        self: Self,
        changes_map: dict[ObjectType, ObjectType] | None = None,
        *,
        increase_time=True,
    ) -> Self:
        res = type(self)(None, None, copying=self, changes=changes_map or {})
        if increase_time:
            res.t += 1
        return res

    def state_distance(self, other: "SimpleGridworld") -> float:
        """Return the distance between the two given states, disregarding time."""
        return ((self.state_embedding() - other.state_embedding()) ** 2).sum().sqrt()

    def _handle_falling(
        self,
        falling: Pinnacle | UnevenGround,
        action: Action,
        agent: Agent,
        n_samples: int,
    ) -> dict[Self, Probability]:
        simulated_actions = [
            a
            for a in self.possible_actions(agent, include_stay=False)
            if a is not action.opposite  # won't fall back to where we came from
        ]
        if not simulated_actions:
            return "none"  # None

        p0 = (
            1 if isinstance(falling, Pinnacle) else self.uneven_ground_prob
        )  # probability of falling off
        trans_dist: dict[Self, Probability] = {}
        # intermediate_state = self.tick(changes, increase_time=False)
        for simulate_action in simulated_actions:
            for successor, (
                probability,
                _,
            ) in self.transition_distribution(
                agent, simulate_action, n_samples, timestep=False
            ).items():
                dp = p0 * probability / len(simulated_actions)
                if successor in trans_dist:
                    trans_dist[successor] += dp
                else:
                    trans_dist[successor] = dp

        if isinstance(falling, UnevenGround):
            trans_dist[self] = 1 - p0

        return trans_dist

    def transition_distribution(
        self, agent: Agent, action: Action, n_samples=None, timestep=True
    ) -> TransitionDistribution:
        assert self._can_move(action, agent)

        # if state is None and action is None:
        #     successor = self._make_state()
        #     return {successor: (1, True)}

        if self.is_terminal():
            return {self.tick(): (1.0, True)}

        current: Cell = self.at(agent.location)
        new_agent = agent.moved(action)
        changes: StateChanges = {}

        # Why don't we do these against `target` at the end of the turn?
        if etw := current.get(EmptyToWall):
            changes[etw] = etw.to_wall()

        if delta := current.get(Delta):
            if delta.state is not DeltaState.COLLECTED:
                changes[delta] = delta.collect()
                new_agent = new_agent.collect(delta.delta)

        changes[agent] = new_agent
        target = self.at(agent.location + action.value)

        if fragile := target.get(Fragile):
            assert self._can_move(action, fragile)
            changes[fragile] = fragile.destroy()

        if box := target.get(Box):
            assert self._can_move(action, delta)
            changes[box] = box.moved(action)

        if glass := target.get(Glass):
            if action in (Direction.UP, Direction.DOWN):
                assert self._can_move(action, glass)
                changes[glass] = glass.moved(action)
            else:
                changes[glass] = glass.break_glass()

        world = self.tick(changes, increase_time=timestep)

        trans_dist: dict[Self, Probability]
        if falling := target.get((Pinnacle, UnevenGround)):
            trans_dist = world._handle_falling(
                falling, action, new_agent, n_samples or 1
            )
        else:
            trans_dist = {world: 1.0}

        #  TODO: implement all deterministic changes

        # if self.move_probability_F > 0:
        #     for fr in ObjectType.by_id.values():
        #         if not isinstance(fr, Fragile) or fr.state is FragileState.DESTROYED:
        #             continue
        #
        #         trans_dist = self._handle_fragile(fr, trans_dist, changes)

        # TODO: update object states and/or object locations, e.g. if the agent picks up an object or moves an object

        return {
            successor: (probability, True)
            for (successor, probability) in trans_dist.items()
        }

    def _handle_fragile(
        self,
        fragile: Fragile,
        trans_dist: dict[Self, Probability],
        changes: StateChanges,
    ) -> dict[Self, Probability]:
        new_trans_dist: dict[Self, Probability] = {}
        #
        # for successor, probability in trans_dist.items():
        #     direction_locs = tuple(
        #         (direction, fragile.location + direction.value)
        #         for direction in successor.possible_actions(fragile, include_stay=False)
        #     )
        #     if (n_directions := len(direction_locs)) == 0:
        #         new_trans_dist[successor] = probability
        #         continue
        #
        #     new_trans_dist[successor] = probability * (1 - self.move_probability_F)
        #     p = probability * self.move_probability_F / n_directions
        #     for direction, obj_target_loc in direction_locs:
        #         hits_an_agent = obj_target_loc in (a.location for a in self.agents)
        #         if hits_an_agent:
        #             changes[fragile] = fragile.destroy()
        #         else:  # it moves
        #             changes[fragile] = fragile.moved(direction)
        #             if not (glass := successor.at(obj_target_loc).get(Glass)):
        #                 continue
        #             changes[glass] = glass.break_glass()
        #         new_successor = successor.tick(changes, increase_time=False)
        #         new_trans_dist[new_successor] = p
        return new_trans_dist


#     @cache
#     def observation_and_reward_distribution(
#         self, state, action, successor, n_samples=None
#     ):
#         """
#         Delta for a state accrues when entering the state, so it depends on successor:
#         """
#         if state is None and action is None:
#             return {(self._make_state(), 0): (1, True)}
#         (
#             t,
#             loc,
#             imm_states,
#             mc_locs,
#             mv_locs,
#             mv_states,
#         ) = self._extract_state_attributes(successor)
#         delta = self.time_deltas[t % self.time_deltas.size]
#         if self.delta_xygrid[loc] in self.cell_code2delta:
#             delta += self.cell_code2delta[self.delta_xygrid[loc]]
#         # loop through all immobile objects with state 0, see if agent has met it, and if so add the corresponding Delta:
#         for i in range(self.n_immobile_objects):
#             if imm_states[i] == 0 and loc == self.immobile_object_locations[i]:
#                 delta += self.immobile_object_state0_deltas[i]
#         # do the same for all mobile variable objects:
#         for i in range(self.n_mobile_variable_objects):
#             if mv_states[i] == 0 and loc == get_loc(mv_locs, i):
#                 delta += self.mobile_variable_object_state0_deltas[i]
#         # do the same for all mobile constant objects:
#         for i in range(self.n_mobile_constant_objects):
#             if loc == get_loc(mc_locs, i):
#                 delta += self.mobile_constant_object_deltas[i]
#         # add timeout Delta:
#         if t == self.max_episode_length and self.xygrid[loc] != "G":
#             delta += self.timeout_delta
#         return {(successor, delta): (1, True)}
#
#     # reset() and step() are inherited from MDPWorldModel and use the above transition_distribution():
#
#     def initial_state(self):
#         return self._make_state()
#
#     def reset(self, seed=None, options=None):
#         ret = super().reset(seed=seed, options=options)
#         if self.render_mode == "human" and self._previous_agent_location is not None:
#             self._render_frame()
#         return ret
#
#     def step(self, action):
#         ret = super().step(action)
#         if self.render_mode == "human":
#             self._render_frame()
#         return ret
#
#     def render(self, additional_data=None):
#         #        if self.render_mode == "rgb_array":
#         return self._render_frame(additional_data=additional_data)
#
#     def _init_human_rendering(self):
#         pygame.font.init()  # you have to call this at the start,
#         # if you want to use this module.
#         self._cell_font = pygame.font.SysFont("Helvetica", 30)
#         self._delta_font = pygame.font.SysFont("Helvetica", 10)
#         self._cell_data_font = pygame.font.SysFont("Helvetica", 10)
#         self._action_data_font = pygame.font.SysFont("Helvetica", 10)
#
#     def _render_frame(self, additional_data=None):
#         if self._window is None and self.render_mode == "human":
#             os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (900, 0)
#             pygame.init()
#             pygame.display.init()
#             self._window = pygame.display.set_mode(self._window_shape)
#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()
#
#         canvas = pygame.Surface(self._window_shape)
#         canvas.fill((255, 255, 255))
#         pix_square_size = (
#             self._window_shape[0] / self.xygrid.shape[0]
#         )  # The size of a single grid square in pixels
#
#         # Draw grid contents:
#         for x in range(self.xygrid.shape[0]):
#             for y in range(self.xygrid.shape[1]):
#                 cell_type = self.xygrid[x, y]
#                 cell_code = self.delta_xygrid[x, y]
#                 if cell_code in self.cell_code2delta:
#                     pygame.draw.rect(
#                         canvas,
#                         (255, 255, 240),
#                         (
#                             x * pix_square_size,
#                             y * pix_square_size,
#                             pix_square_size,
#                             pix_square_size,
#                         ),
#                     )
#                 if cell_type == "#" or (
#                     cell_type == ","
#                     and self._immobile_object_states[self.immobile_object_indices[x, y]]
#                     == 1
#                 ):
#                     pygame.draw.rect(
#                         canvas,
#                         (64, 64, 64),
#                         (
#                             x * pix_square_size,
#                             y * pix_square_size,
#                             pix_square_size,
#                             pix_square_size,
#                         ),
#                     )
#                 elif cell_type == "G":
#                     pygame.draw.rect(
#                         canvas,
#                         (0, 255, 0),
#                         (
#                             x * pix_square_size,
#                             y * pix_square_size,
#                             pix_square_size,
#                             pix_square_size,
#                         ),
#                     )
#                 elif (
#                     cell_type == ","
#                     and self._immobile_object_states[self.immobile_object_indices[x, y]]
#                     != 1
#                 ):
#                     pygame.draw.rect(
#                         canvas,
#                         (64, 64, 64),
#                         (
#                             (x + 0.3) * pix_square_size,
#                             (y + 0.8) * pix_square_size,
#                             0.4 * pix_square_size,
#                             0.1 * pix_square_size,
#                         ),
#                     )
#                 elif cell_type == "Δ":
#                     if (
#                         self._immobile_object_states[self.immobile_object_indices[x, y]]
#                         == 0
#                     ):
#                         # draw a small triangle:
#                         pygame.draw.polygon(
#                             canvas,
#                             (224, 224, 0),
#                             (
#                                 (
#                                     (x + 0.3) * pix_square_size,
#                                     (y + 0.7) * pix_square_size,
#                                 ),
#                                 (
#                                     (x + 0.7) * pix_square_size,
#                                     (y + 0.7) * pix_square_size,
#                                 ),
#                                 (
#                                     (x + 0.5) * pix_square_size,
#                                     (y + 0.3) * pix_square_size,
#                                 ),
#                             ),
#                         )
#                 elif cell_type in render_as_char_types:
#                     canvas.blit(
#                         self._cell_font.render(cell_type, True, (0, 0, 0)),
#                         ((x + 0.3) * pix_square_size, (y + 0.3) * pix_square_size),
#                     )
#                 if self._window is None and self.render_mode == "human":
#                     canvas.blit(
#                         self._delta_font.render(f"{x},{y}", True, (128, 128, 128)),
#                         ((x + 0.8) * pix_square_size, (y + 0.1) * pix_square_size),
#                     )
#                     if cell_code in self.cell_code2delta:
#                         canvas.blit(
#                             self._delta_font.render(
#                                 cell_code + f" {self.cell_code2delta[cell_code]}",
#                                 True,
#                                 (0, 0, 0),
#                             ),
#                             ((x + 0.1) * pix_square_size, (y + 0.1) * pix_square_size),
#                         )
#
#         # Render all mobile objects:
#         for i, object_type in enumerate(self.mobile_constant_object_types):
#             x, y = get_loc(self._mobile_constant_object_locations, i)
#             if object_type == "X":  # a box
#                 pygame.draw.rect(
#                     canvas,
#                     (128, 128, 128),
#                     (
#                         (x + 0.1) * pix_square_size,
#                         (y + 0.1) * pix_square_size,
#                         0.8 * pix_square_size,
#                         0.8 * pix_square_size,
#                     ),
#                 )
#             elif object_type == "|":  # a glass pane
#                 pygame.draw.rect(
#                     canvas,
#                     (192, 192, 192),
#                     (
#                         (x + 0.45) * pix_square_size,
#                         (y + 0.1) * pix_square_size,
#                         0.1 * pix_square_size,
#                         0.8 * pix_square_size,
#                     ),
#                 )
#             elif object_type == "F":  # a fragile object
#                 pygame.draw.circle(
#                     canvas,
#                     (255, 0, 0),
#                     ((x + 0.5) * pix_square_size, (y + 0.5) * pix_square_size),
#                     pix_square_size / 4,
#                 )
#
#         #        for i, object_type in enumerate(self.mobile_variable_object_types):
#         #            x, y = get_loc(self._mobile_variable_object_locations, i)
#
#         # Now we draw the agent and its previous location:
#         pygame.draw.circle(
#             canvas,
#             (0, 0, 255),
#             (np.array(self._previous_agent_location) + 0.5) * pix_square_size,
#             pix_square_size / 4,
#             width=3,
#         )
#         pygame.draw.circle(
#             canvas,
#             (0, 0, 255),
#             (np.array(self._agent_location) + 0.5) * pix_square_size,
#             pix_square_size / 3,
#         )
#
#         # Optionally print some additional data:
#         if additional_data is not None:
#             if "cell" in additional_data:  # draw some list of values onto each cell
#                 for x in range(self.xygrid.shape[0]):
#                     for y in range(self.xygrid.shape[1]):
#                         values = set(additional_data["cell"].get((x, y), []))
#                         if len(values) > 0:  # then it is a list
#                             surf = self._cell_data_font.render(
#                                 "|".join([str(v) for v in values]), True, (0, 0, 255)
#                             )
#                             canvas.blit(
#                                 surf,
#                                 (
#                                     (x + 0.5) * pix_square_size
#                                     - 0.5 * surf.get_width(),
#                                     (y + 0.35) * pix_square_size
#                                     - 0.5 * surf.get_height(),
#                                 ),
#                             )
#             if (
#                 "action" in additional_data
#             ):  # draw some list of values next to each cell boundary
#                 for x in range(self.xygrid.shape[0]):
#                     for y in range(self.xygrid.shape[1]):
#                         for action in range(4):
#                             values = set(
#                                 additional_data["action"].get((x, y, action), [])
#                             )
#                             if len(values) > 0:  # then it is a list
#                                 dx, dy = (
#                                     self._action_to_direction[action]
#                                     if action < 4
#                                     else (0, 0)
#                                 )
#                                 surf = self._action_data_font.render(
#                                     "|".join([str(v) for v in values]),
#                                     True,
#                                     (0, 0, 255),
#                                 )
#                                 canvas.blit(
#                                     surf,
#                                     (
#                                         (x + 0.5 + dx * 0.48) * pix_square_size
#                                         - [0.5, 1, 0.5, 0, 0.5][action]
#                                         * surf.get_width(),
#                                         (y + 0.5 + dx * 0.04 + dy * 0.48)
#                                         * pix_square_size
#                                         - [0, 0.5, 1, 0.5, 0.5][action]
#                                         * surf.get_height(),
#                                     ),
#                                 )
#
#         # Finally, add some gridlines
#         for x in range(self.xygrid.shape[0] + 1):
#             pygame.draw.line(
#                 canvas,
#                 (128, 128, 128),
#                 (pix_square_size * x, 0),
#                 (pix_square_size * x, self._window_shape[1]),
#                 width=3,
#             )
#         for y in range(self.xygrid.shape[1] + 1):
#             pygame.draw.line(
#                 canvas,
#                 (128, 128, 128),
#                 (0, pix_square_size * y),
#                 (self._window_shape[0], pix_square_size * y),
#                 width=3,
#             )
#         # And print the time left into the top-right cell:
#         if self.render_mode == "human":
#             canvas.blit(
#                 self._cell_font.render(
#                     f"{self.max_episode_length - self.t}", True, (0, 0, 0)
#                 ),
#                 (
#                     (self.xygrid.shape[0] - 1 + 0.3) * pix_square_size,
#                     (0.3) * pix_square_size,
#                 ),
#             )
#
#             # The following line copies our drawings from `canvas` to the visible window
#             self._window.blit(canvas, canvas.get_rect())
#             pygame.event.pump()
#             pygame.display.update()
#
#             # We need to ensure that human-rendering occurs at the predefined framerate.
#             # The following line will automatically add a delay to keep the framerate stable.
#             self.clock.tick(self._fps)
#         else:  # rgb_array
#             return np.transpose(
#                 np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
#             )
#
#     def close(self):
#         if self._window is not None:
#             pygame.display.quit()
#             pygame.quit()
