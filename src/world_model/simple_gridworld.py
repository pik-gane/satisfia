from functools import cache, lru_cache
import os

from satisfia.util import distribution
from . import MDPWorldModel

# based in large part on https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

import numpy as np
from numpy import random
import pygame

import gymnasium as gym
from gymnasium import spaces

unenterable_cell_types = ['#']
unsteady_cell_types = ['~', '^', '-']

immobile_object_types = [',']
mobile_constant_object_types = ['X']
mobile_variable_object_types = ['F']

render_as_char_types = unsteady_cell_types + immobile_object_types + ['G']

max_n_object_states = 2


def set_entry(iterable, index, value):
    if type(iterable) is tuple:
        l = list(iterable)
        l[index] = value 
        return tuple(l)
    else:
        iterable[index] = value
        return iterable

def set_loc(locs, index, loc):
    return set_entry(set_entry(locs, 2*index+1, loc[1]), 2*index, loc[0])

def get_loc(locs, index):
    return (locs[2*index], locs[2*index+1])

def state_embedding_for_distance(state):
    """return an embedding of state where all entries -2 are replaced by -100"""
    return tuple(-100 if x == -2 else x for x in state)

class MultiDiscreteRanged(spaces.MultiDiscrete):
    def __init__(self, nvec, start=None):
        if start == None:
            start = [0] * len(nvec)
        assert len(nvec) == len(start)
        self.start = start

        super().__init__(nvec)

    def sample(self):
        result = super().sample()
        return [value + offset for value, offset in zip(result, self.start)]

class SimpleGridworld(MDPWorldModel):
    """A world model of a simple MDP-type Gridworld environment.
    
    A *state* here is a tuple of integers encoding the following sequence of items,
    each one encoded as one or two integers:

    - position 0: the current time step
    - positions 1+2: the previous location x,y of the agent
    - positions 3+4: the current location x,y of the agent
    - positions 5...4+k: for each of k immobile objects with variable state, its state
    - positions 5+k..4+k+2*l: for each of l mobile objects without a variable state, its location x,y 
    - positions 5+k+2*l...4+k+2*l+2*m: for each of m mobile objects with a variable state, its location x,y
    - positions 5+k+2*l+2*m...4+k+2*l+3*m: for each of m mobile objects with a variable state, its state

    A *coordinate* in a location is encoded in as an integer, where:
    - -2 means the object is not present
    - -1 means the object is in the agent's inventory
    - >= 0 is a coordinate in the grid, counted from top-left to bottom-right

    Objects are *ordered* by their initial location in the ascii-art grid representation in row-major order.

    The *grid* and the agent's and all objects' *initial locations* are given as a 2d array of characters,
    each representing one cell of the grid, with the following character meanings: 

    - already implemented:   
        - '#' (hash): wall
        - ' ' (blank): empty space
        - '~': Uneven ground (Agents/boxes might fall off to any side except to where agent came from, 
               with equal probability)
        - '^': Pinnacle (Climbing on it will result in falling off to any side except to where agent came from, 
               with equal probability)
        - 'A': agent's initial location 
        - 'X': Box (can be pushed around but not pulled, can slide and fall off. Heavy, so agent can only push one at a time)

    - not yet implemented, but are planned to be implemented in the future:
        - ',': Empty tile that turns into a wall after leaving it (so that one cannot go back)
        - '-': Slippery ground (Agents and boxes might slide along in a straight line; after sliding by one tile, 
               a coin is tossed to decide whether we slide another tile, and this is repeated 
               until the coin shows heads or we hit an obstacle. All this motion takes place within a single time step.)
        - '%': Death trap (Episode ends when agent steps on it) 
        - 'B': Button (can be stepped on)
        - 'C': Collaborator (might move around)
        - 'D': Door (can only be entered after having collected a key)
        - 'E': Enemy (might move around on its own)
        - 'F': A fragile object or organism (might move around on its own, is destroyed when stepped upon)
        - 'Δ': Delta (positive or negative, can be collected once, does not end the episode)
        - 'G': Goal or exit door (acting while on it ends the episode)
        - 'I': (Potential) interruption (agent might get stuck in forever)
        - 'K': Key (must be collected to be able to pass a door)
        - 'O': Ball (when pushed, will move straight until meeting an obstacle)
        - 'S': Supervisor (might move around on their own)
        - 'T': Teleporter (sends the agent to some destination t)
        - 't': Destination of a teleporter (stepping on it does nothing)
    (TODO: compare with pycolab asciiart conventions, try to harmonize them, and add things that are missing)

    *Deltas* (rewards) can accrue from the following events:
    - Time passing. This is specified by time_delta or a list time_deltas of length max_episode_length.
    - The agent stepping onto a certain object. This is specified by a list object_deltas
      ordered by the objects' initial locations in the ascii-art grid representation in row-major order.
    - The agent currently being in a certain location. This is specified by 
        - another 2d array of characters, delta_grid, of the same size as the grid, 
          containing cell_codes with the following character meanings:
            - ' ' (space): no Delta
            - '<character>': Delta as specified by cell_code2delta['<character>']
        - a dictionary cell_code2delta listing the actual Delta values for each cell_code in that grid
      Note that the delta accrues at each time point when the agent is in a cell, 
      not at the time point it steps onto it!
    """


    ## parameters:
    xygrid = None
    """(2d array of characters) The grid as an array of strings, each string representing one row of the grid,
    each character representing one cell of the grid"""
    delta_xygrid = None
    """(2d array of characters) codes for deltas (rewards) for each cell of the grid"""
    cell_code2delta = None
    """(dictionary) maps cell codes to deltas (rewards)"""

    n_immobile_objects = None
    """The number of immobile objects with variable state."""
    n_mobile_constant_objects = None
    """The number of mobile objects without a variable state."""
    n_mobile_variable_objects = None
    """The number of mobile objects with a variable state."""
    max_episode_length = None
    """The maximum number of steps in an episode."""
    initial_agent_location = None
    """(pair of ints) The initial location of the agent starting with zero."""
    time_deltas = None
    """(list of floats) The deltas (rewards) for each time step."""
    timeout_delta = None
    """(float) The delta (reward) for the timeout event."""
    move_probability_F = None
    """(float) The probability with which objects of type 'F' move uniformly at random."""

    # additional attributes:
    _state = None
    """(singleton list of state) The current state encoded as a tuple of ints."""
    t = None
    """The current time step."""
    _agent_location = None
    """(pair of ints) The current location of the agent starting with zero."""
    _previous_agent_location = None
    """(pair of ints) The previous location of the agent starting with zero."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode = None, 
                 grid = [['A','G']],
                 delta_grid = None,
                 cell_code2delta = {'1': 1},
                 max_episode_length = 1e10,
                 time_deltas = [0],
                 timeout_delta = 0,
                 uneven_ground_prob = 0.25,
                 move_probability_F = 0,
                 fps = 4
                 ):

        self.xygrid = xygrid = np.array(grid).T
        self.delta_xygrid = delta_xygrid = np.array(delta_grid).T if delta_grid is not None else np.full(xygrid.shape, ' ')
        self.cell_code2delta = cell_code2delta
        self.max_episode_length = max_episode_length
        self.time_deltas = np.array(time_deltas).flatten()
        self.timeout_delta = timeout_delta
        self.move_probability_F = move_probability_F
        self.uneven_ground_prob = uneven_ground_prob
        self._fps = fps

        self._window_shape = 800 * np.array(xygrid.shape) / np.max(xygrid.shape)  # The size of the PyGame window in pixels

        # The initial agent location is the first occurrence of 'A' in the grid:
        wh = np.where(xygrid == 'A')
        self.initial_agent_location = (wh[0][0], wh[1][0])

        self.n_immobile_objects = self.n_mobile_constant_objects = self.n_mobile_variable_objects = 0  # TODO: extract from grid

        # Construct an auxiliary grid that contains a unique index of each immobile object 
        # (cells of a type in immobile_object_types), or None if there is none.
        # Also, get lists of objects and their types and initial locations.
        self.immobile_object_indices = np.full(xygrid.shape, None)
        self.mobile_constant_object_types = []
        self.mobile_constant_object_initial_locations = []
        self.mobile_variable_object_types = []
        self.mobile_variable_object_initial_locations = []
        for x in range(xygrid.shape[0]):
            for y in range(xygrid.shape[1]):
                if xygrid[x, y] in immobile_object_types:
                    self.immobile_object_indices[x, y] = self.n_immobile_objects
                    self.n_immobile_objects += 1
                elif xygrid[x, y] in mobile_constant_object_types:
                    self.mobile_constant_object_types.append(xygrid[x, y])
                    self.mobile_constant_object_initial_locations += [x, y]
                    self.n_mobile_constant_objects += 1
                elif xygrid[x, y] in mobile_variable_object_types:
                    self.mobile_variable_object_types.append(xygrid[x, y])
                    self.mobile_variable_object_initial_locations += [x, y]
                    self.n_mobile_variable_objects += 1

        # The observation returned for reinforcement learning equals state, as described above.
        # TODO how to specify start range of each dimension for MultiDiscrete?
        nx, ny = xygrid.shape[0], xygrid.shape[1]
        self.observation_space = MultiDiscreteRanged(
            [max_episode_length,  # current time step
             nx, ny,  # previous location
             nx, ny]  # current location
            + [max_n_object_states] * self.n_immobile_objects 
            + [nx, ny] * self.n_mobile_constant_objects 
            + [nx, ny] * self.n_mobile_variable_objects
            + [max_n_object_states] * self.n_mobile_variable_objects
            , start = 
            [0,  # current time step
             -2, -2,  # current location
             -2, -2]  # previous location
            + [0] * self.n_immobile_objects 
            + [-2, -2] * self.n_mobile_constant_objects 
            + [-2, -2] * self.n_mobile_variable_objects
            + [0] * self.n_mobile_variable_objects
            )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        """
        self._action_to_direction = {
            0: np.array([0, -1]), # up
            1: np.array([1, 0]), # right
            2: np.array([0, 1]), # down
            3: np.array([-1, 0]), # left
            4: np.array([0, 0]), # stay in place
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode == "human":
            self._init_human_rendering()

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self._window = None
        self.clock = None

    def _get_target_location(self, location, action):
        """Return the next location of the agent if it takes the given action from the given location."""
        direction = self._action_to_direction[action]
        return (
            location[0] + direction[0],
            location[1] + direction[1]
        )

    def _can_move(self, from_loc, to_loc, state, who='A'):
        """Return True if the agent can move from the given location to the given target_location."""
        if not (0 <= to_loc[0] < self.xygrid.shape[0]
                and 0 <= to_loc[1] < self.xygrid.shape[1]
                and not self.xygrid[to_loc] in unenterable_cell_types):
            return False
        # TODO: add other conditions for not being able to move, e.g. because of other objects
        t, agent_loc, prev_loc, imm_states, mc_locs, mv_locs, mv_states = self._extract_state_attributes(state)
        if self.xygrid[to_loc] == ',':
            # can only move there if it hasn't turned into a wall yet:
            if imm_states[self.immobile_object_indices[to_loc]] > 0:
                return False
        # loop through all mobile objects and see if they hinder the movement:
        for i, object_type in enumerate(self.mobile_constant_object_types):
            if (mc_locs[2*i],mc_locs[2*i+1]) == to_loc:
                if object_type == 'X':  # a box
                    if who == 'X':
                        return False  # agent can only push one box!
                    # see if it can be pushed:
                    box_target_loc = tuple(2*np.array(to_loc) - np.array(from_loc))
                    if not self._can_move(to_loc, box_target_loc, state, who='X'):
                        return False
            # TODO: implement destroying an 'F' by pushing a 'X' onto it 
        return True

    def opposite_action(self, action):
        """Return the opposite action to the given action."""
        return 4 if action == 4 else (action + 2) % 4
        
    def state_embedding(self, state):
        res = np.array(state, dtype=np.float32)[3:]  # make time and previous position irrelevant
        return res

    @lru_cache(maxsize=None)
    def possible_actions(self, state):
        """Return a list of possible actions from the given state."""
        t, loc, prev_loc, imm_states, mc_locs, mv_locs, mv_states = self._extract_state_attributes(state)
        actions = [action for action in range(5) 
                    if self._can_move(loc, self._get_target_location(loc, action), state)]
        if len(actions) == 0:
            raise ValueError(f"No possible actions from state {state}") # FIXME: raise a more specific exception
        return actions

    def default_policy(self, state):
        """Return a default action, if any"""
        return distribution.categorical([4], [1])  # staying in place

    def _extract_state_attributes(self, state):
        """Return the individual attributes of a state."""
        return (state[0],  # time step
                (state[3], state[4]),  # current location
                (state[1], state[2]),  # previous location
                state[5 
                      : 5+self.n_immobile_objects],  # immobile object states
                state[5+self.n_immobile_objects 
                      : 5+self.n_immobile_objects+2*self.n_mobile_constant_objects],  # mobile constant object locations
                state[5+self.n_immobile_objects+2*self.n_mobile_constant_objects
                      : 5+self.n_immobile_objects+2*self.n_mobile_constant_objects+2*self.n_mobile_variable_objects],  # mobile variable object locations
                state[5+self.n_immobile_objects+2*self.n_mobile_constant_objects+2*self.n_mobile_variable_objects
                      : 5+self.n_immobile_objects+2*self.n_mobile_constant_objects+3*self.n_mobile_variable_objects]  # mobile variable object states
         )

    def _set_state(self, state):
        """Set the current state to the provided one."""
        self._state = state
        self.t, loc, prev_loc, imm_states, mc_locs, mv_locs, mv_states = self._extract_state_attributes(state)
        self._agent_location = loc
        self._previous_agent_location = prev_loc
        self._immobile_object_states = imm_states
        self._mobile_constant_object_locations = mc_locs
        self._mobile_variable_object_locations = mv_locs
        self._mobile_variable_object_states = mv_states

    def _make_state(self, t = 0, loc = None, prev_loc = None, 
                    imm_states = None, mc_locs = None, mv_locs = None, mv_states = None):
        """Compile the given attributes into a state encoding that can be returned as an observation."""
        if loc is None:
            loc = self.initial_agent_location
        if prev_loc is None:
            prev_loc = (-2, -2)
        if mc_locs is None:
            mc_locs = self.mobile_constant_object_initial_locations
        if mv_locs is None:
            mv_locs = self.mobile_variable_object_initial_locations
        # default states are 0:
        if imm_states is None:
            imm_states = np.zeros(self.n_immobile_objects, dtype = int)
        if mv_states is None:
            mv_states = np.zeros(self.n_mobile_variable_objects, dtype = int)
        return (t, 
                prev_loc[0], prev_loc[1],
                loc[0], loc[1],
                *imm_states,
                *mc_locs,
                *mv_locs,
                *mv_states
                )

    @lru_cache(maxsize=None)
    def is_terminal(self, state):
        """Return True if the given state is a terminal state."""
        t, loc, _, _, _, _, _ = self._extract_state_attributes(state)
        is_at_goal = self.xygrid[loc] == 'G'
        return (t == self.max_episode_length) or is_at_goal

    @lru_cache(maxsize=None)
    def state_distance(self, state1, state2):
        """Return the distance between the two given states, disregarding time."""
        return np.sqrt(np.sum(np.power(np.array(state_embedding_for_distance(state1))[1:] 
                                       - np.array(state_embedding_for_distance(state2))[1:], 2)))

    @lru_cache(maxsize=None)
    def transition_distribution(self, state, action, n_samples = None):
        if state is None and action is None:
            successor = self._make_state()
            return {successor: (1, True)}
        t, loc, prev_loc, imm_states, mc_locs, mv_locs, mv_states = self._extract_state_attributes(state)
        cell_type = self.xygrid[loc]
        at_goal = cell_type == 'G'
        if at_goal:
            successor = self._make_state(t + 1, loc, loc, imm_states, mc_locs, mv_locs, mv_states)
            return {successor: (1, True)}
        else:
            if cell_type == ',':
                # turn into a wall:
                imm_states = set_entry(imm_states, self.immobile_object_indices[loc], 1)

            target_loc = self._get_target_location(loc, action)
            target_type = self.xygrid[target_loc]

            # loop through all mobile objects and see if they are affected by the action:
            for i, object_type in enumerate(self.mobile_constant_object_types):
                if (mc_locs[2*i],mc_locs[2*i+1]) == target_loc:
                    if object_type == 'X':  # a box
                        # see if we can push it:
                        box_target_loc = self._get_target_location(target_loc, action)
                        if self._can_move(target_loc, box_target_loc, state):
                            if self.xygrid[box_target_loc] in unsteady_cell_types:
                                raise NotImplementedError("boxes cannot slide/fall yet")  # TODO: let boxes slide/fall like agents!
                            mc_locs = set_loc(mc_locs, i, box_target_loc)

            if target_type in ['^', '~']:
                # see what "falling-off" actions are possible:
                simulated_actions = [a for a in range(4) 
                      if a != self.opposite_action(action) # won't fall back to where we came from
                         and self._can_move(target_loc, self._get_target_location(target_loc, a), state)]
                if len(simulated_actions) > 0:
                    p0 = 1 if target_type == '^' else self.uneven_ground_prob  # probability of falling off
                    intermediate_state = self._make_state(t, target_loc, loc, imm_states, mc_locs, mv_locs, mv_states)
                    trans_dist = {}
                    # compose the transition distribution recursively:
                    for simulate_action in simulated_actions:
                        for (successor, (probability, _)) in self.transition_distribution(intermediate_state, simulate_action, n_samples).items():
                            dp = p0 * probability / len(simulated_actions)
                            if successor in trans_dist:
                                trans_dist[successor] += dp
                            else:
                                trans_dist[successor] = dp
                    if target_type == '~':
                        trans_dist[intermediate_state] = 1 - p0
                    return { successor: (probability, True) for (successor,probability) in trans_dist.items() }
            else:
                # implement all deterministic changes:
                # (none yet)

                # initialize a dictionary of possible successor states as keys and their probabilities as values,
                # which will subsequently be adjusted:
                trans_dist = { self._make_state(t + 1, target_loc, loc, imm_states, mc_locs, mv_locs, mv_states): 1 }  # stay in the same state with probability 1

                # implement all probabilistic changes:

                # again loop through all variable mobile objects encoded in mv_locs and mv_states:
                for i, object_type in enumerate(self.mobile_variable_object_types):
                    object_loc = get_loc(mv_locs, i) 
                    if object_type == 'F':  # a fragile object
                        if object_loc != (-2,-2) and self.move_probability_F > 0:  # object may move
                            # loop through all possible successor states in trans_dist and split them into at most 5 depending on whether F moves and where:
                            new_trans_dist = {}
                            for (successor, probability) in trans_dist.items():
                                succ_t, succ_loc, succ_prev_loc, succ_imm_states, succ_mc_locs, succ_mv_locs, succ_mv_states = self._extract_state_attributes(successor)
                                if object_loc == target_loc:  # object is destroyed
                                    default_successor = self._make_state(succ_t, succ_loc, succ_prev_loc, succ_imm_states, succ_mc_locs, set_loc(succ_mv_locs, i, (-2,-2)), succ_mv_states)
                                else:  # it stays in place
                                    default_successor = successor
                                direction_locs = [(direction, self._get_target_location(object_loc, direction)) 
                                                   for direction in range(4)]
                                direction_locs = [(direction, loc) for (direction, loc) in direction_locs 
                                                  if self._can_move(object_loc, loc, successor, who='F')]
                                n_directions = len(direction_locs)
                                if n_directions == 0:
                                    new_trans_dist[default_successor] = probability
                                else:
                                    new_trans_dist[default_successor] = probability * (1 - self.move_probability_F)
                                    p = probability * self.move_probability_F / n_directions
                                    for (direction, obj_target_loc) in direction_locs:
                                        if obj_target_loc == target_loc:  # object is destroyed
                                            new_successor = self._make_state(succ_t, succ_loc, succ_prev_loc, succ_imm_states, succ_mc_locs, set_loc(succ_mv_locs, i, (-2,-2)), succ_mv_states)
                                        else:  # it moves
                                            new_successor = self._make_state(succ_t, succ_loc, succ_prev_loc, succ_imm_states, succ_mc_locs, set_loc(succ_mv_locs, i, obj_target_loc), succ_mv_states)
                                        new_trans_dist[new_successor] = p
                            trans_dist = new_trans_dist
                            
                # TODO: update object states and/or object locations, e.g. if the agent picks up an object or moves an object

                return {successor: (probability, True) for (successor,probability) in trans_dist.items()} 

    @lru_cache(maxsize=None)
    def observation_and_reward_distribution(self, state, action, successor, n_samples = None):
        """
        Delta for a state accrues when entering the state, so it depends on successor:
        """
        if state is None and action is None:
            return {(self._make_state(), 0): (1, True)}
        t, loc, prev_loc, imm_states, mc_locs, mv_locs, mv_states = self._extract_state_attributes(successor)
        delta = self.time_deltas[t % self.time_deltas.size]
        if self.delta_xygrid[loc] in self.cell_code2delta:
            delta += self.cell_code2delta[self.delta_xygrid[loc]]
        if t == self.max_episode_length and self.xygrid[loc] != 'G':
            delta += self.timeout_delta
        return {(successor, delta): (1, True)}

    # reset() and step() are inherited from MDPWorldModel and use the above transition_distribution():

    def initial_state(self):
        return self._make_state()
    
    def reset(self, seed = None, options = None): 
        ret = super().reset(seed = seed, options = options)
        if self.render_mode == "human":
            self._render_frame()
        return ret

    def step(self, action):
        ret = super().step(action)
        if self.render_mode == "human":
            self._render_frame()
        return ret

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _init_human_rendering(self):
        pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
        self._cell_font = pygame.font.SysFont('Helvetica', 30)
        self._delta_font = pygame.font.SysFont('Helvetica', 10)

    def _render_frame(self):
        if self._window is None and self.render_mode == "human":
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (900,0)
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                self._window_shape
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self._window_shape)
        canvas.fill((255, 255, 255))
        pix_square_size = self._window_shape[0] / self.xygrid.shape[0]  # The size of a single grid square in pixels

        # Draw grid contents:
        for x in range(self.xygrid.shape[0]):
            for y in range(self.xygrid.shape[1]):
                cell_type = self.xygrid[x, y]
                if cell_type == "#" or (cell_type == "," and self._immobile_object_states[self.immobile_object_indices[x, y]] == 1):
                    pygame.draw.rect(
                        canvas,
                        (64, 64, 64),
                        (x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size),
                    )
                elif cell_type == "G":
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        (x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size),
                    )
                elif cell_type in render_as_char_types:
                    canvas.blit(self._cell_font.render(cell_type, True, (0, 0, 0)),
                                      ((x+.3) * pix_square_size, (y+.3) * pix_square_size))
                cell_code = self.delta_xygrid[x, y]
                if cell_code in self.cell_code2delta:
                    canvas.blit(self._delta_font.render(cell_code + f" {self.cell_code2delta[cell_code]}", True, (0, 0, 0)),
                                      ((x+.1) * pix_square_size, (y+.1) * pix_square_size))

        # Render all mobile objects:
        for i, object_type in enumerate(self.mobile_constant_object_types):
            x, y = get_loc(self._mobile_constant_object_locations, i)
            if object_type == 'X':  # a box
                pygame.draw.rect(
                    canvas,
                    (128, 128, 128),
                    ((x+.1) * pix_square_size, (y+.1) * pix_square_size, .8*pix_square_size, .8*pix_square_size),
                )

        for i, object_type in enumerate(self.mobile_variable_object_types):
            x, y = get_loc(self._mobile_variable_object_locations, i)
            if object_type == 'F':  # a fragile object
                pygame.draw.circle(
                    canvas,
                    (255, 0, 0),
                    ((x+.5) * pix_square_size, (y+.5) * pix_square_size),
                    pix_square_size / 4,
                )

        # Now we draw the agent and its previous location:
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.array(self._previous_agent_location) + 0.5) * pix_square_size,
            pix_square_size / 4,
            width = 3,
        )
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.array(self._agent_location) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.xygrid.shape[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self._window_shape[1]),
                width=3,
            )
        for y in range(self.xygrid.shape[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * y),
                (self._window_shape[0], pix_square_size * y),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self._fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
